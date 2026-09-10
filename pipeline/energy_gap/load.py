"""Reads the three BDG2 CSV files and loads them into Postgres.

This module decides WHAT goes into the database (which columns, which rows,
what to drop). db.py knows HOW to put it there.
"""

import pandas as pd

import energy_gap.config as config
from energy_gap.db import copy_frame, execute, run_sql_file


def init_db():
    """Create all the tables. Drops any existing ones first, so re-running is safe."""
    run_sql_file(config.SCHEMA_SQL)
    print("Tables created.")


def _flag(meta, column):
    """True where the metadata records that this meter type exists.

    BDG2 writes 'Yes' where a meter exists and leaves the cell blank otherwise,
    so "is this cell filled in" is the correct test. Verified against the real
    file: 1,578 buildings carry the electricity flag, which matches the meter
    file's 1,578 building columns exactly.
    """
    if column not in meta.columns:
        return pd.Series(False, index=meta.index)
    return meta[column].notna()


def load_metadata():
    """Load the 1,636 building records."""
    meta = pd.read_csv(config.METADATA_CSV)

    # Built in exactly the column order the buildings table expects.
    df = pd.DataFrame({
        "building_id": meta["building_id"],
        "site_id": meta["site_id"],
        "primary_use": meta.get("primaryspaceusage"),
        "sub_primary_use": meta.get("sub_primaryspaceusage"),
        "sqm": meta.get("sqm"),
        "sqft": meta.get("sqft"),
        "timezone": meta.get("timezone"),
        "year_built": meta.get("yearbuilt"),
        "floors": meta.get("numberoffloors"),
        "occupants": meta.get("occupants"),
        "lat": meta.get("lat"),
        "lng": meta.get("lng"),
        # Stated heating fuel. Filled in for 215 buildings (172 of them Gas)
        # and blank for the rest, which is why the meter flags below still
        # matter: a steam or hot-water meter also means heating is not electric.
        "heating_type": meta.get("heatingtype"),
        "has_electricity": _flag(meta, "electricity"),
        "has_chilledwater": _flag(meta, "chilledwater"),
        "has_steam": _flag(meta, "steam"),
        "has_hotwater": _flag(meta, "hotwater"),
        "has_gas": _flag(meta, "gas"),
    })

    # These are whole numbers, but pandas turns a column containing blanks into
    # floats (2001.0). Int64 is the integer type that allows blanks.
    # Measured: year_built is blank for 819 buildings, floors for 1,195.
    for col in ("year_built", "floors"):
        df[col] = df[col].astype("Int64")

    n = copy_frame(df, "buildings", list(df.columns))
    print(f"Loaded {n:,} buildings.")


def load_weather():
    """Load hourly weather for all 19 sites. Expect 331,166 rows."""
    w = pd.read_csv(config.WEATHER_CSV, parse_dates=["timestamp"])

    df = pd.DataFrame({
        "site_id": w["site_id"],
        "ts": w["timestamp"],
        "air_temp": w.get("airTemperature"),
        "dew_temp": w.get("dewTemperature"),
        "cloud_cover": w.get("cloudCoverage"),
        "wind_speed": w.get("windSpeed"),
        "wind_dir": w.get("windDirection"),
        "precip_1h": w.get("precipDepth1HR"),
        "pressure": w.get("seaLvlPressure"),
    })

    # The primary key is (site_id, ts), so a duplicate would be rejected and
    # abort the load. The real file has none, but local timestamps repeat an
    # hour when clocks go back, so this stays as cheap insurance.
    before = len(df)
    df = df.drop_duplicates(subset=["site_id", "ts"], keep="first")
    if before != len(df):
        print(f"  dropped {before - len(df):,} duplicate weather timestamps (clock changes)")

    n = copy_frame(df, "weather", list(df.columns))
    print(f"Loaded {n:,} weather rows.")


def load_meters(sites=None, limit=None):
    """Load the electricity meters, reshaping the file from wide to long.

    The CSV is 17,544 rows x 1,578 building columns: one column per building,
    one row per hour. The table needs one row per building per hour, so each
    column becomes up to 17,544 rows.

    Expect 25,212,579 rows.

    `sites` restricts to site prefixes (e.g. ['Fox']) and `limit` caps the
    number of buildings. Both exist so we can do a quick test run first.
    """
    wide = pd.read_csv(config.ELECTRICITY_CSV, parse_dates=["timestamp"])
    ts = wide["timestamp"]

    building_cols = [c for c in wide.columns if c != "timestamp"]
    if sites:
        building_cols = [c for c in building_cols if c.split("_")[0] in sites]
    if limit:
        building_cols = building_cols[:limit]

    print(f"Loading {len(building_cols):,} buildings...")
    total = 0
    empty = 0

    # One column at a time. Reshaping all 1,578 at once (pandas .melt) would
    # build a 27-million-row frame in memory before a single row reached the
    # database.
    for i, col in enumerate(building_cols, start=1):
        df = pd.DataFrame({"building_id": col, "ts": ts, "kwh": wide[col]})

        # A blank cell is a reading the dataset authors removed, not a zero.
        # Storing it as 0 would tell the model the building switched off.
        df = df.dropna(subset=["kwh"])

        # Guards the (building_id, ts) primary key added at the end.
        df = df.drop_duplicates(subset=["ts"], keep="first")

        if df.empty:
            # A handful of columns are blank for the entire two years.
            empty += 1
            continue

        total += copy_frame(df, "meter_readings", ["building_id", "ts", "kwh"])

        if i % 200 == 0:
            print(f"  {i:,}/{len(building_cols):,} buildings, {total:,} rows so far")

    if empty:
        print(f"  {empty} building column(s) were entirely blank and were skipped")

    # Add the index now that loading has finished. Creating it first would mean
    # updating the index once per row instead of building it once at the end.
    print("Building the index...")
    execute("ALTER TABLE meter_readings ADD PRIMARY KEY (building_id, ts)")
    print(f"Loaded {total:,} meter readings.")