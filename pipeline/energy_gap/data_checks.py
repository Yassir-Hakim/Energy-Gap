"""Four sanity checks on the loaded data. Reads only, changes nothing.

The point is to prove the load is right before anything gets modelled. Check 3
is the one that matters most: a site whose weather is not in local time would
produce wrong temperature coefficients for every building on it, without ever
raising an error.
"""

import energy_gap.config as config
from energy_gap.db import query_df


def check_data():
    _row_counts()
    _coverage()
    _weather_alignment()
    _unit_sanity()


def _row_counts():
    print("\n1. Row counts and span")
    print("-" * 52)
    counts = query_df("""
        SELECT 'buildings' AS name, count(*) AS n FROM buildings
        UNION ALL SELECT 'weather', count(*) FROM weather
        UNION ALL SELECT 'meter_readings', count(*) FROM meter_readings
    """)
    for _, r in counts.iterrows():
        print(f"  {r['name']:<20} {r['n']:>14,}")
    print("  expected:            1,636 / 331,166 / 25,212,579")

    span = query_df("SELECT min(ts) AS first_ts, max(ts) AS last_ts FROM meter_readings")
    print(f"  span: {span['first_ts'][0]}  ->  {span['last_ts'][0]}")


def _coverage():
    print("\n2. Coverage per building")
    print("-" * 52)
    # 17,544 hours across 2016 and 2017. What share does each building have?
    cov = query_df(f"""
        SELECT building_id, 100.0 * count(*) / {config.TOTAL_HOURS} AS pct
        FROM meter_readings
        GROUP BY building_id
    """)
    print(f"  buildings with any data: {len(cov):,}   (expected about 1,570)")
    print(f"  min {cov['pct'].min():.1f}%    median {cov['pct'].median():.1f}%    max {cov['pct'].max():.1f}%")

    # Counted from the buildings table outwards, not from meter_readings.
    # Grouping meter_readings would miss the 46 buildings that have an
    # electricity flag but no baseline-year rows at all: they produce no group,
    # so a HAVING clause never sees them. score.py must count the same way.
    baseline = query_df(f"""
        SELECT count(*) AS n
        FROM buildings b
        WHERE b.has_electricity
          AND COALESCE((
                SELECT count(*) FROM meter_readings m
                WHERE m.building_id = b.building_id
                  AND extract(year FROM m.ts) = {config.BASELINE_YEAR}
              ), 0) < {config.MIN_BASELINE_HOURS}
    """)
    print(f"  below {config.MIN_BASELINE_HOURS:,} baseline hours -> insufficient_data: "
          f"{baseline['n'][0]:,}   (expected 88: 42 sparse + 46 with no 2016 data)")


def _weather_alignment():
    print("\n3. Weather alignment  [THE IMPORTANT ONE]")
    print("-" * 52)
    print("  July temperature should peak mid-afternoon at every site. A site")
    print("  peaking near midnight means its weather is not in local time,")
    print("  which would silently corrupt every temperature coefficient.")
    print()
    peak = query_df("""
        WITH hourly AS (
          SELECT site_id, extract(hour FROM ts) AS hr, avg(air_temp) AS t
          FROM weather
          WHERE extract(month FROM ts) = 7 AND air_temp IS NOT NULL
          GROUP BY site_id, hr
        ),
        ranked AS (
          SELECT site_id, hr,
                 row_number() OVER (PARTITION BY site_id ORDER BY t DESC) AS rn
          FROM hourly
        )
        SELECT site_id, hr AS peak_hour FROM ranked WHERE rn = 1 ORDER BY site_id
    """)
    bad = []
    for _, r in peak.iterrows():
        hour = int(r["peak_hour"])
        ok = 13 <= hour <= 18
        if not ok:
            bad.append(r["site_id"])
        print(f"  {r['site_id']:<12} peaks at {hour:02d}:00   {'ok' if ok else '<-- SUSPICIOUS'}")
    print()
    print(f"  suspicious sites: {bad if bad else 'none'}")


def _unit_sanity():
    print("\n4. Unit sanity")
    print("-" * 52)
    print("  Hourly kWh should sit in the tens to low thousands.")
    pct = query_df("""
        SELECT percentile_cont(0.01) WITHIN GROUP (ORDER BY kwh) AS p1,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY kwh) AS p50,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY kwh) AS p99,
               min(kwh) AS lo, max(kwh) AS hi
        FROM meter_readings
    """)
    r = pct.iloc[0]
    print(f"  1st {r['p1']:,.2f}    median {r['p50']:,.2f}    99th {r['p99']:,.2f}")
    # Four decimals: the true minimum in this dataset is 0.0001, which rounds
    # to 0.0 at one decimal place and looks alarmingly like a dead meter.
    print(f"  min {r['lo']:,.4f}    max {r['hi']:,.2f}")
    if r["lo"] < 0:
        print("  WARNING: negative readings present (none expected in this dataset)")
    print()