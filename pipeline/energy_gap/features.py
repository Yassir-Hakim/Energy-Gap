"""feature engineering for the models to learn from
"""

import numpy as np
import pandas as pd

import energy_gap.config as config


def hour_of_week(timestamps):
    """Turn timestamps into an integer 0-167: which hour of the week is this.

    Monday 00:00 is 0, Monday 01:00 is 1, ... Sunday 23:00 is 167.

    This one column is how the model learns a building's weekly rhythm

    Why not hour and weekday as two columns: a linear model would then have to
    add them, so "Monday at 9am" would be forced to equal "Monday effect" plus
    "9am effect". Real buildings do not work that way -- 9am on a Saturday is
    nothing like 9am on a Monday. One column of 168 values lets every hour of
    the week stand on its own.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    # dayofweek: Monday is 0, Sunday is 6. So Monday 00:00 -> 0*24+0 = 0
    # and Sunday 23:00 -> 6*24+23 = 167.
    return (ts.dt.dayofweek * 24 + ts.dt.hour).astype(int)


def temp_hinges(temps, knots=None):
    """Turn one temperature column into a bent-line temperature response.

    Returns a DataFrame with one plain temperature column plus one hinge column
    per knot. A hinge is max(0, T - knot): flat at zero below the knot, rising
    one-for-one above it.

    Adding one hinge column to a regression buys one permitted change of slope,
    at that temperature. Six knots give seven straight segments, which is
    enough to trace the V shape of a real building -- heating when cold, a flat
    deadband, cooling when hot -- without fitting an actual curve.

    The point of doing it this way: the model stays LINEAR IN ITS PARAMETERS.
    Only the columns are nonlinear in temperature, so ordinary least squares
    still solves it in one shot, which is what makes fitting 1,578 buildings
    fast enough to be practical.

    Each fitted coefficient is the CHANGE in slope at that knot, not the slope
    itself. 
    """
    if knots is None:
        knots = config.TEMP_KNOTS_C

    t = pd.Series(temps, dtype=float).reset_index(drop=True)

    out = pd.DataFrame({"temp": t})
    for k in knots:
        # np.maximum, not np.max: this is elementwise across the column,
        # not the single largest value in it.
        out[f"temp_hinge_{k}"] = np.maximum(0.0, t - k)
    return out


def rolling_temp_24h(temps):
    """Mean outside temperature over the previous 24 hours.

    A proxy for the heat stored in the building's fabric.
    """
    t = pd.Series(temps, dtype=float).reset_index(drop=True)
    return t.rolling(window=24, min_periods=1).mean()


def towt_design(frame):
    """Build the exact input matrix the TOWT regression is fitted on.

    Returns 168 hour-of-week columns plus 7 temperature columns: 175 in total,
    which is the p that goes into the Guideline 14 degrees-of-freedom
    correction.

    The 168 columns are one-hot: exactly one is 1 in each row and the rest are
    0. Its coefficient is that hour's baseline consumption. There is no
    intercept -- the 168 columns already cover every row, so adding one would
    make the fit ambiguous (any amount could be shuffled between the intercept
    and all 168 coefficients without changing a single prediction).

    reindex is what guarantees all 168 columns exist even when the building's
    data never covers some hour of the week. A missing column here would make
    two buildings' coefficient vectors mean different things.
    """
    how = hour_of_week(frame["ts"])
    dummies = pd.get_dummies(how, prefix="how").reindex(
        columns=[f"how_{i}" for i in range(168)], fill_value=0
    ).astype(float)

    temps = temp_hinges(frame["air_temp"])

    design = pd.concat([dummies.reset_index(drop=True),
                        temps.reset_index(drop=True)], axis=1)
    return design


def gbm_features(frame):
    """Build the wider feature set for the gradient-boosting model.

    Gradient boosting finds interactions on its own, so it does not need the
    hinge columns: it can learn a different temperature response at 2pm than at
    2am without being handed the shape. It is given the raw ingredients and
    left to work out the structure.

    Extra weather columns appear here and not in TOWT on purpose. Dew point
    tracks air temperature closely, so in a linear model the two would fight
    over the same signal and produce unstable, uninterpretable coefficients.
    Trees do not care.
    """
    how = hour_of_week(frame["ts"])
    ts = pd.to_datetime(frame["ts"]).reset_index(drop=True)

    return pd.DataFrame({
        "hour_of_week": how,
        "hour": ts.dt.hour,
        "weekday": ts.dt.dayofweek,
        "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
        "air_temp": pd.Series(frame["air_temp"]).reset_index(drop=True),
        "dew_temp": pd.Series(frame["dew_temp"]).reset_index(drop=True),
        "cloud_cover": pd.Series(frame["cloud_cover"]).reset_index(drop=True),
        "wind_speed": pd.Series(frame["wind_speed"]).reset_index(drop=True),
        "temp_24h": rolling_temp_24h(frame["air_temp"]),
    })


def holdout_mask(timestamps, every_nth=None):
    """True on every Nth ISO week. Those weeks are held out of the fit.

    Why hold anything out: a model with 175 parameters can fit noise. If it
    scores well on the hours it was trained on but badly on weeks it never
    saw, it memorised rather than learned, and its 2017 predictions cannot be
    trusted.

    After measuring, the model is refitted on the complete baseline year for
    scoring. The holdout exists to produce an honesty statistic, not to
    withhold data from the final model.
    """
    if every_nth is None:
        every_nth = config.HOLDOUT_EVERY_NTH_WEEK

    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    iso_week = ts.dt.isocalendar().week.astype(int)
    return (iso_week % every_nth == 0).to_numpy()


def build_model_frame(readings, weather, year=None):
    """Join one building's meter readings to its site weather, ready to model.

    `readings` needs columns ts and kwh; `weather` needs ts, air_temp,
    dew_temp, cloud_cover and wind_speed. Both are for a single building and
    its single site.

    Three things happen here, in this order, and the order matters:

    1. Reindex onto a complete hourly range. Both inputs have gaps, and
       rolling_temp_24h counts rows rather than clock time -- with gaps left
       in, "the previous 24 rows" could silently span several days.
    2. Forward-fill weather across short gaps only. A temperature from six
       hours ago is a fair guess; one from three days ago is a fabrication,
       so the limit comes from config rather than being unbounded.
    3. Drop rows still missing either a reading or a temperature. Blanks were
       never filled with zeros at load time and are not invented now: a
       missing reading is not a reading of nothing.

    Returns a frame with ts, kwh, air_temp, dew_temp, cloud_cover, wind_speed
    and temp_24h, sorted by time, with no missing values in the columns the
    models use.
    """
    r = readings.copy()
    w = weather.copy()
    r["ts"] = pd.to_datetime(r["ts"])
    w["ts"] = pd.to_datetime(w["ts"])

    if year is not None:
        r = r[r["ts"].dt.year == year]
        w = w[w["ts"].dt.year == year]

    if r.empty or w.empty:
        return pd.DataFrame(columns=["ts", "kwh", "air_temp", "dew_temp",
                                     "cloud_cover", "wind_speed", "temp_24h"])

    # Step 1: a continuous hourly index covering the period, so that "the
    # previous 24 rows" always means the previous 24 hours.
    full = pd.date_range(min(r["ts"].min(), w["ts"].min()),
                         max(r["ts"].max(), w["ts"].max()), freq="h")

    r = r.set_index("ts").reindex(full)
    w = w.set_index("ts").reindex(full)

    df = pd.DataFrame({
        "ts": full,
        "kwh": r["kwh"].to_numpy(),
        "air_temp": w["air_temp"].to_numpy(),
        "dew_temp": w["dew_temp"].to_numpy(),
        "cloud_cover": w["cloud_cover"].to_numpy(),
        "wind_speed": w["wind_speed"].to_numpy(),
    })

    # Step 2: short weather gaps only. Meter readings are never filled.
    limit = config.WEATHER_FFILL_LIMIT_HOURS
    for col in ("air_temp", "dew_temp", "cloud_cover", "wind_speed"):
        df[col] = df[col].ffill(limit=limit)

    # Computed before dropping rows, so the 24-hour window is a real 24 hours.
    df["temp_24h"] = rolling_temp_24h(df["air_temp"])

    # Step 3: the models need both a target and a temperature.
    df = df.dropna(subset=["kwh", "air_temp"]).reset_index(drop=True)
    return df