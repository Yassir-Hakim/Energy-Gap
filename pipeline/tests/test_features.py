"""Checks for the feature builders, against values worked out by hand.

The failure mode these guard against: a wrong hour-of-week offset or a wrong
hinge boundary does not crash. It produces a model that fits plausible-looking
numbers to the wrong thing, and the error only shows up as a building being
wrongly flagged months later. Same reason the metrics tests exist.
"""

import numpy as np
import pandas as pd
import pytest

from energy_gap.features import (
    build_model_frame,
    gbm_features,
    holdout_mask,
    hour_of_week,
    rolling_temp_24h,
    temp_hinges,
    towt_design,
)


# --- hour_of_week ----------------------------------------------------------

def test_hour_of_week_anchors():
    """2016-01-04 was a Monday, so Monday 00:00 must be 0 and Sunday 23:00 167."""
    ts = pd.to_datetime([
        "2016-01-04 00:00",   # Monday midnight  -> 0
        "2016-01-04 01:00",   # Monday 1am       -> 1
        "2016-01-04 23:00",   # Monday 11pm      -> 23
        "2016-01-05 00:00",   # Tuesday midnight -> 24
        "2016-01-10 23:00",   # Sunday 11pm      -> 167
    ])
    assert list(hour_of_week(ts)) == [0, 1, 23, 24, 167]


def test_hour_of_week_stays_in_range():
    ts = pd.date_range("2016-01-01", "2016-03-01", freq="h")
    how = hour_of_week(ts)
    assert how.min() == 0
    assert how.max() == 167


def test_hour_of_week_repeats_weekly():
    """The same clock hour one week later must give the same value."""
    a = pd.to_datetime(["2016-06-15 14:00"])
    b = pd.to_datetime(["2016-06-22 14:00"])
    assert hour_of_week(a)[0] == hour_of_week(b)[0]


# --- temp_hinges -----------------------------------------------------------

def test_temp_hinges_at_22_degrees():
    """At 22C: knots 5, 10, 15, 20 are passed; 25 and 30 are not."""
    out = temp_hinges([22.0], knots=[5, 10, 15, 20, 25, 30])
    assert out["temp"][0] == 22.0
    assert out["temp_hinge_5"][0] == pytest.approx(17.0)
    assert out["temp_hinge_10"][0] == pytest.approx(12.0)
    assert out["temp_hinge_15"][0] == pytest.approx(7.0)
    assert out["temp_hinge_20"][0] == pytest.approx(2.0)
    assert out["temp_hinge_25"][0] == pytest.approx(0.0)
    assert out["temp_hinge_30"][0] == pytest.approx(0.0)


def test_temp_hinges_are_zero_below_every_knot():
    """Below the lowest knot the response is a single straight line."""
    out = temp_hinges([-5.0], knots=[5, 10, 15, 20, 25, 30])
    hinge_cols = [c for c in out.columns if c.startswith("temp_hinge_")]
    assert (out[hinge_cols].to_numpy() == 0).all()


def test_temp_hinge_is_exactly_zero_at_its_own_knot():
    """The knot is the boundary: a hinge turns on above it, not at it."""
    out = temp_hinges([20.0], knots=[20])
    assert out["temp_hinge_20"][0] == pytest.approx(0.0)


def test_temp_hinges_column_count():
    """One plain temperature column plus one per knot. Six knots -> 7 columns."""
    out = temp_hinges([12.0, 18.0], knots=[5, 10, 15, 20, 25, 30])
    assert out.shape == (2, 7)


def test_hinges_reproduce_a_known_bent_line():
    """A bent line built from hinges must equal one built by hand.

    Slope -2 below 15C, then flat: coefficients b0 = -2 and b_15 = +2, because
    each hinge coefficient is the CHANGE in slope, and -2 + 2 = 0.
    """
    temps = np.array([0.0, 10.0, 15.0, 20.0, 30.0])
    out = temp_hinges(temps, knots=[15])
    predicted = 100.0 + (-2.0) * out["temp"] + 2.0 * out["temp_hinge_15"]
    expected = [100.0, 80.0, 70.0, 70.0, 70.0]   # falls to 15C, flat after
    assert list(predicted) == pytest.approx(expected)


# --- towt_design -----------------------------------------------------------

def test_towt_design_is_175_columns():
    """168 hour-of-week columns + 7 temperature columns = the p in n - p."""
    ts = pd.date_range("2016-01-04", periods=200, freq="h")
    frame = pd.DataFrame({"ts": ts, "air_temp": np.linspace(0, 30, 200)})
    assert towt_design(frame).shape[1] == 175


def test_towt_design_has_all_168_columns_even_for_a_short_period():
    """A building with two days of data must still produce 168 columns.

    Otherwise two buildings' coefficient vectors would mean different things.
    """
    ts = pd.date_range("2016-01-04", periods=48, freq="h")
    frame = pd.DataFrame({"ts": ts, "air_temp": np.full(48, 10.0)})
    design = towt_design(frame)
    assert [c for c in design.columns if c.startswith("how_")] == [
        f"how_{i}" for i in range(168)
    ]


def test_towt_design_is_one_hot():
    """Exactly one hour-of-week column is 1 in each row."""
    ts = pd.date_range("2016-01-04", periods=100, freq="h")
    frame = pd.DataFrame({"ts": ts, "air_temp": np.full(100, 10.0)})
    design = towt_design(frame)
    how_cols = [c for c in design.columns if c.startswith("how_")]
    assert (design[how_cols].sum(axis=1) == 1).all()


# --- rolling_temp_24h ------------------------------------------------------

def test_rolling_temp_24h_of_a_constant_is_that_constant():
    out = rolling_temp_24h(np.full(50, 12.0))
    assert out.to_numpy() == pytest.approx(np.full(50, 12.0))


def test_rolling_temp_24h_first_row_has_no_history():
    """With one row of history the mean is just that row, not a blank."""
    out = rolling_temp_24h([5.0, 15.0])
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(10.0)


# --- holdout_mask ----------------------------------------------------------

def test_holdout_mask_selects_whole_weeks_not_scattered_hours():
    ts = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    mask = holdout_mask(ts, every_nth=4)
    weeks = pd.Series(ts).dt.isocalendar().week.astype(int)
    held = set(weeks[mask])
    kept = set(weeks[~mask])
    assert held and not (held & kept)   # no week is split between the two


def test_holdout_mask_holds_out_roughly_a_quarter():
    ts = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    assert 0.2 < holdout_mask(ts, every_nth=4).mean() < 0.3


# --- build_model_frame -----------------------------------------------------

def _weather(n, start="2016-01-01"):
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "ts": ts,
        "air_temp": np.linspace(0, 20, n),
        "dew_temp": np.linspace(-2, 15, n),
        "cloud_cover": np.zeros(n),
        "wind_speed": np.full(n, 3.0),
    })


def test_build_model_frame_keeps_only_hours_with_both():
    """A meter reading with no weather, or weather with no reading, is dropped."""
    w = _weather(48)
    r = pd.DataFrame({"ts": w["ts"][:30], "kwh": np.full(30, 100.0)})
    out = build_model_frame(r, w)
    assert len(out) == 30
    assert out["kwh"].notna().all()
    assert out["air_temp"].notna().all()


def test_build_model_frame_forward_fills_short_weather_gaps_only():
    """A 3-hour hole is filled; a 20-hour hole is not."""
    w = _weather(72)
    w.loc[10:12, "air_temp"] = np.nan     # 3 hours, under the 6-hour limit
    w.loc[40:59, "air_temp"] = np.nan     # 20 hours, over it
    r = pd.DataFrame({"ts": w["ts"], "kwh": np.full(72, 100.0)})
    out = build_model_frame(r, w)
    kept = set(out["ts"])
    assert w["ts"][11] in kept        # short gap survived
    assert w["ts"][50] not in kept    # long gap was dropped


def test_build_model_frame_never_invents_meter_readings():
    """Missing kwh is dropped, never forward-filled. A blank is not a zero."""
    w = _weather(48)
    r = pd.DataFrame({"ts": w["ts"], "kwh": np.full(48, 100.0)})
    r.loc[20:24, "kwh"] = np.nan
    out = build_model_frame(r, w)
    assert len(out) == 43
    assert (out["kwh"] == 100.0).all()


def test_build_model_frame_filters_by_year():
    w = _weather(48, start="2016-12-31 12:00")   # straddles new year
    r = pd.DataFrame({"ts": w["ts"], "kwh": np.full(48, 100.0)})
    out = build_model_frame(r, w, year=2016)
    assert (pd.to_datetime(out["ts"]).dt.year == 2016).all()


def test_build_model_frame_survives_empty_input():
    """A building with no baseline data must return an empty frame, not crash.

    46 buildings in the real dataset have an electricity flag but no 2016 rows.
    """
    w = _weather(24)
    r = pd.DataFrame({"ts": pd.to_datetime([]), "kwh": []})
    out = build_model_frame(r, w)
    assert out.empty


# --- gbm_features ----------------------------------------------------------

def test_gbm_features_marks_weekends():
    """2016-01-09 was a Saturday, 2016-01-11 a Monday."""
    frame = pd.DataFrame({
        "ts": pd.to_datetime(["2016-01-09 12:00", "2016-01-11 12:00"]),
        "air_temp": [10.0, 10.0], "dew_temp": [5.0, 5.0],
        "cloud_cover": [0.0, 0.0], "wind_speed": [3.0, 3.0],
    })
    out = gbm_features(frame)
    assert list(out["is_weekend"]) == [1, 0]


def test_gbm_features_has_no_hinge_columns():
    """Trees find interactions themselves; hinges are a linear-model device."""
    ts = pd.date_range("2016-01-04", periods=48, freq="h")
    frame = pd.DataFrame({
        "ts": ts, "air_temp": np.full(48, 10.0), "dew_temp": np.full(48, 5.0),
        "cloud_cover": np.zeros(48), "wind_speed": np.full(48, 3.0),
    })
    out = gbm_features(frame)
    assert not [c for c in out.columns if "hinge" in c]