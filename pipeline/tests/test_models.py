"""Checks for the two models.

The central test builds a synthetic building whose weekly profile and
temperature response are known exactly, then asks whether the model recovers
them. That is a far stronger check than "does it run": a wrong hour-of-week
offset or a wrong hinge boundary still produces a model that fits plausible
numbers, and only a known-answer test catches it.
"""

import numpy as np
import pandas as pd
import pytest

from energy_gap.features import hour_of_week
from energy_gap.metrics import cvrmse
from energy_gap.models import (
    fit_both,
    fit_gbm,
    fit_towt,
    temperature_response,
)

# The truth the synthetic building is built from, so the tests can check
# against it rather than against magic numbers buried in the setup.
TRUE_BASE_KWH = 60.0        # overnight and weekend baseload
TRUE_OCCUPIED_KWH = 40.0    # extra during office hours
TRUE_COLD_SLOPE = -1.5      # kWh per degree below 15C (heating)
TRUE_HOT_SLOPE = 2.5        # kWh per degree above 20C (cooling)
NOISE_SD = 4.0


def synthetic_building(n_hours=8760, seed=0, noise_sd=NOISE_SD):
    """A building with a known office-hours profile and a known bent response.

    Weekdays 08:00-18:00 draw an extra 40 kWh. Consumption rises as it gets
    colder below 15C and as it gets hotter above 20C, and is flat between --
    the V shape a real building traces.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2016-01-01", periods=n_hours, freq="h")

    how = hour_of_week(ts).to_numpy()
    hour = how % 24
    day = how // 24
    occupied = (hour >= 8) & (hour < 18) & (day < 5)
    weekly = TRUE_BASE_KWH + TRUE_OCCUPIED_KWH * occupied

    # A year of temperature: seasonal swing plus hourly scatter.
    temp = 12 + 12 * np.sin(np.arange(n_hours) / 8760 * 2 * np.pi)
    temp = temp + rng.normal(0, 3, n_hours)

    response = (TRUE_COLD_SLOPE * np.minimum(temp, 15)
                + TRUE_HOT_SLOPE * np.maximum(0, temp - 20)
                - TRUE_COLD_SLOPE * 15)

    kwh = weekly + response + rng.normal(0, noise_sd, n_hours)

    return pd.DataFrame({
        "ts": ts,
        "kwh": kwh,
        "air_temp": temp,
        "dew_temp": temp - 3,
        "cloud_cover": rng.uniform(0, 8, n_hours),
        "wind_speed": rng.uniform(0, 10, n_hours),
        "temp_24h": pd.Series(temp).rolling(24, min_periods=1).mean(),
    })


# --- the central test ------------------------------------------------------

def test_towt_recovers_a_synthetic_building():
    """CV(RMSE) must be under 5%, close to the noise floor.

    The noise floor is the best any model could do: NOISE_SD as a share of the
    mean. Landing near it means the model recovered everything except the noise
    that was deliberately added.
    """
    frame = synthetic_building()
    model = fit_towt(frame)
    pred = model.predict(frame)

    score = cvrmse(frame["kwh"], pred, n_params=model.n_params)
    floor = 100 * NOISE_SD / frame["kwh"].mean()

    assert score < 5.0
    assert score < floor * 1.2   # within 20% of the best possible


def test_towt_recovers_the_temperature_slopes():
    """The fitted bend must match the slopes the building was built with.

    Each hinge coefficient is the CHANGE in slope at its knot, so the slope in
    any band is the plain temperature coefficient plus every hinge coefficient
    whose knot has been passed.
    """
    frame = synthetic_building()
    model = fit_towt(frame)
    coef = dict(zip(model.feature_names, model.estimator.coef_))

    cold = coef["temp"]                                    # below 5C
    mid = cold + coef["temp_hinge_5"] + coef["temp_hinge_10"] + coef["temp_hinge_15"]
    hot = mid + coef["temp_hinge_20"] + coef["temp_hinge_25"] + coef["temp_hinge_30"]

    assert cold == pytest.approx(TRUE_COLD_SLOPE, abs=0.3)
    assert mid == pytest.approx(0.0, abs=0.3)
    assert hot == pytest.approx(TRUE_HOT_SLOPE, abs=0.5)


def test_towt_recovers_the_weekly_profile():
    """Occupied hours must sit about 40 kWh above unoccupied ones."""
    frame = synthetic_building()
    model = fit_towt(frame)
    coef = dict(zip(model.feature_names, model.estimator.coef_))

    # Tuesday 10am is occupied; Tuesday 3am and Sunday 10am are not.
    occupied = coef["how_34"]        # day 1, hour 10
    overnight = coef["how_27"]       # day 1, hour 3
    weekend = coef["how_154"]        # day 6, hour 10

    assert occupied - overnight == pytest.approx(TRUE_OCCUPIED_KWH, abs=4)
    assert occupied - weekend == pytest.approx(TRUE_OCCUPIED_KWH, abs=4)


# --- shape and interface ---------------------------------------------------

def test_towt_has_175_parameters():
    """168 hour-of-week + 7 temperature. This is the p in the G14 denominator."""
    assert fit_towt(synthetic_building(n_hours=2000)).n_params == 175


def test_both_models_share_one_interface():
    """score.py loops over both without branching, so both must behave alike."""
    frame = synthetic_building(n_hours=3000)
    models = fit_both(frame)

    assert set(models) == {"towt", "gbm"}
    for name, model in models.items():
        pred = model.predict(frame)
        assert len(pred) == len(frame)
        assert np.isfinite(pred).all()
        assert model.name == name
        assert isinstance(model.params(), dict)


def test_gbm_fits_at_least_as_well_as_towt():
    """The flexible model should win on training data. If it loses, check why."""
    frame = synthetic_building()
    models = fit_both(frame)
    scores = {n: cvrmse(frame["kwh"], m.predict(frame), m.n_params)
              for n, m in models.items()}
    assert scores["gbm"] <= scores["towt"] + 1.0


def test_params_are_json_safe():
    """They are written to a jsonb column, so no numpy types may leak through."""
    import json
    frame = synthetic_building(n_hours=2000)
    for model in fit_both(frame).values():
        json.dumps(model.params())   # raises if anything is not JSON-safe


# --- extrapolation, the difference that matters ----------------------------

def test_gbm_cannot_extrapolate_but_towt_can():
    """Beyond the temperatures seen in training, the two behave differently.

    Trees predict as if the hottest training hour were the ceiling. TOWT keeps
    going along its last fitted slope. Neither is simply right, but the
    difference has to be known: a scoring year hotter than the baseline year is
    exactly where GBM quietly under-predicts.
    """
    frame = synthetic_building()
    models = fit_both(frame)

    hottest = frame["air_temp"].max()
    beyond = pd.DataFrame({
        "ts": pd.to_datetime(["2016-07-05 14:00"] * 2),
        "air_temp": [hottest, hottest + 15],
        "dew_temp": [hottest - 3, hottest + 12],
        "cloud_cover": [2.0, 2.0],
        "wind_speed": [3.0, 3.0],
        "temp_24h": [hottest, hottest + 15],
    })

    towt = models["towt"].predict(beyond)
    gbm = models["gbm"].predict(beyond)

    assert towt[1] > towt[0] + 10        # keeps climbing
    assert gbm[1] == pytest.approx(gbm[0], abs=5)   # flattens off


# --- a building whose heating is not electric ------------------------------

def test_cold_side_is_flat_when_heating_is_not_electric():
    """Gas-heated buildings must fit a hockey stick, not a V.

    Nothing tells the model the fuel. If no cold-weather signal exists in the
    meter, the low hinges should simply come out near zero -- the hinges permit
    a heating response, they do not impose one.
    """
    frame = synthetic_building()
    # Rebuild with cooling only: no response below 20C.
    frame["kwh"] = (frame["kwh"]
                    - (TRUE_COLD_SLOPE * np.minimum(frame["air_temp"], 15)
                       - TRUE_COLD_SLOPE * 15))

    model = fit_towt(frame)
    coef = dict(zip(model.feature_names, model.estimator.coef_))
    cold_slope = coef["temp"]
    hot_slope = cold_slope + sum(coef[f"temp_hinge_{k}"] for k in (5, 10, 15, 20, 25, 30))

    assert abs(cold_slope) < 0.5      # flat where there is no heating signal
    assert hot_slope > 1.5            # still finds the cooling


# --- temperature_response --------------------------------------------------

def test_temperature_response_traces_the_fitted_curve():
    frame = synthetic_building()
    model = fit_towt(frame)
    curve = temperature_response(model, [0, 5, 10, 15, 20, 25, 30, 35])

    assert curve[0] > curve[3]    # falls from 0C to 15C   (heating)
    assert curve[7] > curve[4]    # rises from 20C to 35C  (cooling)


def test_temperature_response_rejects_gbm():
    """A tree ensemble has no closed-form response curve to trace."""
    model = fit_gbm(synthetic_building(n_hours=2000))
    with pytest.raises(ValueError):
        temperature_response(model, [10, 20])