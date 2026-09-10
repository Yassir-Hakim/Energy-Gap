import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from energy_gap import config
from energy_gap.features import gbm_features, towt_design

# Two models per building, deliberately different in kind:
#
#   TOWT  time-of-week + bent temperature response. The industry standard for
#         measurement and verification. Every coefficient is readable, and it
#         extrapolates beyond the temperatures it was trained on.
#   GBM   gradient boosting. Finds interactions on its own, usually fits
#         better, and cannot extrapolate at all.

class FittedModel:
    # Holds a fitted estimator together with the feature builder it was trained
    # on. Pairing them means score.py can call model.predict(frame) on either
    # model without knowing which one it is holding.

    def __init__(self, name, estimator, build_features, n_params, feature_names=None):
        self.name = name                    # 'towt' or 'gbm'
        self.estimator = estimator
        self.build_features = build_features
        self.n_params = n_params            # the p in the G14 (n - p) denominator
        self.feature_names = feature_names

    def predict(self, frame):
        X = self.build_features(frame)
        return self.estimator.predict(X)

    def coefficients(self):
        # TOWT only. The temperature coefficients are the building's heating
        # and cooling sensitivity in kWh per degree per hour, which is a number
        # an engineer can act on.
        if self.name != "towt":
            return {}
        return {n: float(c) for n, c in zip(self.feature_names, self.estimator.coef_)}

    def params(self):
        # Written to the params jsonb column. Coefficients for TOWT, settings
        # for GBM, which is 300 trees and has nothing readable to store.
        if self.name == "towt":
            return self.coefficients()
        return dict(config.GBM_PARAMS)


def fit_towt(frame):
    # 168 one-hot hour-of-week columns + 7 temperature columns = 175 parameters.
    # Read the fitted model as a weekly profile of 168 baseline levels, plus a
    # bent temperature response.
    X = towt_design(frame)
    y = frame["kwh"].to_numpy(dtype=float)

    #
    # fit_intercept=False because the 168 one-hot columns already cover every
    # row. An intercept would make the fit ambiguous: any constant could shift
    # between it and all 168 coefficients without changing a prediction.
    model = Ridge(alpha=config.RIDGE_ALPHA, fit_intercept=False)
    model.fit(X, y)

    return FittedModel("towt", model, towt_design,
                       n_params=X.shape[1], feature_names=list(X.columns))


def fit_gbm(frame):
    # Same algorithm family as LightGBM, which won the ASHRAE Great Energy
    # Predictor III competition on this dataset.

    #Hist for speed on large datasets and deals with missing valyes
    X = gbm_features(frame)
    y = frame["kwh"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(**config.GBM_PARAMS)
    model.fit(X, y)

    return FittedModel("gbm", model, gbm_features, n_params=1)


def fit_both(frame):
    return {"towt": fit_towt(frame), "gbm": fit_gbm(frame)}


def temperature_response(model, temps, hour_of_week=33):
    # What the fitted TOWT model says about temperature, as a curve for the
    # building page. Holds the hour of week fixed (default 33 = Tuesday 9am, a
    # typical occupied hour) and sweeps temperature.
    #
    # The clearest single view of what the model learned. A V shape means the
    # building heats and cools on electricity. A flat left arm rising on the
    # right means heating runs on another fuel.
    if model.name != "towt":
        raise ValueError("temperature_response is only defined for TOWT")

    coef = model.coefficients()
    temps = np.asarray(temps, dtype=float)

    # Each hinge coefficient is the CHANGE in slope at its knot, so the slope
    # at any temperature is coef['temp'] plus every hinge already passed.
    out = coef.get(f"how_{hour_of_week}", 0.0) + coef["temp"] * temps
    for k in config.TEMP_KNOTS_C:
        out = out + coef[f"temp_hinge_{k}"] * np.maximum(0.0, temps - k)
    return out