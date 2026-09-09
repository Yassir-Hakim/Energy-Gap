"""Model fit statistics from ASHRAE Guideline 14"""

import numpy as np


def cvrmse(y, yhat, n_params=0):
    """Coefficient of variation of the RMSE, as a percentage.

    Root mean squared error expressed as a share of the mean reading, so
    buildings of different sizes are comparable. G14 hourly limit: 30%.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    resid = y - yhat
    degrees_of_freedom = len(y) - n_params
    return 100.0 * np.sqrt(np.sum(resid**2) / degrees_of_freedom) / np.mean(y)


def nmbe(y, yhat, n_params=0):
    """Normalised mean bias error, as a percentage.

    Whether the model is systematically high or low. Positive means the model
    under-predicts on average. G14 hourly limit: +/-10%.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    degrees_of_freedom = len(y) - n_params
    return 100.0 * np.sum(y - yhat) / (degrees_of_freedom * np.mean(y))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def resid_stats(y, yhat):
    """Residual spread and stickiness, for the gap uncertainty in section 5.6.

    Returns (sigma, rho): the standard deviation of the residuals, and their
    lag-1 autocorrelation. rho measures how much each hour's error resembles
    the previous hour's; it inflates the standard error of a summed gap.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    resid = y - yhat
    sigma = float(np.std(resid, ddof=1))
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return sigma, rho