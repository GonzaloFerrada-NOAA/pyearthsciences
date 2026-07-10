from types import SimpleNamespace

import numpy as np


def _fmt_num(x: float, ndp: int = 3, is_percent: bool = False) -> str:
    """Format a metric value: fixed-point with `ndp` decimals, falling back
    to scientific notation when fixed-point would round to 0, with trailing
    zeros trimmed. Optionally appends '%'."""
    if x is None or np.isnan(x):
        s = 'NaN'
    elif np.isinf(x):
        s = '+Inf' if x > 0 else '-Inf'
    else:
        ax = abs(x)
        tiny = 10.0 ** (-ndp)
        if ax > 0 and ax < tiny:
            s = f"{x:+.2e}"
        else:
            s = f"{x:+.{ndp}f}"
            if '.' in s:
                s = s.rstrip('0').rstrip('.')

    if is_percent:
        s = s[:-1] + '%'  # replaces the last digit with '%' to make room for it (intentional)
    return s


def metrics(observation, modeled, garea=None) -> SimpleNamespace:
    """
    Common evaluation metrics comparing observation vs. modeled data.

    Usage:
        out = metrics(observation_data, modeled_data)
        out = metrics(observation_data, modeled_data, garea)

    Optional:
        garea : weights for each point (same size as inputs, e.g. from
                eartharea()). Metrics are computed as weighted averages.

    Returns a SimpleNamespace with fields:
        N, R, R2, MB, NMB, MSE, RMSE, Weights, Slope, Intercept,
        LinearX, LinearY, Text (list of 5 label lines), TextH (one line).
    """
    obs = np.asarray(observation, dtype=float).ravel()
    model = np.asarray(modeled, dtype=float).ravel()
    use_weights = garea is not None

    if obs.size != model.size:
        raise ValueError("Inputs should have the same number of elements")

    if use_weights:
        garea_vec = np.asarray(garea, dtype=float).ravel()
        if garea_vec.size != obs.size:
            raise ValueError("Inputs should have the same number of elements")
        if np.any(garea_vec < 0):
            raise ValueError("garea must be nonnegative")
        invalid_garea = (np.isfinite(obs) | np.isfinite(model)) & np.isnan(garea_vec)
        if np.any(invalid_garea):
            raise ValueError("garea contains NaNs where observation or modeled values are finite")
    else:
        garea_vec = None

    # Remove NaNs:
    nan_mask = np.isnan(model) | np.isnan(obs)
    obs = obs[~nan_mask]
    model = model[~nan_mask]

    if use_weights:
        garea_vec = garea_vec[~nan_mask]
        if np.any(np.isnan(garea_vec)):
            raise ValueError("garea contains NaNs where observation or modeled values are finite")
        weight_sum = np.sum(garea_vec)
        if weight_sum == 0:
            raise ValueError("garea weights sum to zero")
        weights = garea_vec / weight_sum
    else:
        weights = np.ones(obs.shape) / max(obs.size, 1)

    out = SimpleNamespace()
    out.N = obs.size

    mean_model = np.sum(weights * model)
    mean_obs = np.sum(weights * obs)
    out.R = np.sum(weights * (model - mean_model) * (obs - mean_obs)) / np.sqrt(
        np.sum(weights * (model - mean_model) ** 2) * np.sum(weights * (obs - mean_obs) ** 2)
    )
    out.R2 = out.R ** 2
    out.MB = np.sum(weights * (model - obs))
    out.NMB = (np.sum(weights * (model - obs)) / np.sum(weights * obs)) * 100
    out.MSE = np.sum(weights * (model - obs) ** 2)
    out.RMSE = np.sqrt(out.MSE)
    out.Weights = weights

    # Linear regression coefficients:
    out.Slope = np.sum(weights * (obs - mean_obs) * (model - mean_model)) / np.sum(weights * (obs - mean_obs) ** 2)
    out.Intercept = mean_model - out.Slope * mean_obs

    # Sample data to plot the fit line:
    combined = np.concatenate([obs, model])
    mini = np.min(combined) * 0.01
    maxi = np.max(combined) * 20
    out.LinearX = np.linspace(mini, maxi, 500)
    out.LinearY = out.Slope * out.LinearX + out.Intercept

    # Text labels (metric-specific formatting):
    str_n = f"{out.N}"
    str_r = _fmt_num(out.R, 3, False)
    str_mb = _fmt_num(out.MB, 3, False)
    str_nmb = _fmt_num(out.NMB, 3, True)
    str_rmse = _fmt_num(out.RMSE, 3, False)
    str_rmse = str_rmse[1:]  # Removing leading '+' since RMSE >= 0

    # Vertical text: right-justified for alignment.
    W = max(len(str_n), len(str_r), len(str_mb), len(str_nmb), len(str_rmse))
    out.Text = [
        f"   N = {str_n.rjust(W)}",
        f"   R = {str_r.rjust(W)}",
        f"  MB = {str_mb.rjust(W)}",
        f" NMB = {str_nmb.rjust(W)}",
        f"RMSE = {str_rmse.rjust(W)}",
    ]

    # Horizontal text: compact, no padding.
    out.TextH = f"N={str_n} R={str_r} MB={str_mb} NMB={str_nmb} RMSE={str_rmse}"

    return out
