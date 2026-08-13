import re
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np


def _adjust_precision(num_format: str, delta: int) -> str:
    """Shift the precision (the digits after '.') of a format spec like
    '.4f' or '8.4f' by `delta`, clamped at 0. Specs without an explicit
    precision are returned unchanged."""
    m = re.search(r"\.(\d+)", num_format)
    if not m:
        return num_format
    new_prec = max(int(m.group(1)) + delta, 0)
    start, end = m.span(1)
    return num_format[:start] + str(new_prec) + num_format[end:]


def _fmt_num(x: float, ndp: int = 3, is_percent: bool = False, num_format: Optional[str] = None) -> str:
    """Format a metric value.

    If `num_format` is given, values are formatted with that format spec
    (e.g. '.4f'); percent values use one fewer decimal than `num_format`
    to leave room for the '%'. Otherwise, the default is fixed-point with
    `ndp` decimals, falling back to scientific notation when fixed-point
    would round to 0, and percent values replace their last digit with
    '%' to make room for it."""
    if x is None or np.isnan(x):
        s = 'NaN'
    elif np.isinf(x):
        s = '+Inf' if x > 0 else '-Inf'
    elif num_format is not None:
        fmt = _adjust_precision(num_format, -1) if is_percent else num_format
        s = f"{x:{fmt}}"
        return s + '%' if is_percent else s
    else:
        ax = abs(x)
        tiny = 10.0 ** (-ndp)
        if ax > 0 and ax < tiny:
            s = f"{x:+.2e}"
        else:
            s = f"{x:+.{ndp}f}"

    if is_percent:
        s = s[:-1] + '%'  # replaces the last digit with '%' to make room for it (intentional)
    return s


_ALL_METRICS = ["N", "R", "R2", "MB", "NMB", "MSE", "RMSE", "cRMSE"]
_DEFAULT_TEXT_METRICS = ["N", "R", "MB", "NMB", "RMSE", "cRMSE"]
_NONNEGATIVE_METRICS = {"R2", "MSE", "RMSE", "cRMSE"}
_SIGNED_METRICS = {"R", "MB", "NMB"}
_PERCENT_METRICS = {"NMB"}


def metrics(
    observation,
    modeled,
    garea=None,
    metrics_to_show: Optional[Sequence[str]] = None,
    num_format: Optional[str] = None,
) -> SimpleNamespace:
    """
    Common evaluation metrics comparing observation vs. modeled data.

    Usage:
        out = metrics(observation_data, modeled_data)
        out = metrics(observation_data, modeled_data, garea)
        out = metrics(observation_data, modeled_data, metrics_to_show=["R", "MB", "cRMSE"])
        out = metrics(observation_data, modeled_data, num_format='.4f')

    Optional:
        garea : weights for each point (same size as inputs, e.g. from
                eartharea()). Metrics are computed as weighted averages.
        metrics_to_show : which metrics to include in Text/TextH, and in
                what order. Any of "N", "R", "R2", "MB", "NMB", "MSE",
                "RMSE", "cRMSE". Default: ["N", "R", "MB", "NMB", "RMSE",
                "cRMSE"].
        num_format : format spec (e.g. '.4f') applied to all float metrics
                in Text/TextH. NMB is formatted with one fewer decimal
                place than `num_format` to leave room for the '%'. Not
                applied to N. Default: legacy 3-decimal formatting with
                a scientific-notation fallback for tiny values.

    Returns a SimpleNamespace with fields:
        N, R, R2, MB, NMB, MSE, RMSE, cRMSE, Weights, Slope, Intercept,
        LinearX, LinearY, Text (list of label lines), TextH (one line).
    """
    if metrics_to_show is not None:
        unknown = [name for name in metrics_to_show if name not in _ALL_METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s) in metrics_to_show: {unknown}")
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
    out.cRMSE = np.sqrt(out.MSE - out.MB ** 2)  # bias-removed (centered) RMSE
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
    show = list(metrics_to_show) if metrics_to_show is not None else _DEFAULT_TEXT_METRICS
    values = {"N": out.N, "R": out.R, "R2": out.R2, "MB": out.MB, "NMB": out.NMB,
              "MSE": out.MSE, "RMSE": out.RMSE, "cRMSE": out.cRMSE}

    strs = {}
    for name in show:
        if name == "N":
            strs[name] = f"{out.N}"
            continue
        s = _fmt_num(values[name], 3, name in _PERCENT_METRICS, num_format)
        if name in _NONNEGATIVE_METRICS and s.startswith('+'):
            s = s[1:]  # Removing leading '+' since this metric is always >= 0
        elif name in _SIGNED_METRICS and s[:1].isdigit():
            s = '+' + s  # num_format without a '+' flag: still show sign for +/- metrics
        strs[name] = s

    # Vertical text: right-justified for alignment.
    label_width = max(len(name) for name in show)
    value_width = max(len(strs[name]) for name in show)
    out.Text = [f"{name.rjust(label_width)} = {strs[name].rjust(value_width)}" for name in show]

    # Horizontal text: compact, no padding.
    out.TextH = " ".join(f"{name}={strs[name]}" for name in show)

    return out
