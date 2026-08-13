"""ticks2geo: add geographic graticule lines and labels to a Matplotlib axes.

Consolidates the old ticks2geo.m, robgeoticks.m, and lambgeoticks.m into one
function that accepts the same `proj`/`origin` convention as world() and
spatial() (see projcode.py) -- mirrors the equivalent consolidation already
done in the Julia port (src/ticks2geo.jl).

Usage:
    ticks2geo(ax)                                            # lat-lon (default)
    ticks2geo(ax, proj='robinson')
    ticks2geo(ax, proj='robinson', origin=[180], int_lon=30, int_lat=15)
    ticks2geo(ax, proj='lambert', origin=[33, 45, 40, -97])
    ticks2geo(ax, proj='lambert', origin=[40, -97], int_lon=10)

Orthographic/Perspective are not labeled (too complex for automatic labels;
calling with those projections is a no-op), matching the Julia port.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from .ll2rob import ll2rob
    from .ll2lamb import ll2lamb
    from .projcode import resolve_projection_code
except ImportError:  # pragma: no cover
    from ll2rob import ll2rob
    from ll2lamb import ll2lamb
    from projcode import resolve_projection_code


def _fmt(value: float) -> str:
    return f"{abs(value):g}"


def _lon_label(v: float) -> str:
    if v < 0:
        return f"{_fmt(v)}°W"
    if v > 0:
        return f"{_fmt(v)}°E"
    return f"{_fmt(v)}°"


def _lat_label(v: float) -> str:
    if v < 0:
        return f"{_fmt(v)}°S"
    if v > 0:
        return f"{_fmt(v)}°N"
    return f"{_fmt(v)}°"


def ticks2geo(ax=None, proj='latlon', origin=None, int_lon=None, int_lat=None,
              grid_alpha=None, grid_linewidth: float = 0.5, font_size=None):
    """
    Draw a geographic graticule (meridians + parallels) on a Matplotlib axes.

    For 'latlon', the existing numeric ticks are simply relabeled as degrees.
    For 'robinson' and 'lambert', grid lines are drawn and labeled manually.
    Orthographic and General Perspective are a no-op (not labeled).

    Parameters
    ----------
    ax        : target Axes (default: current axes)
    proj      : projection code or alias (see projcode.py); default 'latlon'
    origin    : Robinson: [center_lon] (default center_lon=0)
                Lambert : [lat0, lon0] or [lat1, lat2, lat0, lon0] (required)
    int_lon   : meridian spacing in degrees (default: 60 for Robinson, 10 for Lambert)
    int_lat   : parallel spacing in degrees (default: same as int_lon)
    grid_alpha: graticule line opacity (default: 0.1 Robinson / 0.2 Lambert)
    grid_linewidth : graticule line width (default 0.5)
    font_size : label font size (default: current axes font size)
    """
    if ax is None:
        ax = plt.gca()

    code = resolve_projection_code(proj)

    if code == 0:
        _ticks_latlon(ax, font_size)
    elif code == 1:
        center_lon = float(np.atleast_1d(origin)[0]) if origin is not None else 0.0
        il = 60.0 if int_lon is None else float(int_lon)
        ip = 30.0 if int_lat is None else float(int_lat)
        ga = 0.1 if grid_alpha is None else grid_alpha
        _ticks_robinson(ax, center_lon, il, ip, ga, grid_linewidth, font_size)
    elif code == 2:
        if origin is None:
            raise ValueError("ticks2geo(proj='lambert', ...) requires `origin` "
                              "([lat0,lon0] or [lat1,lat2,lat0,lon0]).")
        il = 10.0 if int_lon is None else float(int_lon)
        ip = il if int_lat is None else float(int_lat)
        ga = 0.2 if grid_alpha is None else grid_alpha
        _ticks_lambert(ax, origin, il, ip, ga, grid_linewidth, font_size)
    # else: orthographic/perspective -- no automatic labels


# ---------------------------------------------------------------------------
# Lat-lon: just reformat existing axis ticks as degree strings
# ---------------------------------------------------------------------------
def _ticks_latlon(ax, font_size=None):
    # In MATLAB, setting XTick/YTick never touches XLim/YLim. In Matplotlib,
    # set_xticks()/set_yticks() *can* silently expand the view to fit
    # whatever ticks the locator hands back -- which happens easily on a
    # small axes (e.g. one of several subplots), where the default locator
    # may pick "nice" round ticks that overshoot the actual data range.
    # Save/restore the limits around the relabeling so this is a true no-op
    # on the view, matching ticks2geo.m.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_ticks = [t for t in ax.get_xticks() if -180 <= t <= 180]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([_lon_label(t) for t in x_ticks], fontsize=font_size)

    y_ticks = [t for t in ax.get_yticks() if -90 <= t <= 90]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_lat_label(t) for t in y_ticks], fontsize=font_size)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Robinson graticule
# ---------------------------------------------------------------------------
def _ticks_robinson(ax, center_lon, int_lon, int_lat, grid_alpha, grid_linewidth, font_size):
    ax.grid(False)
    grid_color = (0.15, 0.15, 0.15, grid_alpha)

    # --- Meridians (constant longitude) ---
    lons = np.arange(0, 180 + int_lon, int_lon)
    lons = np.concatenate([-np.flip(lons[1:]), lons])
    lats = np.arange(-90, 90 + 0.1, 0.1)

    for lon0 in lons:
        lat = lats
        lon = np.full(lat.shape, lon0)
        x, y = ll2rob(lon, lat, center_lon)
        ax.plot(x, y, '-', linewidth=grid_linewidth, color=grid_color)
        ax.text(x[0], y[0], _lon_label(lon0), va='top', ha='center', fontsize=font_size)

    # --- Parallels (constant latitude) ---
    lons_d = np.arange(-180, 180 + 0.1, 0.1)
    lats_ticks = np.arange(0, 89.999 + int_lat, int_lat)
    lats_ticks = lats_ticks[lats_ticks <= 89.999]
    lats_ticks = np.concatenate([-np.flip(lats_ticks[1:]), lats_ticks])

    xlim = ax.get_xlim()
    xrange = xlim[1] - xlim[0]

    for lat0 in lats_ticks:
        lon = lons_d
        lat = np.full(lon.shape, lat0)
        x, y = ll2rob(lon, lat, center_lon)
        ax.plot(x, y, '-', linewidth=grid_linewidth, color=grid_color)

        offx = xrange * abs(y[0]) * 0.012 / 60
        ax.text(x[0] - offx, y[0], _lat_label(lat0), va='center', ha='right', fontsize=font_size)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Lambert graticule
# ---------------------------------------------------------------------------
def _ticks_lambert(ax, proj_specs, int_lon, int_lat, grid_alpha, grid_linewidth, font_size):
    proj_specs = list(np.atleast_1d(proj_specs).astype(float))
    if len(proj_specs) == 2:
        proj_specs = [proj_specs[0], proj_specs[0], proj_specs[0], proj_specs[1]]

    # Fix in case the two standard parallels coincide (avoid division by zero
    # in ll2lamb's `n` computation):
    if proj_specs[0] == proj_specs[1]:
        proj_specs[0] += 1e-5
        proj_specs[1] -= 1e-5

    lon_ticks = np.arange(-180, 180 + int_lon, int_lon)
    lat_ticks = np.arange(-80, 80 + int_lat, int_lat)

    grid_color = (0.15, 0.15, 0.15, grid_alpha)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ticks_y, tlabs_y = [], []
    ticks_x, tlabs_x = [], []

    # Latitude lines (constant lat, swept across longitude)
    for aux_lat in lat_ticks:
        lat_tick_lon = np.linspace(-200, 200, 4000)
        lat_tick_lat = np.full(lat_tick_lon.shape, aux_lat)

        tickx, ticky = ll2lamb(lat_tick_lat, lat_tick_lon, proj_specs)

        idx_out = (ticky < y_lim[0]) | (ticky > y_lim[1]) | (tickx < x_lim[0]) | (tickx > x_lim[1])
        tickx = np.where(idx_out, np.nan, tickx)
        ticky = np.where(idx_out, np.nan, ticky)

        idx = ~np.isnan(ticky)
        ticky, tickx = ticky[idx], tickx[idx]

        if tickx.size >= 1:
            ax.plot(tickx, ticky, '-', linewidth=grid_linewidth, color=grid_color)

            dist_axis = tickx[0] - x_lim[0]
            if abs(dist_axis) < (x_lim[1] - x_lim[0]) * 0.01:
                ticks_y.append(ticky[0])
                tlabs_y.append(_lat_label(aux_lat))

    # Longitude lines (constant lon, swept across latitude)
    for aux_lon in lon_ticks:
        lon_tick_lat = np.linspace(-80, 80, 10000)
        lon_tick_lon = np.full(lon_tick_lat.shape, aux_lon)

        tickx, ticky = ll2lamb(lon_tick_lat, lon_tick_lon, proj_specs)

        idx_out = (ticky < y_lim[0]) | (ticky > y_lim[1]) | (tickx < x_lim[0]) | (tickx > x_lim[1])
        tickx = np.where(idx_out, np.nan, tickx)
        ticky = np.where(idx_out, np.nan, ticky)

        idx = ~np.isnan(tickx)
        ticky, tickx = ticky[idx], tickx[idx]

        if tickx.size >= 1:
            ax.plot(tickx, ticky, '-', linewidth=grid_linewidth, color=grid_color)

            dist_axis = ticky[0] - y_lim[0]
            if abs(dist_axis) < (y_lim[1] - y_lim[0]) * 0.01:
                ticks_x.append(tickx[0])
                tlabs_x.append(_lon_label(aux_lon))

    ax.set_xticks(ticks_x)
    ax.set_xticklabels(tlabs_x, fontsize=font_size)
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(tlabs_y, fontsize=font_size)
    ax.tick_params(direction='out', length=0)
    ax.grid(False)
