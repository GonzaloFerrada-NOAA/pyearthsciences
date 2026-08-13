from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from .hue import hue
    from .world import world as world_lines
    from .ll2lamb import ll2lamb
    from .ll2rob import ll2rob
    from .ll2ort import ll2ort
    from .ll2pers import ll2pers
    from .surf2img import surf2img
    from .ticks2geo import ticks2geo
    from .projcode import resolve_projection_code
    from .axtags import mark_colorbar_axes
except ImportError:  # pragma: no cover - allow running as a script
    from hue import hue
    from world import world as world_lines
    from ll2lamb import ll2lamb
    from ll2rob import ll2rob
    from ll2ort import ll2ort
    from ll2pers import ll2pers
    from surf2img import surf2img
    from ticks2geo import ticks2geo
    from projcode import resolve_projection_code
    from axtags import mark_colorbar_axes


def _nice_ticks(vmin: float, vmax: float) -> np.ndarray:
    """Stand-in for MATLAB's default auto tick-choosing algorithm (spatial.m
    harvests it via a disposable temporary axes; here we just ask
    Matplotlib's own default locator for 'nice' values over the range)."""
    return np.asarray(AutoLocator().tick_values(vmin, vmax))


def data2levels(datain: np.ndarray, levels):
    """Port of spatial.m's data2levels: bin data into discrete level indices."""
    levels = np.asarray(levels, dtype=float)
    dataout = np.full(datain.shape, np.nan)
    cbarticks = np.arange(1, levels.size + 1)
    clabels = [f"{v:g}" for v in levels]

    dataout[datain < levels[0]] = 1
    for i in range(levels.size - 1):
        idx = (datain >= levels[i]) & (datain < levels[i + 1])
        dataout[idx] = i + 1
    dataout[datain >= levels[-1]] = levels.size

    return dataout, cbarticks, clabels


def spatial(lon, lat, variable, *args, **kwargs):
    """
    SPATIAL (Python)
    Plot georeferenced 2-D data on a map, reprojected with this package's own
    ll2rob/ll2lamb/ll2ort/ll2pers formulas on a plain Matplotlib Axes --
    mirrors spatial.m directly and, like the rest of this toolkit, uses
    neither the MATLAB Mapping Toolbox nor Cartopy.

    Required:
        lon        : longitude in degrees (1-D or 2-D)
        lat        : latitude  in degrees (1-D or 2-D)
        variable   : 2-D data to plot

    Optional (MATLAB-like):
        Projection : projection code (0-4) or string alias -- see projcode.py:
                     0/'latlon' (default), 1/'rob'/'robinson', 2/'lambert'/'lamb',
                     3/'orth'/'orthogonal', 4/'pers'/'perspective'.
                     All projection parameters are given via `Origin`:
                       Robinson    : Origin=[center_lon] (default center_lon=0)
                       Lambert     : Origin=[lat0,lon0] or [lat1,lat2,lat0,lon0]
                                     (omit Origin to auto-center at the data's
                                     mean lat/lon)
                       Orthographic: Origin=[lon0, lat0]
                       Perspective : Origin=[lon0, lat0] or [lon0, lat0, height_km]
        Type       : 'surf' (default), 'rsurf', 'im'/'imsc'
        Colormap   : a Matplotlib Colormap, or anything hue() accepts.
                     Default matches spatial.m: hue('jet3','log',64).
        Colorbar   : 'on' (default) or 'off'
        Levels     : None (continuous) or 1-D array-like of bin edges
        MapRes     : 'na1' (default). See world() for the full list.
        MapWidth   : map boundary linewidth (default 0.5)
        MapColor   : RGB tuple or matplotlib color spec (default (0.25,0.25,0.25))
        GeoTicks   : 'on' (default) or 'off'
        Origin     : projection parameters -- see Projection above.
        Shading    : pcolormesh shading mode for Type='surf'/'rsurf' (default
                     'auto'). 'auto' treats lon/lat as cell centers (matching
                     how this toolkit's data is actually sampled -- each
                     point is a grid box's center, not a sample to blend
                     towards its neighbors) and flat-fills each cell with its
                     own value, no interpolation. This can print a "not
                     monotonically increasing" warning on curved grids (e.g.
                     Lambert) -- harmless, and suppressed by default (see
                     spatial.py's _SUPPRESS_SHADING_WARNING). 'gouraud'
                     instead treats lon/lat as vertices and smoothly
                     interpolates color across each face; only use it if you
                     actually want that blending look.

    Returns:
        (cbar, mappable)
    """
    Projection = kwargs.pop('Projection', 0)
    Type = kwargs.pop('Type', 'surf').lower()
    Colormap = kwargs.pop('Colormap', hue('jet3', 64))
    Colorbar = kwargs.pop('Colorbar', 'on').lower()
    Levels = kwargs.pop('Levels', None)
    MapRes = kwargs.pop('MapRes', 'na1')
    MapWidth = kwargs.pop('MapWidth', 0.5)
    MapColor = kwargs.pop('MapColor', (0.25, 0.25, 0.25))
    GeoTicks = kwargs.pop('GeoTicks', 'on').lower()
    Shading = kwargs.pop('Shading', 'auto')
    Origin = kwargs.pop('Origin', None)
    origin_given = Origin is not None
    if Origin is None:
        Origin = [0, 0]

    var = np.asarray(variable, dtype=float)
    if var.ndim != 2:
        raise ValueError("Input 'variable' must be 2-D.")

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # Fix longitude/latitude in case they are 1-D (matches ndgrid in spatial.m)
    if lon.ndim == 1 or lat.ndim == 1:
        lon, lat = np.meshgrid(np.ravel(lon), np.ravel(lat), indexing='ij')

    if lon.shape != var.shape:
        if lon.shape == var.T.shape:
            lon = lon.T
            lat = lat.T
        else:
            raise ValueError("The size of the input coordinates (lon, lat) do not match the variable's.")

    # ------------------------------
    # Determine projection
    # ------------------------------
    proj_code = resolve_projection_code(Projection)
    islatlon = proj_code == 0
    isrobinson = proj_code == 1
    islambert = proj_code == 2
    isorthogonal = proj_code == 3
    isperspective = proj_code == 4

    # Fix lat/lon in case Type == 'rsurf' (skipped for Lambert; applied to
    # the projected x/y further down instead, matching spatial.m)
    if Type == 'rsurf' and not islambert:
        lon, lat = surf2img(lon, lat)

    # ------------------------------
    # Reproject data if Projection != 0
    # ------------------------------
    robinson_center_lon = float(np.atleast_1d(Origin)[0]) if (isrobinson and origin_given) else 0.0
    if isrobinson:
        lon, lat = ll2rob(lon, lat, robinson_center_lon)

    lon_ticks = lat_ticks = None
    isglob = False
    lambert_specs = None
    if islambert:
        if origin_given:
            # Origin=[lat0,lon0] or [lat1,lat2,lat0,lon0]: all Lambert specs
            # now live exclusively in Origin (matches world()'s convention).
            lambert_specs = list(np.atleast_1d(Origin).astype(float))
        else:
            # Auto-center at the data's mean lat/lon, matching spatial.m's
            # behavior when no explicit specs are given.
            lambert_specs = [float(np.nanmean(lat)), float(np.nanmean(lon))]

        # "Nice" tick spacing over the raw lon/lat domain (stand-in for
        # spatial.m's disposable-axes trick), used later by ticks2geo.
        lon_ticks = _nice_ticks(float(np.nanmin(lon)), float(np.nanmax(lon)))
        lat_ticks = _nice_ticks(float(np.nanmin(lat)), float(np.nanmax(lat)))

        dlon = np.nanmax(lon) - np.nanmin(lon)
        dlat = np.nanmax(lat) - np.nanmin(lat)
        isglob = dlon > 270 or dlat > 120

        lon, lat = ll2lamb(lat, lon, lambert_specs)

        # Matches world.m's own Lambert filtering: points that project way
        # outside the visible cone (e.g. the "back side" of the globe for
        # near-global data, which Lambert conic was never meant to show)
        # blow up to huge values. Drop them instead of letting them corrupt
        # the plot/autoscale.
        far = (np.abs(lon) > 5e4) | (np.abs(lat) > 5e4)
        if np.any(far):
            lon = np.where(far, np.nan, lon)
            lat = np.where(far, np.nan, lat)

        if Type == 'rsurf':
            lon, lat = surf2img(lon, lat)

    if isorthogonal:
        lon, lat = ll2ort(lon, lat, Origin)

    if isperspective:
        if len(np.atleast_1d(Origin)) == 2:
            lon, lat = ll2pers(lon, lat, Origin)
        else:
            lon, lat = ll2pers(lon, lat, Origin[:2], Origin[2])

    # ------------------------------
    # Custom variable levels
    # ------------------------------
    cbartick = cbarticklabel = None
    if Levels is not None and len(np.atleast_1d(Levels)) > 0:
        var, cbartick, cbarticklabel = data2levels(var, Levels)

    # ------------------------------
    # Colormap
    # ------------------------------
    cmap = plt.get_cmap(Colormap) if isinstance(Colormap, str) else Colormap
    cmap = cmap.copy() if hasattr(cmap, 'copy') else cmap
    if hasattr(cmap, 'set_bad'):
        cmap.set_bad(alpha=0)

    # ------------------------------
    # Set up figure/axis (plain Matplotlib Axes, no cartopy)
    # ------------------------------
    ax = plt.gca()

    # pcolormesh/imshow tolerate NaN *data* but not NaN *coordinates*.
    # ll2ort/ll2pers NaN-out points on the far side of the globe, and the
    # Lambert far-point filter above does the same -- replace those
    # coordinates with a finite placeholder and mask the data there instead,
    # so the hidden points stay invisible without crashing the plot call.
    bad_xy = ~np.isfinite(lon) | ~np.isfinite(lat)
    if np.any(bad_xy):
        lon = np.where(bad_xy, np.nanmean(lon), lon)
        lat = np.where(bad_xy, np.nanmean(lat), lat)
        var = np.where(bad_xy, np.nan, var)

    # ------------------------------
    # Plot
    # ------------------------------
    if Type in ('surf', 'rsurf'):
        data_masked = np.ma.masked_invalid(var)
        with warnings.catch_warnings():
            # shading='auto' infers cell edges from lon/lat treated as cell
            # centers; on a curved/projected grid (e.g. Lambert) rows/columns
            # aren't strictly monotonic, which is expected here and not a
            # sign of bad data -- silence just this one specific warning.
            warnings.filterwarnings(
                'ignore',
                message=".*not monotonically increasing or decreasing.*",
                category=UserWarning,
            )
            h = ax.pcolormesh(lon, lat, data_masked, cmap=cmap, shading=Shading)
        ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
        ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
    elif Type in ('im', 'imsc'):
        data_masked = np.ma.masked_invalid(var.T)
        h = ax.imshow(
            data_masked,
            extent=[lon[:, 0].min(), lon[:, 0].max(), lat[0, :].min(), lat[0, :].max()],
            origin='lower',
            cmap=cmap,
            aspect='auto',
        )
    else:
        raise ValueError(f"{Type} is not a valid option for the 'Type' property")

    if cbartick is not None:
        h.set_clim(cbartick[0], cbartick[-1])

    # ------------------------------
    # Add coastlines / borders, and define axis limits
    # ------------------------------
    if MapColor != 'none':
        if islatlon:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            world_lines(MapRes, color=MapColor, linewidth=MapWidth, ax=ax, Projection=0)
            ax.set_xlim(xl); ax.set_ylim(yl)
        elif isrobinson:
            world_lines(MapRes, color=MapColor, linewidth=MapWidth, ax=ax, Projection=1, Origin=[robinson_center_lon])
        elif isorthogonal:
            world_lines(MapRes, color=MapColor, linewidth=MapWidth, ax=ax, Projection=3, Origin=Origin)
        elif isperspective:
            world_lines(MapRes, color=MapColor, linewidth=MapWidth, ax=ax, Projection=4, Origin=Origin)
        elif islambert:
            world_lines(MapRes, color=MapColor, linewidth=MapWidth, ax=ax, Projection=2, Origin=lambert_specs)

    # ------------------------------
    # Colorbar (kept to the axes' own height/width, MATLAB-style)
    # ------------------------------
    cb = None
    if Colorbar == 'on':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        mark_colorbar_axes(cax)
        cb = plt.colorbar(h, cax=cax, orientation='vertical')

        if cbartick is not None:
            cb.set_ticks(cbartick)
            cb.set_ticklabels(cbarticklabel)

    # ------------------------------
    # Figure properties
    # ------------------------------
    ax.set_axisbelow(False)  # 'Layer','top'
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_aspect('equal', adjustable='box')

    # ------------------------------
    # Geoticks
    # ------------------------------
    if GeoTicks == 'on':
        if islatlon:
            ticks2geo(ax=ax, proj=0)
        elif isrobinson:
            ticks2geo(ax=ax, proj=1, origin=[robinson_center_lon])
        elif islambert:
            axl = np.abs(np.concatenate([ax.get_xlim(), ax.get_ylim()]))
            if np.max(axl) > 1e4:
                ax.set_xlim(-2700, 2700)
                ax.set_ylim(-2000, 2000)
            ticks2geo(ax=ax, proj=2, origin=lambert_specs,
                      int_lon=float(np.mean(np.diff(lon_ticks))),
                      int_lat=float(np.mean(np.diff(lat_ticks))))

    # ------------------------------
    # Special settings for Robinson
    # ------------------------------
    if isrobinson:
        # set_xlim/set_ylim on the data plot above disabled autoscaling on
        # this axes; re-enable it so the view expands from the data-only
        # extent to include the full frame/coastlines just added (matches
        # MATLAB's `axis tight`, which always recomputes from what's
        # currently plotted rather than a one-time, now-frozen, view).
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------
    # Special settings for Lambert
    # ------------------------------
    if islambert and isglob:
        ax.set_xlim(-2700, 2700)
        ax.set_ylim(-1800, 1800)

    # ------------------------------
    # Special settings for Orthographic / Perspective: these only ever show
    # a disk (the visible hemisphere/horizon circle world() already drew),
    # so the rectangular axes box around it is never meaningful -- hide it.
    # (spatial.m only does this for Orthographic via world.m's XColor/YColor
    # override; Perspective's equivalent line is commented out in world.m.
    # Hiding it for both here since a stray box makes no sense for either.)
    # ------------------------------
    if isorthogonal or isperspective:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Restore this axes as pyplot's "current axes" -- without this, the
    # colorbar axes created above (the last one added to the figure) stays
    # current, so any subsequent plt.xlim/ylim/axis(...) calls silently
    # target the colorbar instead of the map.
    plt.sca(ax)

    return cb, h
