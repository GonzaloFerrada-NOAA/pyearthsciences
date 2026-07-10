"""Port of reorganizeaxes.m: lay out a figure's axes on a fixed pixel grid.

MATLAB figures are natively sized/positioned in pixels; Matplotlib figures
are sized in inches (at a given dpi) but position axes as figure-fraction
[left, bottom, width, height] -- same convention MATLAB uses for Axes
Position, so that part translates directly. Pixel sizes here are converted
to inches via the figure's dpi.

Colorbars: matplotlib has no separate "colorbar" object type -- a
colorbar's axes is a plain Axes and would otherwise be swept up alongside
the "real" content axes. Skip it via axtags.is_colorbar_axes (set
automatically by spatial(), or manually via axtags.mark_colorbar_axes for
colorbars created some other way).
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

try:
    from .axtags import is_colorbar_axes
except ImportError:  # pragma: no cover
    from axtags import is_colorbar_axes


def reorganizeaxes(nrows: int, ncols: int, width=None, height=None,
                    spacing_horiz: float = 0, spacing_vert: float = 0,
                    remove_tick_labels: bool = False, margin='auto', fig=None):
    """
    Lay out the current figure's (non-colorbar) axes on a fixed pixel grid.

    Parameters
    ----------
    nrows, ncols   : layout shape.
    width, height  : size of each axes, in pixels. At least one is required;
                     if only one is given, the other is derived from the
                     first axes' current aspect ratio (matches MATLAB's
                     PlotBoxAspectRatio fallback).
    spacing_horiz  : horizontal spacing between axes, in pixels.
    spacing_vert   : vertical spacing between axes, in pixels.
    remove_tick_labels : if True, hide Y tick labels on non-first-column
                     axes and X tick labels on non-last-row axes.
    margin         : blank space reserved around the whole grid, in pixels,
                     for things added afterward (colorbars, figid() labels,
                     ...). One of:
                       'auto' (default) -- left/right margins equal to the
                         panel width, top/bottom margins equal to the
                         (possibly derived) panel height, so the reserved
                         space scales with the grid instead of a fixed
                         guess. MATLAB used a fixed 120px on all sides;
                         pass an explicit value below to match that instead.
                       a single number -- applied to all four sides.
                       (left, right, top, bottom) -- asymmetric margins,
                         e.g. margin=(120, 120, 120, 250) for extra room
                         for a colorbar you'll add below the grid.
    fig            : target Figure (default: current figure).
    """
    if fig is None:
        fig = plt.gcf()

    axes_handles = [a for a in fig.axes if not is_colorbar_axes(a)]
    num_axes = len(axes_handles)
    if num_axes == 0:
        raise ValueError("No (non-colorbar) axes found in the figure.")

    if width is None and height is None:
        raise ValueError("Either width or height must be specified.")

    if width is None or height is None:
        # Derive the missing dimension from the first axes' current pixel
        # aspect ratio (stand-in for MATLAB's PlotBoxAspectRatio).
        fig.canvas.draw()
        bbox = axes_handles[0].get_window_extent()
        aspect = bbox.height / bbox.width if bbox.width else 1.0
        if height is None:
            height = width * aspect
        else:
            width = height / aspect

    if isinstance(margin, str) and margin == 'auto':
        margin_left = margin_right = width
        margin_top = margin_bottom = height
    elif isinstance(margin, (int, float)):
        margin_left = margin_right = margin_top = margin_bottom = margin
    else:
        margin_left, margin_right, margin_top, margin_bottom = margin

    fig_width = ncols * width + (ncols - 1) * spacing_horiz + margin_left + margin_right
    fig_height = nrows * height + (nrows - 1) * spacing_vert + margin_top + margin_bottom

    dpi = fig.dpi
    fig.set_size_inches(fig_width / dpi, fig_height / dpi)

    for i, ax in enumerate(axes_handles, start=1):
        row = math.ceil(i / ncols)
        col = (i - 1) % ncols + 1

        left = margin_left + (col - 1) * (width + spacing_horiz)
        bottom = fig_height - margin_top - row * height - (row - 1) * spacing_vert

        # Any axes that owns a colorbar created via make_axes_locatable
        # (see spatial.py) has a *dynamic* axes_locator installed on it by
        # that divider, which keeps recomputing/overriding its position
        # relative to its colorbar on every draw. set_position() alone
        # doesn't detach that locator, so this axes would silently ignore
        # the fixed grid position below and keep whatever size the divider
        # last computed for it -- clear it first so the position sticks.
        ax.set_axes_locator(None)
        ax.set_position([left / fig_width, bottom / fig_height,
                          width / fig_width, height / fig_height])

        if remove_tick_labels:
            if col != 1:
                ax.tick_params(labelleft=False)
            if row != nrows:
                ax.tick_params(labelbottom=False)
