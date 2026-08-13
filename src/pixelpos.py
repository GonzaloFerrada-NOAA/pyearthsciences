"""Pixel-based axes positioning, MATLAB-Position style.

Matplotlib positions axes in figure-fraction units (0-1), which is why the
same numeric offset means a different number of physical pixels depending
on figure size. MATLAB's Position is always in pixels. move_axes() lets you
keep thinking in pixels (nudge/resize by a fixed physical amount) while it
handles the conversion.
"""
from __future__ import annotations


def move_axes(ax, dx_px: float = 0, dy_px: float = 0,
              dw_px: float = 0, dh_px: float = 0, fig=None):
    """
    Nudge/resize `ax` (e.g. a colorbar's `.ax`) by pixel amounts.

    Parameters
    ----------
    ax     : the Axes to move/resize (e.g. `cb.ax`).
    dx_px, dy_px : shift left/bottom by this many pixels (negative = down/left).
    dw_px, dh_px : grow width/height by this many pixels (negative = shrink).
    fig    : owning Figure (default: ax.figure).

    Returns
    -------
    The new (x0, y0, width, height) in figure-fraction units.
    """
    if fig is None:
        fig = ax.figure

    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    fig_h_px = fig.get_size_inches()[1] * fig.dpi

    # Detach any leftover divider locator (e.g. from make_axes_locatable in
    # spatial()) so the explicit position below actually sticks.
    ax.set_axes_locator(None)

    x0, y0, w, h = ax.get_position().bounds
    new_bounds = [
        x0 + dx_px / fig_w_px,
        y0 + dy_px / fig_h_px,
        w + dw_px / fig_w_px,
        h + dh_px / fig_h_px,
    ]
    ax.set_position(new_bounds)
    return tuple(new_bounds)
