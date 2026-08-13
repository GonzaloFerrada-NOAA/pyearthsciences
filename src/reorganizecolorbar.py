"""Port of reorganizecolorbar.m: reposition a colorbar relative to a
nrows x ncols grid of axes.

One thing MATLAB gets "for free" that Matplotlib doesn't: setting a MATLAB
colorbar's `Location` to 'northoutside'/'southoutside' (top/bottom) vs.
'westoutside'/'eastoutside' (left/right) also flips its internal orientation
between horizontal and vertical. A Matplotlib Colorbar's orientation is
fixed at creation time, so moving between a top/bottom side and a left/right
side here removes the old colorbar and creates a new one (reusing the same
mappable) rather than just nudging its position -- moving within the same
family (e.g. right -> left) just repositions and flips the tick side.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

try:
    from .axtags import is_colorbar_axes, mark_colorbar_axes
except ImportError:  # pragma: no cover
    from axtags import is_colorbar_axes, mark_colorbar_axes

_LOCATION_MAP = {
    'top': ('horizontal', 'top'),
    'north': ('horizontal', 'top'),
    'bottom': ('horizontal', 'bottom'),
    'south': ('horizontal', 'bottom'),
    'left': ('vertical', 'left'),
    'west': ('vertical', 'left'),
    'right': ('vertical', 'right'),
    'east': ('vertical', 'right'),
}


def reorganizecolorbar(colorbar, nrows: int, ncols: int, location: str = 'right',
                        proportion: float = 1.0, extent_axes=None,
                        mappable=None, fig=None):
    """
    Reposition `colorbar` (a matplotlib Colorbar, e.g. the one returned by
    spatial()) relative to a nrows x ncols grid of the figure's other axes.

    Parameters
    ----------
    colorbar    : the Colorbar to reposition (e.g. spatial()'s first return value).
    nrows, ncols: shape of the axes grid (must match the number of
                  non-colorbar axes currently in the figure).
    location    : 'top'/'north', 'bottom'/'south', 'left'/'west', 'right'/'east'.
    proportion  : 0 < proportion <= 1, fraction of the spanned axes' extent
                  the colorbar should occupy (centered).
    extent_axes : 1-based [start, end] row/col range the colorbar spans
                  (rows for left/right, columns for top/bottom). Default:
                  the full span ([1, ncols] or [1, nrows]).
    mappable    : override for the ScalarMappable to reattach when the
                  colorbar must be recreated (orientation change). Defaults
                  to `colorbar.mappable`.
    fig         : target Figure (default: current figure).

    Returns
    -------
    The Colorbar in its final position (same object if only repositioned;
    a new one if it had to be recreated for an orientation change).
    """
    if fig is None:
        fig = plt.gcf()

    key = location.lower()
    if key not in _LOCATION_MAP:
        raise ValueError("Invalid location option. Use: top, bottom, left, or right.")
    orientation, side = _LOCATION_MAP[key]

    if extent_axes is None:
        extent_axes = [1, ncols] if orientation == 'horizontal' else [1, nrows]
    elif extent_axes[0] > extent_axes[1]:
        raise ValueError(f"extent_axes[0] is greater than extent_axes[1]: {extent_axes}")

    axes_handles = [a for a in fig.axes if not is_colorbar_axes(a)]
    if len(axes_handles) != nrows * ncols:
        raise ValueError("Number of axes does not match specified nrows and ncols.")

    # If this colorbar (or any of the grid's axes) was created via
    # make_axes_locatable (spatial()'s default), it carries a *dynamic*
    # axes_locator that keeps recomputing its position relative to its
    # sibling axes on every draw -- force one now so get_position() below
    # reflects where things actually are, not a stale/mid-relayout value.
    fig.canvas.draw()

    def pos(idx0):
        return axes_handles[idx0].get_position().bounds  # (x0, y0, w, h)

    cb_x0, cb_y0, cb_w, cb_h = colorbar.ax.get_position().bounds
    # The colorbar's "thickness" (its short dimension) is cb_w if it's
    # currently vertical, cb_h if currently horizontal -- reuse that as the
    # new bar's thickness regardless of which side we're moving it to, so an
    # orientation flip doesn't inherit the old bar's *length* as its new
    # thickness (MATLAB recomputes a sensible thickness automatically when
    # Location crosses between north/south and east/west; Matplotlib doesn't).
    cb_thickness = cb_w if colorbar.orientation == 'vertical' else cb_h

    if side == 'top':
        row_index = 0  # always top row
        col_start, col_end = extent_axes[0] - 1, extent_axes[1] - 1
        xs0, ys0, ws0, hs0 = pos(row_index * ncols + col_start)
        xs1, ys1, ws1, hs1 = pos(row_index * ncols + col_end)
        X = xs0
        Y = ys0 + hs0 + 0.02
        W = (xs1 + ws1 - xs0) * proportion
        H = cb_thickness
        X = X + (1 - proportion) * (W / proportion) / 2

    elif side == 'bottom':
        row_index = nrows - 1  # always bottom row
        col_start, col_end = extent_axes[0] - 1, extent_axes[1] - 1
        xs0, ys0, ws0, hs0 = pos(row_index * ncols + col_start)
        xs1, ys1, ws1, hs1 = pos(row_index * ncols + col_end)
        X = xs0
        Y = ys0 - 0.05
        W = (xs1 + ws1 - xs0) * proportion
        H = cb_thickness
        X = X + (1 - proportion) * (W / proportion) / 2

    elif side == 'left':
        col_index = 0  # always left-most column
        row_start, row_end = extent_axes[0] - 1, extent_axes[1] - 1
        xs0, ys0, ws0, hs0 = pos(row_end * ncols + col_index)
        xs1, ys1, ws1, hs1 = pos(row_start * ncols + col_index)
        X = xs0 - 0.05
        Y = ys0
        W = cb_thickness
        H = (ys1 + hs1 - ys0) * proportion
        Y = Y + (1 - proportion) * (H / proportion) / 2

    else:  # right
        col_index = ncols - 1  # always right-most column
        row_start, row_end = extent_axes[0] - 1, extent_axes[1] - 1
        xs0, ys0, ws0, hs0 = pos(row_end * ncols + col_index)
        xs1, ys1, ws1, hs1 = pos(row_start * ncols + col_index)
        X = xs0 + ws0 + 0.02
        Y = ys0
        W = cb_thickness
        H = (ys1 + hs1 - ys0) * proportion
        Y = Y + (1 - proportion) * (H / proportion) / 2

    if colorbar.orientation != orientation:
        # Matplotlib bakes orientation into the Colorbar at creation time --
        # unlike MATLAB, we can't just flip a property. Recreate it in place,
        # carrying over any custom ticks/labels (e.g. spatial()'s discrete
        # Levels tick labels) and axis label, which a fresh colorbar()
        # wouldn't otherwise know about.
        old_ticks = colorbar.get_ticks()
        old_axis = colorbar.ax.yaxis if colorbar.orientation == 'vertical' else colorbar.ax.xaxis
        old_labels = [t.get_text() for t in old_axis.get_ticklabels()]
        old_label_text = old_axis.get_label().get_text()

        mp = mappable if mappable is not None else colorbar.mappable
        colorbar.remove()
        new_ax = fig.add_axes([X, Y, W, H])
        mark_colorbar_axes(new_ax)
        colorbar = fig.colorbar(mp, cax=new_ax, orientation=orientation)

        if len(old_labels) == len(old_ticks) and any(old_labels):
            colorbar.set_ticks(old_ticks)
            colorbar.set_ticklabels(old_labels)
        if old_label_text:
            colorbar.set_label(old_label_text)
    else:
        colorbar.ax.set_axes_locator(None)  # detach any leftover divider locator
        colorbar.ax.set_position([X, Y, W, H])

    if side == 'top':
        colorbar.ax.xaxis.set_ticks_position('top')
        colorbar.ax.xaxis.set_label_position('top')
    elif side == 'bottom':
        colorbar.ax.xaxis.set_ticks_position('bottom')
        colorbar.ax.xaxis.set_label_position('bottom')
    elif side == 'left':
        colorbar.ax.yaxis.set_ticks_position('left')
        colorbar.ax.yaxis.set_label_position('left')
    else:
        colorbar.ax.yaxis.set_ticks_position('right')
        colorbar.ax.yaxis.set_label_position('right')

    return colorbar
