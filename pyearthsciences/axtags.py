"""Axes tagging helpers.

MATLAB's `findall(fig, 'Type', 'axes')` naturally excludes colorbars --
MATLAB colorbars are their own graphics object Type ('ColorBar'), not
'axes'. Matplotlib has no such distinction: a colorbar's axes (`cbar.ax`)
is a plain `Axes` and shows up in `fig.axes` right alongside the "real"
content axes. reorganizeaxes()/reorganizecolorbar() need to tell them apart,
so we tag colorbar axes explicitly with a reserved label when we create
them (see spatial.py) via mark_colorbar_axes().
"""
from __future__ import annotations

_COLORBAR_LABEL = "_pyearthsciences_colorbar_"


def mark_colorbar_axes(ax) -> None:
    """Tag `ax` as a colorbar's axes so reorganizeaxes()/reorganizecolorbar()
    skip it when collecting "real" content axes."""
    ax.set_label(_COLORBAR_LABEL)


def is_colorbar_axes(ax) -> bool:
    if ax.get_label() == _COLORBAR_LABEL:
        return True
    # Best-effort fallback for colorbars created without going through
    # mark_colorbar_axes (e.g. a bare plt.colorbar() elsewhere): recent
    # Matplotlib versions stash a back-reference on the axes.
    return getattr(ax, "_colorbar", None) is not None
