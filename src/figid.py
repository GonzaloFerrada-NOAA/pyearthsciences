"""Port of figid.m: attach a panel-label letter/text to an axes.

Directly translatable: MATLAB's text(..., 'Units', 'normalized', ...) is
exactly Matplotlib's ax.text(..., transform=ax.transAxes, ...) -- axes-
fraction (0-1) coordinates that stay pinned to the axes regardless of data
limits/zoom. MATLAB's own dance of temporarily switching the axes' Units to
'normalized' and back is only needed there because 'Units' is a shared axes
property; transAxes carries that meaning on its own, so it's dropped here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

_OFFSET = 0.015

_LOCATIONS = {
    'inleft':      ('top',    'left',  0.0 + _OFFSET,       1.0 - _OFFSET),
    'innerleft':   ('top',    'left',  0.0 + _OFFSET,       1.0 - _OFFSET),
    'inright':     ('top',    'right', 1.0 - _OFFSET,       1.0 - _OFFSET),
    'innerright':  ('top',    'right', 1.0 - _OFFSET,       1.0 - _OFFSET),
    'outerleft':   ('bottom', 'left',  0.0 + _OFFSET / 2,   1.0 + _OFFSET / 2),
    'outleft':     ('bottom', 'left',  0.0 + _OFFSET / 2,   1.0 + _OFFSET / 2),
    'outerright':  ('bottom', 'right', 1.0 - _OFFSET / 2,   1.0 + _OFFSET / 2),
    'outright':    ('bottom', 'right', 1.0 - _OFFSET / 2,   1.0 + _OFFSET / 2),
    'bottomleft':  ('bottom', 'left',  0.0 - _OFFSET,       0.0 + _OFFSET),
    'bottomright': ('bottom', 'right', 1.0 - _OFFSET,       0.0 + _OFFSET),
}


def figid(
    text: str,
    ax=None,
    Location: str = 'outleft',
    FontSize: float = 10,
    Background='none',
    FontName: str = 'monospace',
    Color=(0.15, 0.15, 0.15),
    FontWeight: str = 'normal',
    EdgeColor='none',
    Box: bool = False,
):
    """
    Add a panel-label letter/text pinned to an axes corner.

    Location: 'inleft'/'innerleft', 'inright'/'innerright',
              'outleft'/'outerleft', 'outright'/'outerright',
              'bottomleft', 'bottomright'  (default 'outleft')
    """
    if ax is None:
        ax = plt.gca()

    key = Location.lower()
    if key not in _LOCATIONS:
        raise ValueError(f"The location option provided ({Location}) is not available.")
    va, ha, x, y = _LOCATIONS[key]

    if Box:
        if isinstance(Background, str) and Background.lower() == 'none':
            Background = 'w'
        if isinstance(EdgeColor, str) and EdgeColor.lower() == 'none':
            EdgeColor = (0.15, 0.15, 0.15)

    bbox = None
    has_bg = not (isinstance(Background, str) and Background.lower() == 'none')
    has_edge = not (isinstance(EdgeColor, str) and EdgeColor.lower() == 'none')
    if has_bg or has_edge:
        bbox = dict(
            facecolor=Background if has_bg else 'none',
            edgecolor=EdgeColor if has_edge else 'none',
            boxstyle='square,pad=0.2',
        )

    h = ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=FontSize,
        family=FontName,
        color=Color,
        va=va, ha=ha,
        fontweight=FontWeight,
        bbox=bbox,
    )
    return h
