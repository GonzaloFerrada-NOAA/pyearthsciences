"""Port of surf2img.m: rebuild a regular lon/lat grid shifted by half a cell
so pcolormesh/imagesc-style plotting aligns cell edges with the original
grid-point centers (matches MATLAB's ndgrid-based surf2img)."""
from __future__ import annotations

import numpy as np


def surf2img(xin, yin):
    xin = np.asarray(xin, dtype=float)
    yin = np.asarray(yin, dtype=float)

    if xin.ndim > 1 and yin.ndim > 1:
        xx1, xx2 = xin[0, :], xin[:, 0]
        yy1, yy2 = yin[0, :], yin[:, 0]

        dx1 = np.max(xx1) - np.min(xx1)
        dx2 = np.max(xx2) - np.min(xx2)
        dy1 = np.max(yy1) - np.min(yy1)
        dy2 = np.max(yy2) - np.min(yy2)

        if dx1 > dx2:
            xv = xx1
        elif dx2 > dx1:
            xv = xx2
        else:
            raise ValueError("xin data is not a regular grid.")

        if dy1 > dy2:
            yv = yy1
        elif dy2 > dy1:
            yv = yy2
        else:
            raise ValueError("yin data is not a regular grid.")
    else:
        xv = xin
        yv = yin

    dx = np.abs(np.diff(xv))
    dy = np.abs(np.diff(yv))

    if dy.size and np.min(dy) != np.max(dy):
        import warnings
        warnings.warn("yin is not regularly/equally spaced. May produce undesirable results.")
    if dx.size and np.min(dx) != np.max(dx):
        import warnings
        warnings.warn("xin is not regularly/equally spaced. May produce undesirable results.")

    rx = np.mean(dx) / 2
    ry = np.mean(dy) / 2

    x1 = np.linspace(np.min(xv) - rx, np.max(xv) + rx, xv.size)
    y1 = np.linspace(np.min(yv) - ry, np.max(yv) + ry, yv.size)

    x, y = np.meshgrid(x1, y1, indexing='ij')
    return x, y
