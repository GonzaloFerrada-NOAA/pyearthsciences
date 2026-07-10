# pyearthsciences

Plotting & mapping utilities for Earth sciences, ported from a MATLAB toolkit
of the same purpose. Built to plot georeferenced 2-D data (model output,
satellite retrievals, etc.) on projected maps **without** the MATLAB Mapping
Toolbox or Cartopy — projections, coastlines/borders, and colorbars are all
handled with plain Matplotlib.

## Installation

```bash
pip install git+https://github.com/<your-username>/pyearthsciences.git
```

This works the same whether your target Python is a plain venv or a conda
environment — `pip` runs inside an activated conda env just like anywhere
else. (This is different from publishing to the `conda-forge` channel for
`conda install pyearthsciences`, which is a separate, heavier process — not
needed just to let someone else install this.)

For local development (editable install, so edits to the source take effect
immediately without reinstalling):

```bash
git clone https://github.com/<your-username>/pyearthsciences.git
cd pyearthsciences
pip install -e .
```

## Key functions

- **`spatial(lon, lat, data, ...)`** — the main plotting function. Plots
  gridded data on a map with a chosen projection, discrete or continuous
  color levels, an auto-fitted colorbar, and geographic tick labels.
  Supports lat-lon, Robinson, Lambert Conformal Conic, Orthographic, and
  General Perspective projections via `Projection=` (a code or a string
  like `'lambert'`) and `Origin=` (projection-specific parameters).

- **`world(res, ...)`** — draws coastlines/country/state boundaries on an
  existing (or new) Matplotlib axes, in the same projection conventions as
  `spatial()`. Used internally by `spatial()`, but callable standalone too.

- **`hue(*names, N, ...)`** — returns either a single RGB color or a
  Matplotlib `ListedColormap` interpolated from named/preset color stops
  (e.g. `hue('gmao', 64)`), including a library of built-in presets
  (`'jet3'`, `'hrrr'`, `'ssta'`, ...).

- **`metrics(observation, modeled, garea=None)`** — common model evaluation
  statistics (R, R², mean bias, normalized mean bias, RMSE, linear fit),
  optionally area-weighted, returned with ready-to-plot text summaries.

- **`eartharea(lon, lat)` / `earthmean(data, gridarea, ...)`** — grid-cell
  area (for a regular lon/lat grid) and area-weighted spatial averaging.

- **`figid(text, ax=None, Location=..., ...)`** — pins a panel label
  (e.g. `"a)"`) to a fixed corner of an axes, independent of zoom/data limits.

- **`reorganizeaxes(nrows, ncols, width, height, ...)` /
  `reorganizecolorbar(cb, nrows, ncols, location, ...)`** — lay out a grid
  of existing axes to exact pixel dimensions/spacing, and reposition a
  colorbar relative to that grid (including moving it to a different side,
  e.g. from vertical-right to horizontal-bottom).

## Quick example

```python
import numpy as np
import matplotlib.pyplot as plt
import pyearthsciences as es

lon = np.linspace(-130, -60, 60)
lat = np.linspace(20, 55, 40)
lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
data = np.sin(np.radians(lon2d)) * np.cos(np.radians(lat2d))

es.spatial(lon, lat, data, Projection='lambert', Origin=[38.5, -97.5], MapRes='na1')
plt.show()
```

## Data files

`pyearthsciences/world.npz` (coastline/border geometry) and
`pyearthsciences/htmlcolors.csv` (named-color lookup table for `hue()`) ship
as package data and are required at runtime.
