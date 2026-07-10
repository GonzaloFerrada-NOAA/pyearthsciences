from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    from .ll2lamb import ll2lamb
    from .ll2rob import ll2rob
    from .ll2ort import ll2ort
    from .ll2pers import ll2pers
    from .projcode import resolve_projection_code
except ImportError:  # pragma: no cover - allow running as a script
    from ll2lamb import ll2lamb
    from ll2rob import ll2rob
    from ll2ort import ll2ort
    from ll2pers import ll2pers
    from projcode import resolve_projection_code


# --------- CONFIG ---------
# default path: same folder as this file
_DEFAULT_MAT = Path(__file__).resolve().parent / "world.mat"

# mapping of user-facing keys to variables in MAT file
# (based on your world.m switch)
_KEYMAP = {
    # Medium
    "mc": "mc", "mcoast": "mc",
    "m0": "m0", "mres": "m0", "mres0": "m0",
    "m1": "m1", "mres1": "m1",
    "inm1": "inm1",
    # High
    "hc": "hc", "hcoast": "hc",
    "h0": "h0", "hires": "h0", "hires0": "h0",
    "h1": "h1", "hires1": "h1",
    "inh1": "inh1",
    "lakes": "lakes",
    # Super-high
    "sh0": "sh0",
    "sh1": "sh1",
    # USA
    "usastates": "usastates", "usa1": "usastates",
    "usacounties": "usacounties", "usa2": "usacounties",
    # Major countries adm1 / mixes
    "na1": "nonus", "na2": "nonus", "nonus": "nonus",
    # Other
    "hstates": "hstates",
}

def _cell_to_list_of_arrays(cell) -> List[np.ndarray]:
    """
    Convert MATLAB cell array (as loaded by scipy.io.loadmat) to a list of (N,2) float arrays.
    """
    arrs: List[np.ndarray] = []
    flat = np.atleast_1d(cell).ravel()
    for item in flat:
        # Each item should be an ndarray Nx2
        a = np.array(item, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 2:
            arrs.append(a[:, :2])
    return arrs

@lru_cache(maxsize=1)
def _load_world_npz(npz_path: str | Path) -> Dict[str, List[np.ndarray]]:
    """
    Load world_np.npz into the same format as _load_world_mat().
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as z:
        return {k: list(z[k]) for k in z.files}

def _load_world_data(path: Path) -> Dict[str, List[np.ndarray]]:
    """
    Try NPZ first, fallback to MAT.
    """
    if path.suffix == ".npz":
        return _load_world_npz(path)

    npz_candidate = path.with_name("world.npz")
    if npz_candidate.exists():
        return _load_world_npz(npz_candidate)

    return _load_world_mat(path)

def _load_world_mat(mat_path: str | Path = _DEFAULT_MAT) -> Dict[str, List[np.ndarray]]:
    """
    Load world.mat once and normalize everything to dict[name] -> list of (N,2) arrays [lon,lat].
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"world.mat not found at {mat_path}")

    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # Pull only known keys; each is a MATLAB cell array -> Python object array -> list of arrays
    out: Dict[str, List[np.ndarray]] = {}
    for k in {
        "mc", "m0", "m1", "inm1",
        "hc", "h0", "h1", "inh1", "lakes",
        "sh0", "sh1",
        "usastates", "usacounties",
        "nonus", "hstates"
    }:
        if k in md:
            out[k] = _cell_to_list_of_arrays(md[k])
        else:
            out[k] = []
    return out

def _as_linecollection(
    segments_lonlat: Iterable[np.ndarray],
    color=(0.3, 0.3, 0.3),
    linewidth: float = 1.0,
) -> LineCollection:
    """
    Build a single LineCollection from a list of (N,2) lon/lat arrays.
    """
    segs = [s[:, :2] for s in segments_lonlat if isinstance(s, np.ndarray) and s.size >= 4]
    if not segs:
        return LineCollection([], colors=[color], linewidths=linewidth, zorder=10)

    return LineCollection(segs, colors=[color], linewidths=linewidth, zorder=10)


def _normalize_projection(Projection=0, Origin=None) -> Tuple[str, dict]:
    """
    Map a Projection code/alias + Origin params to a projection identifier and its parameters.

    Projection (int code, or string alias -- see projcode.py):
        0 / 'latlon'              : latlon (identity)
        1 / 'rob'/'robinson'      : Robinson     (Origin[0] -> central longitude)
        2 / 'lambert'/'lamb'      : Lambert      (Origin len 2 -> [lat0, lon0]; len 4 -> [lat1, lat2, lat0, lon0])
        3 / 'orth'/'orthogonal'   : Orthographic (Origin[0:2] -> [lon0, lat0])
        4 / 'pers'/'perspective'  : Perspective  (Origin[0:2] -> [lon0, lat0], Origin[2] -> height)
    """
    code = resolve_projection_code(Projection)

    origin = None if Origin is None else list(np.atleast_1d(Origin))

    if code == 0:
        return "latlon", {}

    if code == 1:
        center_lon = float(origin[0]) if origin and len(origin) >= 1 else 0.0
        return "robinson", {"center_lon": center_lon}

    if code == 2:
        if origin is None:
            specs = [0.0, 0.0]
        elif len(origin) == 2:
            specs = [float(origin[0]), float(origin[1])]
        elif len(origin) == 4:
            specs = [float(origin[0]), float(origin[1]), float(origin[2]), float(origin[3])]
        else:
            raise ValueError("Lambert Origin must have length 2 ([lat0, lon0]) or 4 ([lat1, lat2, lat0, lon0]).")
        return "lambert", {"lambert_specs": specs}

    if code == 3:
        if origin is None or len(origin) < 2:
            center = [0.0, 0.0]
        else:
            center = [float(origin[0]), float(origin[1])]
        return "orthographic", {"center": center}

    if code == 4:
        if origin is None or len(origin) < 2:
            center = [0.0, 0.0]; height = 35786.0
        else:
            center = [float(origin[0]), float(origin[1])]
            height = float(origin[2]) if len(origin) >= 3 else 35786.0
        return "perspective", {"center": center, "height": height}

    raise ValueError("Unsupported Projection code. Use 0–4.")


def _project_segments(segments: Iterable[np.ndarray], proj_code: str, proj_params: dict) -> List[np.ndarray]:
    """
    Apply the requested projection to each lon/lat segment.
    """
    projected: List[np.ndarray] = []

    for seg in segments:
        if not isinstance(seg, np.ndarray) or seg.size < 4:
            continue

        lon = seg[:, 0]
        lat = seg[:, 1]

        if proj_code == "latlon":
            x, y = lon, lat
        elif proj_code == "robinson":
            x, y = ll2rob(lon, lat, proj_params.get("center_lon", 0.0))
        elif proj_code == "lambert":
            # Matches world.m: drop points exactly on the +-180 meridian before
            # projecting, then drop anything that lands far outside the visible
            # cone (>5e4 units) after projecting.
            lon_f = lon.copy()
            lat_f = lat.copy()
            seam = (lon_f == 180) | (lon_f == -180)
            lon_f[seam] = np.nan
            lat_f[seam] = np.nan
            x, y = ll2lamb(lat_f, lon_f, proj_params["lambert_specs"])
            far = (np.abs(x) > 5e4) | (np.abs(y) > 5e4)
            x = np.where(far, np.nan, x)
            y = np.where(far, np.nan, y)
        elif proj_code == "orthographic":
            x, y = ll2ort(lon, lat, proj_params["center"])
        elif proj_code == "perspective":
            x, y = ll2pers(lon, lat, proj_params["center"], proj_params["height"])
        else:
            raise RuntimeError(f"Unhandled projection '{proj_code}'")

        projected.append(np.column_stack((x, y)))

    return projected


def _draw_projection_boundary(ax: plt.Axes, proj_code: str, proj_params: dict, color, linewidth): # pyright: ignore[reportPrivateImportUsage]
    """
    Draw Earth boundary for projections that need it.
    """
    if proj_code == "robinson":
        # Matches world.m: four explicit edges (top, right, bottom, left) with
        # NaN breaks between them, always projected at center_lon=0. The
        # Robinson envelope shape is invariant to center_lon (only which raw
        # longitude maps to the seam changes), so this stays correct
        # regardless of the data's actual center_lon -- unlike naively
        # reusing proj_params's center_lon here, which would draw an
        # incorrectly narrowed frame for any center_lon != 0.
        n = 720
        lon_top = np.linspace(-180.0, 180.0, n); lat_top = np.full(n, 90.0)
        lon_rgt = np.full(n, 180.0);              lat_rgt = np.linspace(90.0, -90.0, n)
        lon_bot = np.linspace(180.0, -180.0, n);  lat_bot = np.full(n, -90.0)
        lon_lft = np.full(n, -180.0);             lat_lft = np.linspace(-90.0, 90.0, n)

        nan1 = np.array([np.nan])
        lon = np.concatenate([lon_top, nan1, lon_rgt, nan1, lon_bot, nan1, lon_lft])
        lat = np.concatenate([lat_top, nan1, lat_rgt, nan1, lat_bot, nan1, lat_lft])

        x, y = ll2rob(lon, lat, 0.0)
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=9)

    elif proj_code == "orthographic":
        R = 100.0  # same radius used in ll2ort
        theta = np.linspace(0, 2 * np.pi, 1000)
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=9)
        ax.tick_params(labelbottom=False, labelleft=False)

    elif proj_code == "perspective":
        R = 6378.0
        rho = R + proj_params.get("height", 35786.0)
        radius = 1.0 / np.sqrt(1 - (R / rho) ** 2)  # maximum projected radius (horizon)
        theta = np.linspace(0, 2 * np.pi, 720)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=9)


def _maybe_set_axes_limits(ax: plt.Axes, proj_code: str, pre_xlim, pre_ylim) -> bool: # type: ignore
    """
    Apply projection-specific default limits.

    pre_xlim/pre_ylim are the axes limits captured BEFORE any data was added
    (matches world.m, which reads XLim/YLim before plotting): a fresh axes
    still at its default [0,1]/[0,1] square gets the default Lambert view;
    an axes the caller already zoomed/panned keeps that view.

    Returns True if limits were handled here (to skip generic autoscaling).
    """
    if proj_code != "lambert":
        return False

    # MATLAB: sum(y_lim == x_lim) == 2
    if np.allclose(pre_ylim, pre_xlim):
        ax.set_xlim(-2700, 2700)
        ax.set_ylim(-2000, 2000)
    else:
        ax.set_xlim(pre_xlim)
        ax.set_ylim(pre_ylim)
    return True

def world(
    res: str = "m0",
    color=(0.25, 0.25, 0.25),
    linewidth: float = 0.5,
    *,
    ax: Optional[plt.Axes] = None, # pyright: ignore[reportPrivateImportUsage]
    mat_path: str | Path = _DEFAULT_MAT,
    Projection=0,
    Origin=None,
) -> LineCollection:
    """
    Draw borders/lines from your world.npz/world.mat on regular Matplotlib axes,
    using simple numeric projections.

    Parameters
    ----------
    res : str
        Map resolution key (e.g., 'm0', 'h1', 'sh0', 'usastates', 'na1', ...).
    color : tuple or matplotlib color
        Line color.
    linewidth : float
        Line width.
    ax : Matplotlib Axes (optional)
        If None, uses plt.gca().
    mat_path : path-like
        Path to world.mat (default: same folder as this file).
    Projection : int code or string alias
        0/None, 'latlon'             -> latlon (lon/lat)
        1, 'rob'/'robinson'          -> Robinson (Origin[0] as center longitude)
        2, 'lambert'/'lamb'          -> Lambert  (Origin len 2 -> [lat0, lon0]; len 4 -> [lat1, lat2, lat0, lon0])
        3, 'orth'/'ort'/'orthogonal' -> Orthographic (Origin[0:2] as center lon/lat)
        4, 'pers'/'perspective'      -> General Perspective (Origin[0:2] center, Origin[2] height in km)

    Returns
    -------
    LineCollection
        The collection added to the axes (so you can further tweak/remove if needed).
    """
    ax = ax or plt.gca()
    proj_code, proj_params = _normalize_projection(Projection, Origin)

    # Capture the axes' pre-existing limits BEFORE adding any data (mirrors
    # world.m capturing XLim/YLim prior to plotting).
    pre_xlim = ax.get_xlim()
    pre_ylim = ax.get_ylim()

    data = _load_world_data(Path(mat_path))

    key = _KEYMAP.get(res.lower())
    if key is None:
        raise ValueError(
            "Invalid map resolution. Supported keys include: "
            + ", ".join(sorted(set(_KEYMAP.keys())))
        )

    # Base layer
    segs = list(data.get(key, []))

    # Extra bundles, matching your MATLAB behavior
    if res.lower() in ("na1",):
        # world lvl0 + major countries lvl1 (hstates)
        segs += data.get("hstates", [])
    elif res.lower() in ("na2",):
        # world.m draws usastates + usacounties for na2 under lambert/robinson,
        # but only usacounties under the plain lat-lon branch.
        if proj_code in ("lambert", "robinson"):
            segs += data.get("usastates", [])
        segs += data.get("usacounties", [])

    projected_segments = _project_segments(segs, proj_code, proj_params)

    lc = _as_linecollection(projected_segments, color=color, linewidth=linewidth)
    ax.add_collection(lc)

    _draw_projection_boundary(ax, proj_code, proj_params, color, linewidth)

    handled_limits = _maybe_set_axes_limits(ax, proj_code, pre_xlim, pre_ylim)
    if not handled_limits:
        ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    plt.sca(ax)
    return lc


# ------------------ Optional: one-time converter to NPZ for speed ------------------

def convert_world_mat_to_npz(mat_path: str | Path = _DEFAULT_MAT,
                             out_npz: str | Path = None) -> Path: # type: ignore
    """
    Convert world.mat to a compact NPZ for faster loads.
    Saves each layer as an object array of (N,2) float arrays.

    Usage:
        convert_world_mat_to_npz("world.mat")  # writes world_np.npz

    Then load with: np.load(npz_path, allow_pickle=True)
    """
    mat_path = Path(mat_path)
    npz_path = Path(out_npz) if out_npz else mat_path.with_name("world_np.npz")
    d = _load_world_mat(mat_path)   # uses cached loader
    # Save dict of lists; numpy must pickle lists-of-arrays
    np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in d.items()}) # pyright: ignore[reportArgumentType]
    return npz_path
