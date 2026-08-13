from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

ColorLike = Union[str, Tuple[float, float, float], Tuple[int, int, int], List[int]]

# ---------------------------------------------------------------------
# Load htmlcolors.csv ONCE (same directory as this file)
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_html_colors() -> dict:
    """
    Returns a dict: lowercased name -> (r, g, b) as floats in [0,1].
    Expects htmlcolors.csv in the same folder as hue.py.
    """
    here = Path(__file__).resolve().parent
    csv_path = here / "htmlcolors.csv"
    table = {}

    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["Name"].strip().lower()
                r = float(row["R"]) / 255.0
                g = float(row["G"]) / 255.0
                b = float(row["B"]) / 255.0
                table[name] = (r, g, b)
    else:
        # Fallback: use Matplotlib CSS4 colors if CSV not present
        for name, hexcode in mcolors.CSS4_COLORS.items():
            table[name.lower()] = mcolors.to_rgb(hexcode)
    return table


# ---------------------------------------------------------------------
# Your single-letter / short aliases (customized to your MATLAB values)
# Returned as floats in [0,1]
# ---------------------------------------------------------------------
_SHORTS = {
    "k": (59/255, 55/255, 53/255),
    "r": (223/255, 70/255, 51/255),
    "b": (16/255, 110/255, 167/255),
    "g": (40/255, 149/255, 50/255),
    "lb": (68/255, 157/255, 208/255),
    "o": (222/255, 130/255, 48/255),
    "y": (246/255, 211/255, 72/255),
    "br": (156/255, 95/255, 62/255),
    "pk": (217/255, 135/255, 151/255),
    "p": (105/255, 53/255, 136/255),
    "lg": (143/255, 188/255, 86/255),
    "gy": (180/255, 180/255, 180/255),
}

# ---------------------------------------------------------------------
# Predefined colormaps -> lists of color "stops". This is a straight,
# name-for-name, value-for-value mirror of hue.m's getpredefinedcmap
# (verified against the current hue.m; re-sync this whole block if hue.m's
# colors change again). Each stop can be an HTML name (str) or [R,G,B]
# integers. Order follows hue.m's own grouping comments.
# ---------------------------------------------------------------------
_PRESETS = {
    # -- white friendly --
    "jet2": ["royalblue", "cyan", "yellow", "red"],
    "jet3": [
        [201, 206, 239], [114, 140, 244], [34, 106, 252], [10, 186, 238], [48, 239, 249],
        [43, 201, 150], [34, 226, 21], [153, 252, 3], [234, 242, 47], [255, 181, 7],
        [249, 135, 40], [249, 76, 18], [226, 22, 107], [247, 109, 214],
    ],
    "gmao": [
        [242, 236, 252], [228, 217, 251], [200, 193, 250], [171, 168, 248], [118, 161, 222],
        [134, 207, 169], [226, 241, 101], [239, 213, 92], [241, 145, 91], [230, 102, 112], [200, 70, 159],
    ],
    "hrrr": [
        [208, 225, 242], [148, 196, 223], [74, 152, 201], [22, 100, 171], [16, 132, 70],
        [84, 180, 94], [162, 215, 106], [255, 246, 176], [252, 170, 95], [247, 132, 78], [237, 95, 60],
        [194, 27, 39], [165, 0, 37], [153, 0, 250],
    ],
    "cams": [
        [210, 214, 234], [167, 174, 214], [135, 145, 190], [162, 167, 144], [189, 188, 101], [215, 209, 56],
        [242, 230, 9], [243, 197, 5], [245, 164, 5], [247, 131, 4], [248, 97, 4], [250, 65, 1], [252, 31, 0],
    ],
    "aod": [[195, 231, 245], "skyblue", "gold", [232, 50, 35], "darkred"],
    "ncl": [
        [179, 227, 247], [108, 180, 222], [58, 136, 177], [60, 163, 93], [153, 199, 84],
        [250, 200, 87], [248, 110, 54], [226, 54, 44], [187, 25, 38], [137, 20, 28],
    ],
    "o3": ["skyblue", [145, 204, 113], "yellow", "orange", "salmon", "mediumvioletred"],
    "co": ["Wheat", [255, 255, 112], "orange", "crimson", "PaleVioletRed", "MediumPurple", "DarkTurquoise"],
    "sat": ["lightblue", "DarkTurquoise", "royalblue", "salmon", "pink"],
    "oc": [[145, 204, 113], "yellow", "orangered", "darkred"],
    "bc": [[145, 204, 113], "yellow", "orange", "MediumVioletRed", [152, 102, 203]],
    "hum": ["blanchedalmond", "wheat", [241, 229, 11], [145, 204, 113], "royalblue", "plum"],
    "wind": ["lightskyblue", [145, 204, 113], "yellow", "tomato", "pink"],
    "pm": ["wheat", "yellow", "tomato", "crimson", "darkred"],
    "pastel": [[221, 209, 231], "skyBlue", "yellow", "tomato", "pink"],
    "nox": [
        [178, 203, 225], [157, 176, 178], [217, 186, 109], [230, 176, 92],
        [224, 158, 83], [199, 117, 59], [146, 73, 34], [78, 31, 9],
    ],

    # -- full color --
    "emis": [[25, 62, 139], "skyblue", [145, 204, 113], "gold", [232, 50, 35]],
    "finn": [
        [247, 246, 246], [238, 237, 238], [229, 223, 232], [211, 208, 224], [192, 196, 220], [183, 200, 218],
        [178, 219, 217], [167, 216, 180], [158, 218, 128], [183, 221, 115], [230, 232, 109], [223, 163, 76], [216, 80, 47],
    ],
    "usgs": [
        [225, 230, 240], [199, 204, 225], [193, 204, 251], [183, 216, 251], [174, 227, 252], [165, 240, 253], [159, 252, 254],
        [156, 252, 203], [156, 251, 155], [201, 252, 106], [254, 254, 85], [251, 226, 76], [247, 199, 68], [244, 173, 61], [238, 104, 44],
    ],
    "blh": ["lightblue", "lightyellow", "sandybrown", "chocolate"],
    "temp": [
        [215, 190, 215], [184, 196, 229], [151, 202, 243], [128, 194, 247], [114, 172, 242], [100, 148, 236],
        [145, 204, 113], [180, 220, 77], [216, 237, 40], [253, 254, 2], [255, 196, 0], [255, 131, 0],
        [255, 69, 0], [255, 109, 67], [255, 150, 134], [255, 191, 202],
    ],
    "rainbow": ["red", [255, 119, 58], [255, 237, 70], [0, 248, 57], [0, 202, 251], [18, 51, 249], [179, 64, 250]],
    "ww": [
        [58, 121, 200], [103, 188, 176], [201, 226, 161], [252, 250, 190], [254, 212, 128],
        [253, 140, 80], [224, 82, 103], [177, 54, 121], [116, 31, 127], [68, 18, 110],
    ],
    "giss": [
        [152, 0, 16], [197, 21, 24], [242, 48, 35], [242, 76, 44], [241, 109, 55], [241, 145, 67],
        [240, 184, 80], [239, 231, 95], [182, 236, 87], [111, 228, 80], [62, 159, 116], [0, 88, 248],
        [47, 120, 248], [77, 157, 250], [111, 194, 251], [135, 222, 252],
    ],
    "acc": [
        [184, 243, 254], [149, 231, 253], [123, 217, 253], [43, 196, 252], [6, 162, 226], [0, 109, 224],
        [111, 73, 194], [181, 94, 177], [224, 62, 220], [240, 152, 221],
    ],
    "frp": ["lightyellow", "orange", [232, 50, 35], "indigo"],
    "bright": [
        [241, 248, 250], [187, 222, 238], [150, 191, 220], [123, 162, 208], [97, 131, 195], [128, 92, 155],
        [184, 111, 143], [231, 135, 136], [254, 170, 138], [255, 205, 138], [255, 238, 199],
    ],

    # -- black friendly --
    "ext": [
        [0, 2, 46], [10, 38, 75], [23, 64, 96], [37, 93, 119], [47, 113, 135], [57, 132, 151],
        [65, 149, 166], [79, 177, 187], [92, 203, 207], [101, 223, 223], [111, 238, 146], [171, 246, 77],
        [254, 245, 82], [244, 208, 72], [234, 171, 62], [225, 135, 52], [215, 98, 42], [205, 61, 32],
        [218, 112, 146], [255, 181, 192],
    ],
    "pro": ["black", "midnightblue", "CadetBlue", "LemonChiffon", "orange", "red", "darkred"],
    "city": [[2, 32, 44], [74, 55, 143], [167, 85, 118], [252, 133, 69], [232, 244, 97]],
    "clouds": ["black", "gray", "whitesmoke", "royalblue", "limegreen", "yellow", "darkorange", "firebrick", "pink", "lavender"],

    # -- divergent --
    "ssta": [
        [122, 0, 112], [85, 60, 179], [1, 24, 199], [0, 109, 255], [1, 235, 255], [254, 254, 254],
        [242, 245, 0], [240, 185, 2], [249, 110, 0], [244, 35, 1], [150, 23, 0],
    ],
    "sea": [
        [0, 3, 127], [0, 6, 165], [0, 9, 204], [25, 25, 255], [75, 76, 255], [127, 127, 255], [178, 179, 255], [204, 203, 255], [254, 254, 254],
        [255, 254, 24], [255, 203, 25], [255, 178, 25], [254, 127, 24], [255, 76, 26], [255, 25, 26], [204, 0, 0], [152, 0, 1],
    ],
    "pan": [
        [4, 14, 216], [32, 80, 255], [65, 150, 255], [109, 193, 255], [134, 217, 255], [156, 238, 255], [175, 245, 255], [206, 255, 255], [254, 254, 254],
        [255, 254, 71], [255, 235, 0], [255, 196, 0], [255, 144, 0], [255, 72, 0], [255, 0, 0], [213, 0, 0], [158, 0, 0],
    ],
    "ufs": [
        [7, 30, 70], [7, 46, 108], [9, 87, 156], [33, 113, 181], [66, 146, 199], [90, 160, 205], [120, 191, 214], [170, 220, 230], [219, 245, 255],
        [255, 255, 255], [255, 224, 224], [252, 187, 169], [252, 146, 114], [251, 106, 74], [240, 59, 43], [204, 23, 30], [166, 15, 20], [120, 10, 16], [95, 0, 0],
    ],
    "br1": [[36, 126, 177], [146, 190, 216], [254, 254, 254], [251, 136, 83], [212, 55, 72]],
    "br2": [[38, 66, 155], [88, 197, 219], [254, 254, 254], [255, 148, 99], [229, 35, 51]],
    "br3": [[91, 81, 157], [114, 141, 166], [186, 200, 227], [252, 252, 252], [250, 214, 150], [229, 107, 79], [146, 29, 67]],
    "br4": [[91, 81, 157], [124, 191, 166], [223, 244, 163], [252, 252, 252], [250, 224, 150], [229, 117, 79], [146, 29, 67]],
    "cpc": [
        [34, 24, 82], [1, 93, 161], [119, 181, 226], [191, 203, 228], [254, 254, 254], [231, 177, 103], [218, 87, 49], [178, 46, 5], [112, 33, 0],
    ],
    "rh": ["tan", "wheat", [254, 254, 254], "lightskyblue", "royalblue"],
    "daod": [
        [94, 54, 138], [127, 112, 166], [165, 158, 197], [198, 197, 220], [230, 231, 241], [250, 250, 250],
        [255, 230, 201], [255, 202, 147], [254, 171, 95], [231, 129, 50], [197, 97, 38],
    ],
    "melt": [
        [0, 60, 109], [5, 95, 129], [64, 128, 154], [100, 163, 175], [137, 196, 199], [247, 245, 242],
        [255, 200, 154], [255, 178, 123], [255, 156, 91], [255, 135, 64], [255, 114, 47],
    ],
    "grav": [
        [38, 66, 155], [0, 126, 181], [48, 178, 207], [128, 217, 230], [205, 241, 246], [254, 252, 206],
        [255, 229, 164], [255, 180, 117], [255, 116, 81], [255, 40, 49], [229, 35, 51],
    ],
    "ceres": [
        [36, 126, 177], [77, 174, 160], [131, 204, 157], [188, 226, 154], [233, 244, 158], [255, 253, 187],
        [255, 225, 146], [255, 184, 110], [251, 136, 83], [238, 87, 67], [212, 55, 72],
    ],
    "ppa": [
        [184, 119, 60], [212, 142, 79], [234, 196, 142], [249, 233, 206], [254, 254, 254],
        [156, 226, 218], [59, 186, 175], [0, 148, 140], [0, 134, 125],
    ],
    "pp2": [
        [80, 47, 47], [147, 70, 57], [187, 109, 51], [237, 209, 145], [254, 254, 254], [179, 217, 171], [72, 180, 48], [3, 120, 20], [40, 83, 0],
    ],
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_rgb01(c: ColorLike, html_table: dict) -> Tuple[float, float, float]:
    """
    Convert a color spec to (r,g,b) in [0,1].
    - str: try _SHORTS, then htmlcolors table, then matplotlib parser
    - tuple/list of 3 ints: treat as 0..255
    - tuple/list of 3 floats: assume already 0..1
    """
    if isinstance(c, str):
        key = c.strip().lower()
        if key in _SHORTS:
            return _SHORTS[key]
        if key in html_table:
            return html_table[key]
        # last resort: Matplotlib's name/hex parser
        return mcolors.to_rgb(c)

    if isinstance(c, (list, tuple)) and len(c) == 3:
        a, b, d = c[0], c[1], c[2]
        # ints?
        if all(isinstance(x, (int, np.integer)) for x in (a, b, d)):
            return (a/255.0, b/255.0, d/255.0)
        # floats?
        if all(isinstance(x, (float, np.floating, int, np.integer)) for x in (a, b, d)):
            # assume already in 0..1
            return (float(a), float(b), float(d))

    raise ValueError(f"Unrecognized color spec: {c!r}")


def _interp_colormap(stops: List[ColorLike], N: int, name: str, html_table: dict,
                      use_log: bool = False) -> ListedColormap:
    """
    Make a discrete ListedColormap with N colors by interpolation between the
    provided color stops (>=2). Mirrors hue.m's interp1 over Xi=linspace(0,1,.)
    control points, with an optional log-spaced query grid (Xq).
    """
    if len(stops) < 2:
        # degenerate: just repeat the single color N times
        rgb = np.array([_to_rgb01(stops[0], html_table)])
        arr = np.repeat(rgb, N, axis=0)
        return ListedColormap(arr, name=name)

    # normalize the control points
    stop_rgb = np.array([_to_rgb01(s, html_table) for s in stops], dtype=float)
    x_ctrl = np.linspace(0, 1, len(stops))
    if use_log and N > 1:
        x = (np.logspace(0, 1, N) - 1) / 9
    else:
        x = np.linspace(0, 1, N)
    out = np.empty((N, 3), dtype=float)
    for j in range(3):
        out[:, j] = np.interp(x, x_ctrl, stop_rgb[:, j])
    return ListedColormap(out, name=name)


def _is_log_token(arg) -> bool:
    return isinstance(arg, str) and arg.strip().lower() == "log"


# ---------------------------------------------------------------------
# Public API: hue
# ---------------------------------------------------------------------
def hue(*args: Union[str, ColorLike, int]) -> Union[Tuple[float, float, float], ListedColormap]:
    """
    MATLAB-like hue():
      - hue('red') -> (r,g,b) floats in [0,1]
      - hue('gmao') or hue('gmao', 64) -> ListedColormap
      - hue('royalblue','cyan','yellow','red', 256) -> ListedColormap
      - hue([254,254,254], 'royalblue', 'cyan', 'yellow', 'red', 128) -> ListedColormap
      - hue('white', 'jet2', 'black') -> preset(s) expanded in place, mixed with
        individual colors, all interpolated into one colormap
      - hue('gmao', 64, 'log') -> log-spaced interpolation between stops

    Notes:
      * Among args[1:], a scalar int is interpreted as N (colormap length) and
        the literal string 'log' switches to log-spaced interpolation. args[0]
        is always treated as a color/preset name (mirrors hue.m).
      * Preset names are expanded in place, so they can be freely combined
        with individual colors and other presets.
      * If exactly one name remains after expansion and it is NOT a preset,
        returns a single color.
      * Otherwise, returns a colormap interpolating the resulting stops.
    """
    if not args:
        raise ValueError("hue() requires at least one argument.")

    N = 256
    use_log = False
    names = [args[0]]
    for arg in args[1:]:
        if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
            N = int(arg)
        elif _is_log_token(arg):
            use_log = True
        else:
            names.append(arg)

    # Expand any predefined colormap name in-place; non-preset inputs are
    # kept as individual colors so callers can prepend/append freely.
    expanded_names: list = []
    found_cmap = False
    for name in names:
        key = name.strip().lower() if isinstance(name, str) else None
        if key is not None and key in _PRESETS:
            expanded_names.extend(_PRESETS[key])
            found_cmap = True
        else:
            expanded_names.append(name)
    names = expanded_names

    html_table = _load_html_colors()

    # Single remaining stop that isn't a preset => a single color
    if len(names) == 1 and not found_cmap:
        return _to_rgb01(names[0], html_table)

    return _interp_colormap(names, N, name="custom", html_table=html_table, use_log=use_log)


# ---------------------------------------------------------------------
# Optional: register your presets in Matplotlib so you can call them by string
# ---------------------------------------------------------------------
def register_hue_cmaps(default_N: int = 256, overwrite: bool = False) -> None:
    """
    Register all _PRESETS with Matplotlib's global colormap registry.
    After this, you can do: plt.colormaps['gmao'] or Colormap='gmao'.
    """
    for name in _PRESETS:
        cmap = _interp_colormap(_PRESETS[name], default_N, name=name, html_table=_load_html_colors())
        try:
            mcolors.register_cmap(name=name, cmap=cmap, override_builtin=overwrite)
        except TypeError:
            # Older Matplotlib: register_cmap doesn't have override_builtin
            mcolors.register_cmap(name=name, cmap=cmap)