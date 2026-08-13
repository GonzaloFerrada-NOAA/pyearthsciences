from .spatial import spatial
from .hue import hue
from .world import world
from .earthmean import earthmean
from .eartharea import eartharea
from .metrics import metrics
from .figid import figid
from .msg import msg
from .reorganizeaxes import reorganizeaxes
from .reorganizecolorbar import reorganizecolorbar
from .axtags import mark_colorbar_axes, is_colorbar_axes
from .pixelpos import move_axes

# If you want everything in one namespace:
__all__ = ["spatial", "hue", "world", "earthmean", "eartharea", "metrics", "figid", "msg",
           "reorganizeaxes", "reorganizecolorbar", "mark_colorbar_axes", "is_colorbar_axes",
           "move_axes"]

