"""Shared projection-code resolution for world()/spatial()/ticks2geo().

Accepts either a numeric code (0-4) or a case-insensitive string alias, so
callers don't have to memorize the numbers.
"""
from __future__ import annotations

import numpy as np

_ALIASES = {
    'latlon': 0, 'lat-lon': 0, 'll': 0,
    'rob': 1, 'robinson': 1,
    'lambert': 2, 'lamb': 2,
    'orth': 3, 'ort': 3, 'orthogonal': 3, 'orthographic': 3,
    'pers': 4, 'perspective': 4,
}

VALID_CODES = (0, 1, 2, 3, 4)


def resolve_projection_code(projection) -> int:
    """Normalize a Projection argument (int code or string alias) to 0-4."""
    if projection is None:
        return 0

    if isinstance(projection, str):
        key = projection.strip().lower()
        if key in _ALIASES:
            return _ALIASES[key]
        try:
            code = int(key)
        except ValueError:
            raise ValueError(
                f"Unknown projection '{projection}'. Use a code 0-4 or one of: "
                "'latlon'/'lat-lon', 'lambert'/'lamb', 'rob'/'robinson', "
                "'orth'/'ort'/'orthogonal', 'pers'/'perspective'."
            )
    else:
        arr = np.atleast_1d(projection)
        if arr.size != 1:
            raise ValueError(
                "Projection must be a scalar code or string alias; "
                "pass projection parameters via Origin."
            )
        code = int(arr[0])

    if code not in VALID_CODES:
        raise ValueError(f"Unsupported projection code {code}. Use 0-4.")
    return code
