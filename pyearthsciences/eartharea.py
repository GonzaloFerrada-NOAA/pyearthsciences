import numpy as np


def eartharea(lon, lat) -> np.ndarray:
    """
    Grid-cell area (m^2) for a regular 1-D lon/lat grid, assuming each cell's
    edges sit halfway to its neighbors.

    Parameters
    ----------
    lon, lat : 1-D arrays (degrees), uniformly spaced.

    Returns
    -------
    2-D array of shape (len(lon), len(lat)) with grid-cell areas in m^2,
    in the same lat order as the input.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat should be 1-D.")

    lat_is_descending = lat.size > 1 and lat[1] < lat[0]
    lat_calc = np.flip(lat) if lat_is_descending else lat

    dlon = abs(np.mean(np.diff(lon)))
    dlat = np.mean(np.diff(lat_calc))

    R = 6371000.0  # Earth's radius in meters

    # Build the area grid using ascending latitude, then restore input order.
    lon_grid, lat_grid = np.meshgrid(lon, lat_calc, indexing='ij')

    dlon_rad = np.deg2rad(dlon)

    # Latitude edges (half-cell shift)
    lat1 = np.deg2rad(lat_grid - dlat / 2)  # southern edge
    lat2 = np.deg2rad(lat_grid + dlat / 2)  # northern edge

    garea_calc = R**2 * dlon_rad * (np.sin(lat2) - np.sin(lat1))

    if lat_is_descending:
        return np.fliplr(garea_calc)
    return garea_calc
