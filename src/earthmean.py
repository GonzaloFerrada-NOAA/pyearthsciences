import numpy as np


def earthmean(data, gridarea, nanflag: str = 'includenan') -> float:
    """
    Area-weighted global mean of a 2-D gridded variable.

    Parameters
    ----------
    data     : 2-D array
    gridarea : 2-D array of grid-cell areas (same shape as data)
    nanflag  : 'omitnan' or 'includenan' (default)

    Returns
    -------
    Area-weighted mean.
    """
    data = np.asarray(data, dtype=float)
    gridarea = np.asarray(gridarea, dtype=float)

    if data.shape != gridarea.shape:
        raise ValueError("data and gridarea must have the same size.")

    if np.any(gridarea < 0):
        raise ValueError("gridarea must be nonnegative.")

    flag = nanflag.lower()
    if flag == 'omitnan':
        mask = ~np.isnan(data) & ~np.isnan(gridarea)
        wsum = np.sum(gridarea[mask])
        if wsum == 0:
            return np.nan
        return float(np.sum(data[mask] * gridarea[mask]) / wsum)

    if flag == 'includenan':
        wsum = np.sum(gridarea)
        if wsum == 0:
            return np.nan
        return float(np.sum(data * gridarea) / wsum)

    raise ValueError("nanflag must be 'omitnan' or 'includenan'.")
