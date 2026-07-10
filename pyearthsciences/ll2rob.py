import numpy as np
from scipy.interpolate import PchipInterpolator

def ll2rob(londeg, latdeg, center_lon=0):
    """
    Transforms longitude and latitude in degrees to Robinson projection coordinates.
    
    Parameters:
    londeg (array_like): Longitude in degrees.
    latdeg (array_like): Latitude in degrees.
    center_lon (float): Central longitude in degrees (default 0).
    
    Returns:
    lon (numpy.ndarray): Projected X coordinates.
    lat (numpy.ndarray): Projected Y coordinates.
    """
    
    # Robinson projection lookup table (Latitude, X-factor, Y-factor)
    Rob = np.array([
        [-90.0000, 0.5322, -1.0000],
        [-85.0000, 0.5722, -0.9761],
        [-80.0000, 0.6213, -0.9394],
        [-75.0000, 0.6732, -0.8936],
        [-70.0000, 0.7186, -0.8435],
        [-65.0000, 0.7597, -0.7903],
        [-60.0000, 0.7986, -0.7346],
        [-55.0000, 0.8350, -0.6769],
        [-50.0000, 0.8679, -0.6176],
        [-45.0000, 0.8962, -0.5571],
        [-40.0000, 0.9216, -0.4958],
        [-35.0000, 0.9427, -0.4340],
        [-30.0000, 0.9600, -0.3720],
        [-25.0000, 0.9730, -0.3100],
        [-20.0000, 0.9822, -0.2480],
        [-15.0000, 0.9900, -0.1860],
        [-10.0000, 0.9954, -0.1240],
        [ -5.0000, 0.9986, -0.0620],
        [  0.0000, 1.0000,  0.0000],
        [  5.0000, 0.9986,  0.0620],
        [ 10.0000, 0.9954,  0.1240],
        [ 15.0000, 0.9900,  0.1860],
        [ 20.0000, 0.9822,  0.2480],
        [ 25.0000, 0.9730,  0.3100],
        [ 30.0000, 0.9600,  0.3720],
        [ 35.0000, 0.9427,  0.4340],
        [ 40.0000, 0.9216,  0.4958],
        [ 45.0000, 0.8962,  0.5571],
        [ 50.0000, 0.8679,  0.6176],
        [ 55.0000, 0.8350,  0.6769],
        [ 60.0000, 0.7986,  0.7346],
        [ 65.0000, 0.7597,  0.7903],
        [ 70.0000, 0.7186,  0.8435],
        [ 75.0000, 0.6732,  0.8936],
        [ 80.0000, 0.6213,  0.9394],
        [ 85.0000, 0.5722,  0.9761],
        [ 90.0000, 0.5322, 1.0000]
    ])

    R = 63.71  # Earth radius

    # Convert inputs to numpy float arrays (copies, since we mutate below)
    londeg = np.array(londeg, dtype=float)
    latdeg = np.array(latdeg, dtype=float)

    # Replicate MATLAB's logic: if inputs are 1D vectors of different sizes, create a grid.
    # MATLAB's `ndgrid` corresponds to numpy's `meshgrid` with indexing='ij'.
    did_expand = False
    if (londeg.size != latdeg.size) and (londeg.ndim <= 1) and (latdeg.ndim <= 1):
        londeg, latdeg = np.meshgrid(londeg, latdeg, indexing='ij')
        did_expand = True

    # --- Wrap longitude relative to center_lon, keeping +180 as +180 ---
    d = londeg - center_lon
    dlon = np.mod(d + 180, 360) - 180
    dlon[(dlon == -180) & (d > 0)] = 180

    # --- For polyline vectors: break segments across the seam to avoid long connectors ---
    is_polyline = (londeg.ndim == 1 and latdeg.ndim == 1
                   and londeg.shape == latdeg.shape and not did_expand)
    if is_polyline:
        finite = np.isfinite(dlon) & np.isfinite(latdeg)
        if np.any(finite):
            k = np.flatnonzero(finite)
            dk = np.abs(np.diff(dlon[k]))
            jump = k[1:][dk > 180]
            londeg[jump] = np.nan
            latdeg[jump] = np.nan
            dlon[jump] = np.nan

    if center_lon != 0:
        # Remove points at the 180 meridian (avoids seam artifacts near the poles)
        IX = ((londeg >= 180 - 1e-5) | (londeg <= -180 + 1e-5)) & \
             (((latdeg > 62) & (latdeg < 74)) | (latdeg < -84))
        latdeg[IX] = np.nan
        londeg[IX] = np.nan
        dlon[IX] = np.nan

    # Create Interpolators using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    # This matches MATLAB's interp1(..., 'pchip')
    rob_lats = Rob[:, 0]
    rob_x_vals = Rob[:, 1]
    rob_y_vals = Rob[:, 2]

    pchip_x = PchipInterpolator(rob_lats, rob_x_vals)
    pchip_y = PchipInterpolator(rob_lats, rob_y_vals)

    # Perform Interpolation (NaN latitudes produce NaN outputs, matching interp1)
    valid = np.isfinite(latdeg)
    X = np.full(latdeg.shape, np.nan)
    Y = np.full(latdeg.shape, np.nan)
    X[valid] = pchip_x(latdeg[valid])
    Y[valid] = pchip_y(latdeg[valid])

    # --- Robinson forward equations (spherical) ---
    # lon formula: 0.8487 * R * X * deg2rad(dlon)
    # lat formula: 1.3523 * R * Y
    lon_out = 0.8487 * R * X * np.deg2rad(dlon)
    lat_out = 1.3523 * R * Y

    return lon_out, lat_out