import numpy as np

def Lambert(lat, lon, lat1, lat2, lat0, lon0):
    """
    Generalized Lambert Conformal Conic Projection.
    Calculates raw projected coordinates X, Y (km).
    
    Parameters:
    lat, lon    : Input coordinates (degrees)
    lat1, lat2  : Standard parallels (True Latitudes) (degrees)
    lat0, lon0  : Center of the projection (degrees)
    """
    # 1. Constants (matches ll2lamb.m's spherical radius)
    R_earth = 6377.39715500000

    # 2. Convert to radians
    lat_rad = np.deg2rad(lat)
    # Wrap lon around lon0 so (lon - lon0) stays in [-180, 180)
    dlon = np.mod(np.asarray(lon) - lon0 + 180, 360) - 180
    lon_rad = np.deg2rad(lon0 + dlon)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    
    # 3. Calculate n
    # Handle the case where lat1 == lat2 (Tangent cone) to avoid division by zero
    if abs(lat1 - lat2) < 1e-10:
        n = np.sin(lat1_rad)
    else:
        # MATLAB formula:
        # n = (log(cos(lat1) ./ cos(lat2))) ./ (log(tan(pi/4 + lat2/2) ./ tan(pi/4 + lat1/2)))
        numerator = np.log(np.cos(lat1_rad) / np.cos(lat2_rad))
        denominator = np.log(np.tan(np.pi/4 + lat2_rad/2) / np.tan(np.pi/4 + lat1_rad/2))
        n = numerator / denominator
    
    # 4. Calculate F
    # F = (cos(lat1) .* (tan(pi/4 + lat1/2).^n)) / n
    F = (np.cos(lat1_rad) * (np.tan(np.pi/4 + lat1_rad/2)**n)) / n
    
    # 5. Calculate rho and rho0
    # rho = (Re * F) ./ (tan(pi/4 + lat/2).^n)
    rho = (R_earth * F) / (np.tan(np.pi/4 + lat_rad/2)**n)
    rho0 = (R_earth * F) / (np.tan(np.pi/4 + lat0_rad/2)**n)
    
    # 6. Calculate theta
    # theta = n * (lon - lon0)
    theta = n * (lon_rad - lon0_rad)
    
    # 7. Calculate X and Y
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    
    return x, y

def ll2lamb(lat, lon, specs):
    """
    Wrapper for Lambert projection.
    
    specs: List/Array.
           If len 2: [lat0, lon0]. Tangent cone: lat1=lat2=lat0.
           If len 4: [lat1, lat2, lat0, lon0].
    
    Returns: x, y (km) centered at (lat0, lon0).
    """
    specs = np.atleast_1d(specs)
    
    if len(specs) == 2:
        lat0, lon0 = specs[0], specs[1]
        lat1 = lat0
        lat2 = lat0  # tangent cone, matches ll2lamb.m's 2-element case
    elif len(specs) == 4:
        lat1, lat2, lat0, lon0 = specs[0], specs[1], specs[2], specs[3]
    else:
        raise ValueError("Projection specs must be length 2 or 4.")
    
    # Calculate projection for data
    X, Y = Lambert(lat, lon, lat1, lat2, lat0, lon0)
    
    # Calculate projection for center (to shift origin to 0,0)
    X0, Y0 = Lambert(lat0, lon0, lat1, lat2, lat0, lon0)
    
    x = X - X0
    y = Y - Y0
    
    return x, y