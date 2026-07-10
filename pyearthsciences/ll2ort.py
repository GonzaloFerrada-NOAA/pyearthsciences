import numpy as np

def ll2ort(lon, lat, param):
    """
    Transforms longitude and latitude in degrees to Orthographic projection coordinates.
    
    Parameters:
    lon (array_like): Longitude in degrees.
    lat (array_like): Latitude in degrees.
    param (list or array_like): [center_longitude, center_latitude] in degrees.
    
    Returns:
    X (numpy.ndarray): Projected X coordinates.
    Y (numpy.ndarray): Projected Y coordinates.
    """
    # Radius constant as defined in the MATLAB file
    R = 100.0
    
    # Ensure inputs are numpy arrays for element-wise operations
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    
    # Extract projection center parameters
    center_lon = param[0]
    center_lat = param[1]
    
    # Convert degrees to radians for numpy trig functions
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    clon_rad = np.deg2rad(center_lon)
    clat_rad = np.deg2rad(center_lat)
    
    # Calculate longitude difference
    lon_diff = lon_rad - clon_rad
    
    # Calculate X
    # MATLAB: X = R .* cosd(lat) .* sind(lon - param(1));
    X = R * np.cos(lat_rad) * np.sin(lon_diff)
    
    # Calculate Y
    # MATLAB: Y = R .* (cosd(param(2)) .* sind(lat) - sind(param(2)) .* cosd(lat) .* cosd(lon - param(1)));
    Y = R * (np.cos(clat_rad) * np.sin(lat_rad) - 
             np.sin(clat_rad) * np.cos(lat_rad) * np.cos(lon_diff))
    
    # Calculate cosine of the angular distance from center (cosc)
    # MATLAB: cosc = sind(param(2)) .* sind(lat) + cosd(param(2)) .* cosd(lat) .* cosd(lon - param(1));
    cosc = (np.sin(clat_rad) * np.sin(lat_rad) + 
            np.cos(clat_rad) * np.cos(lat_rad) * np.cos(lon_diff))
    
    # Filter points outside the visible hemisphere
    # MATLAB uses acos(cosc). We clip input to [-1, 1] to avoid warnings from floating point errors.
    cosc_clipped = np.clip(cosc, -1.0, 1.0)
    angular_dist = np.arccos(cosc_clipped)
    
    # MATLAB condition: acos(cosc) < -pi/2 | acos(cosc) > pi/2
    # Note: acos is always >= 0, so we only check > pi/2 (points on the far side of the globe)
    mask = angular_dist > (np.pi / 2)
    
    # Apply NaN to hidden points
    # We ensure X and Y are floating point to store NaNs (numpy arrays from trig are floats by default)
    if np.any(mask):
        X[mask] = np.nan
        Y[mask] = np.nan
        
    return X, Y