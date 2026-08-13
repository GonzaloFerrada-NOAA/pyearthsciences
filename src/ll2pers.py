import numpy as np

def ll2pers(longitude_degrees, latitude_degrees, center, Height_viewpoint=35786):
    """
    Converts latitude and longitude to X, Y in General Perspective Projection.
    
    Parameters:
    longitude_degrees (array_like): Array of longitudes (in degrees).
    latitude_degrees (array_like): Array of latitudes (in degrees).
    center (list or array_like): [center_lon, center_lat] (in degrees) defining the viewpoint center.
    Height_viewpoint (float, optional): Height of viewpoint above Earth in km. 
                                        Default is 35,786 km (Geostationary orbit).
    
    Returns:
    X (numpy.ndarray): Projected X coordinates.
    Y (numpy.ndarray): Projected Y coordinates.
    """
    
    # Constants
    R = 6378.0  # Radius of Earth in km

    # Ensure inputs are numpy arrays
    lon_deg = np.asarray(longitude_degrees)
    lat_deg = np.asarray(latitude_degrees)
    
    # Convert inputs to radians
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    center_lon = np.deg2rad(center[0])
    center_lat = np.deg2rad(center[1])

    # Auxiliary variables
    rho = R + Height_viewpoint
    
    # Calculate cos_c (Cosine of the angular distance)
    # Formula: sin(lat0)sin(lat) + cos(lat0)cos(lat)cos(lon - lon0)
    cos_c = (np.sin(center_lat) * np.sin(lat) + 
             np.cos(center_lat) * np.cos(lat) * np.cos(lon - center_lon))

    # Visibility condition
    # From MATLAB: visible = (rho * cos_c > R);
    # This filters out points on the far side of the Earth or beyond the horizon
    visible = (rho * cos_c > R)

    # Initialize output with NaNs
    X = np.full(lon_deg.shape, np.nan)
    Y = np.full(lat_deg.shape, np.nan)

    # Calculate projected coordinates only for visible points
    if np.any(visible):
        # Calculate the denominator for the visible points
        denom = rho - R * cos_c[visible]
        
        # Calculate X
        # MATLAB: X = rho .* cos(lat) .* sin(lon - center_lon) ./ denom
        X[visible] = (rho * np.cos(lat[visible]) * np.sin(lon[visible] - center_lon)) / denom
        
        # Calculate Y
        # MATLAB: Y = rho .* (cos(center_lat) .* sin(lat) - ... ) ./ denom
        Y[visible] = (rho * (np.cos(center_lat) * np.sin(lat[visible]) - 
                             np.sin(center_lat) * np.cos(lat[visible]) * np.cos(lon[visible] - center_lon))) / denom

    return X, Y