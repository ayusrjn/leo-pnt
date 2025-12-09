# doppler_pkg/utils.py
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in km."""
    R = 6371.0  # Earth radius in km
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c
