                           
import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.timelib import Time

class SatellitePropagator:
    """
    Module B: SatellitePropagator (Physics Engine)
    Responsibility: Handle all orbital math.
    """
    def __init__(self, satellite: EarthSatellite):
        self.satellite = satellite
        self.ts = load.timescale()

    def get_position_and_velocity(self, t: Time, observer_lat: float, observer_lon: float, observer_alt: float = 0.0):
        """
        Returns the satellite's position and velocity relative to a specific ground location.
        
        Args:
            t: Skyfield Time object
            observer_lat: Latitude in degrees
            observer_lon: Longitude in degrees
            observer_alt: Altitude in meters
            
        Returns:
            position (numpy array): [x, y, z] in meters (relative to observer)
            velocity (numpy array): [vx, vy, vz] in m/s (relative to observer)
        """
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt)
        difference = self.satellite - observer
        topocentric = difference.at(t)
        
        pos, vel = topocentric.position.m, topocentric.velocity.m_per_s
        return pos, vel

    def calculate_range_rate(self, t: Time, lat: float, lon: float) -> float:
        """
        Helper to calculate range rate (relative velocity projected onto line of sight).
        v_rel = (r . v) / |r|
        """
        pos, vel = self.get_position_and_velocity(t, lat, lon)
        r_mag = np.linalg.norm(pos)
        if r_mag == 0:
            return 0.0
        range_rate = np.dot(pos, vel) / r_mag
        return range_rate
