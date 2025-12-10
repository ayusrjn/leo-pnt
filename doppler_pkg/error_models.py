import numpy as np
from enum import Enum

class ClockType(Enum):
    OCXO = "OCXO"
    ATOMIC = "ATOMIC"

class ClockErrorModel:
    """
    Simulates satellite clock errors (bias, drift, drift rate) + random noise.
    """
    def __init__(self, clock_type: ClockType):
        self.clock_type = clock_type
        
        # Allan Deviation / Stability parameters
        if clock_type == ClockType.OCXO:
            self.h0 = 1e-11 # White FM
            self.h2 = 1e-12 # Random Walk FM (Guess)
        else: # ATOMIC
            self.h0 = 1e-12
            self.h2 = 1e-14
            
        # Initial states
        self.bias = np.random.uniform(-1e-4, 1e-4) # seconds
        self.drift = np.random.uniform(-1e-9, 1e-9) # s/s
        self.drift_rate = 0.0
        
        self.last_update = 0.0

    def get_error(self, t: float) -> float:
        """
        Get clock error at time t (seconds from start).
        Updates internal state with random noise.
        """
        dt = t - self.last_update
        if dt <= 0:
            return self.bias
            
        # Add noise (Random Walk Frequency Noise)
        # Phase variance ~ h0/2 * dt + 2*pi^2/3 * h2 * dt^3 ... simplified:
        # Just add random walk to drift
        
        # Random walk on frequency (drift)
        # sigma_drift = sqrt(h2 * dt) ?
        # Simplified simulation:
        # bias += drift * dt + noise
        # drift += noise
        
        # White FM noise on phase (bias)
        w_bias = np.random.normal(0, self.h0 * np.sqrt(dt))
        
        # Random Walk FM noise on frequency (drift)
        w_drift = np.random.normal(0, self.h2 * np.sqrt(dt))
        
        self.bias += self.drift * dt + w_bias
        self.drift += w_drift
        
        self.last_update = t
        return self.bias

class AtmosphericErrorModel:
    """
    Simulates Ionospheric and Tropospheric delays.
    """
    def __init__(self):
        pass
        
    def get_ionosphere_delay(self, sat_pos, user_pos, t):
        """
        Compute Ionospheric delay (meters).
        LEO satellites are inside the ionosphere, so error is reduced.
        Simple model: Klobuchar-like mapping but scaled down.
        """
        # Elevation angle
        el, az, dist = self._get_el_az_dist(sat_pos, user_pos)
        if el < 0: return 0.0
        
        # Vertical TEC (Total Electron Content)
        # Simple diurnal model
        # VTEC varies with local time (sun angle)
        # Max at 14:00 local, Min at 02:00
        
        # Just a constant VTEC for now, mapped to slant
        vtec_delay = 5.0 # meters (GPS L1 is ~2-10m vertical)
        
        # Mapping function: 1 / sin(el) approx
        slant_factor = 1.0 / np.sin(np.radians(el) + 0.1) # Avoid singularity
        
        # LEO Reduction Factor (40-85% reduction)
        # Let's say 0.4 (60% reduction)
        leo_factor = 0.4
        
        return vtec_delay * slant_factor * leo_factor

    def get_troposphere_delay(self, sat_pos, user_pos):
        """
        Compute Tropospheric delay (meters).
        Standard Saastamoinen model.
        """
        el, az, dist = self._get_el_az_dist(sat_pos, user_pos)
        if el < 0: return 0.0
        
        # Zenith delay ~2.3m
        ztd = 2.3
        
        # Mapping function
        map_fn = 1.0 / np.sin(np.radians(el) + 0.05)
        
        return ztd * map_fn

    def _get_el_az_dist(self, sat_pos, user_pos):
        # Simple ECEF to ENU conversion
        # user_pos is [x, y, z]
        # sat_pos is [x, y, z]
        
        dx = sat_pos[0] - user_pos[0]
        dy = sat_pos[1] - user_pos[1]
        dz = sat_pos[2] - user_pos[2]
        
        # User Lat/Lon for rotation
        r = np.linalg.norm(user_pos)
        lat = np.arcsin(user_pos[2] / r)
        lon = np.arctan2(user_pos[1], user_pos[0])
        
        # Rotation matrix ECEF -> ENU
        sl = np.sin(lat)
        cl = np.cos(lat)
        so = np.sin(lon)
        co = np.cos(lon)
        
        R = np.array([
            [-so, co, 0],
            [-sl*co, -sl*so, cl],
            [cl*co, cl*so, sl]
        ])
        
        enu = R @ np.array([dx, dy, dz])
        e, n, u = enu
        
        dist = np.linalg.norm(enu)
        el = np.degrees(np.arcsin(u / dist))
        az = np.degrees(np.arctan2(e, n))
        
        return el, az, dist

class MeasurementNoiseModel:
    """
    Adds white Gaussian noise to measurements.
    """
    def __init__(self):
        self.code_std = 0.30 # 30 cm
        self.phase_std = 0.003 # 3 mm
        
    def add_noise(self, range_m):
        code = range_m + np.random.normal(0, self.code_std)
        phase = range_m + np.random.normal(0, self.phase_std)
        return code, phase
