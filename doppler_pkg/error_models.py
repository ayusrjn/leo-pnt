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
        
                                                
        if clock_type == ClockType.OCXO:
            self.h0 = 1e-11           
            self.h2 = 1e-12                         
        else:         
            self.h0 = 1e-12
            self.h2 = 1e-14
            
                        
        self.bias = np.random.uniform(-1e-4, 1e-4)          
        self.drift = np.random.uniform(-1e-9, 1e-9)      
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
            
                                                 
                                                                           
                                       
        
                                          
                                       
                                
                                    
                        
        
                                        
        w_bias = np.random.normal(0, self.h0 * np.sqrt(dt))
        
                                                   
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
                         
        el, az, dist = self._get_el_az_dist(sat_pos, user_pos)
        if el < 0: return 0.0
        
                                               
                              
                                                 
                                          
        
                                                       
        vtec_delay = 5.0                                     
        
                                              
        slant_factor = 1.0 / np.sin(np.radians(el) + 0.1)                    
        
                                                 
                                       
        leo_factor = 0.4
        
        return vtec_delay * slant_factor * leo_factor

    def get_troposphere_delay(self, sat_pos, user_pos):
        """
        Compute Tropospheric delay (meters).
        Standard Saastamoinen model.
        """
        el, az, dist = self._get_el_az_dist(sat_pos, user_pos)
        if el < 0: return 0.0
        
                            
        ztd = 2.3
        
                          
        map_fn = 1.0 / np.sin(np.radians(el) + 0.05)
        
        return ztd * map_fn

    def _get_el_az_dist(self, sat_pos, user_pos):
                                       
                               
                              
        
        dx = sat_pos[0] - user_pos[0]
        dy = sat_pos[1] - user_pos[1]
        dz = sat_pos[2] - user_pos[2]
        
                                   
        r = np.linalg.norm(user_pos)
        lat = np.arcsin(user_pos[2] / r)
        lon = np.arctan2(user_pos[1], user_pos[0])
        
                                     
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
        self.code_std = 0.30        
        self.phase_std = 0.003       
        
    def add_noise(self, range_m):
        code = range_m + np.random.normal(0, self.code_std)
        phase = range_m + np.random.normal(0, self.phase_std)
        return code, phase
