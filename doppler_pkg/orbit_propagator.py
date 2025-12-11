import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import Geocentric
from skyfield.timelib import Time
from .constellation import KeplerianElements

           
GM = 3.986004418e14                                             
RE = 6378137.0                         
J2 = 1.08262668e-3                   
CD = 2.2                                             
CR = 1.8                                             
AREA_MASS_RATIO = 0.01                              
P_SUN = 4.56e-6                                                

class ForceModel:
    def __init__(self):
        self.ts = load.timescale()
        self.eph = load('de421.bsp')                   
        self.sun = self.eph['sun']
        self.earth = self.eph['earth']

    def compute_acceleration(self, t_jd: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute total acceleration in ECI frame.
        r: Position vector (m)
        v: Velocity vector (m/s)
        t_jd: Time in Julian Date
        """
        r_mag = np.linalg.norm(r)
        
                             
                         
        a_grav = -GM / (r_mag**3) * r
        
                            
                                           
                  
                                                                  
                                                                  
                                                                  
        
        z2 = r[2]**2
        r2 = r_mag**2
        factor_j2 = 1.5 * J2 * (RE**2 / r2)
        tx = (1 - 5 * z2 / r2)
        tz = (3 - 5 * z2 / r2)
        
        a_j2 = np.zeros(3)
        a_j2[0] = a_grav[0] * factor_j2 * tx
        a_j2[1] = a_grav[1] * factor_j2 * tx
        a_j2[2] = a_grav[2] * factor_j2 * tz                                          
        
                                                                                              
                                                                    
                                                                  
        
        pre_factor = -1.5 * J2 * GM * (RE**2) / (r_mag**5)
        a_j2_vec = np.zeros(3)
        a_j2_vec[0] = pre_factor * r[0] * (1 - 5 * z2 / r2)
        a_j2_vec[1] = pre_factor * r[1] * (1 - 5 * z2 / r2)
        a_j2_vec[2] = pre_factor * r[2] * (3 - 5 * z2 / r2)

                             
                                                                 
                                             
                                         
        h = r_mag - RE
        rho = self._get_density(h)
        
                                                         
                                            
                                       
                           
        omega_e = 7.2921159e-5
        v_rel = np.array([
            v[0] + omega_e * r[1],
            v[1] - omega_e * r[0],
            v[2]
        ])
        v_rel_mag = np.linalg.norm(v_rel)
        
        a_drag = -0.5 * rho * (v_rel_mag**2) * CD * AREA_MASS_RATIO * (v_rel / v_rel_mag)
        
                                     
                         
                                            
        t = self.ts.tt_jd(t_jd)
                             
        sun_pos = self.earth.at(t).observe(self.sun).position.m
        sun_vec = sun_pos - r                                                                     
                                            
                                                                 
                                                              
        r_sun_sat = sun_pos - r
        dist_sun = np.linalg.norm(r_sun_sat)
        u_sun = r_sun_sat / dist_sun
        
                                                 
        in_sun = self._check_visibility(r, sun_pos)
        
        a_srp = np.zeros(3)
        if in_sun:
            a_srp = -P_SUN * CR * AREA_MASS_RATIO * u_sun                          
                                                   
                                  
                                                                                                                   
                                                                       
                                  
                                       
            a_srp = -P_SUN * CR * AREA_MASS_RATIO * u_sun

        return a_grav + a_j2_vec + a_drag + a_srp

    def _get_density(self, h_m: float) -> float:
                                  
                                               
        if h_m > 1000000: return 0.0
        
                                        
        h0 = 800000         
        rho0 = 1.170e-14                                                               
        H = 150000               
        
        return rho0 * np.exp(-(h_m - h0) / H)

    def _check_visibility(self, r_sat: np.ndarray, r_sun: np.ndarray) -> bool:
                                  
                                        
                                         
                                                                               
                                            
        
                            
                                              
                                                   
        
                                         
                                         
        
                             
                                                   
        
                                                   
                                  
                                                                      
                                                                  
        
        u_sun = r_sun / np.linalg.norm(r_sun)
        L = np.dot(r_sat, u_sun)                           
        
        if L > 0:              
            return True
            
                                
        D = np.linalg.norm(r_sat - L * u_sun)
        if D > RE:
            return True
            
        return False


class NumericalPropagator:
    def __init__(self, elements: KeplerianElements):
        self.elements = elements
        self.force_model = ForceModel()
        
                                                         
        self.t_current_jd = None
        self.state = None                        
        self._init_state()

    def _init_state(self):
                                        
                                   
                                                                                
                                                                
        
        a = self.elements.a
        e = self.elements.e
        i = self.elements.i
        om = self.elements.omega
        raan = self.elements.RAAN
        M = self.elements.M
        
                                       
                         
        E = M
        for _ in range(10):
            E = M + e * np.sin(E)
            
                               
        nu = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        r_c = a * (1 - e * np.cos(E))
        
        o = np.zeros(3)
        o[0] = r_c * np.cos(nu)
        o[1] = r_c * np.sin(nu)
        
        v_c = np.sqrt(GM * a) / r_c
                                
                      
                          
                                     
                          
        
                 
                                            
                                         
                                                     
        n = np.sqrt(GM / a**3)
        x_dot = -(n * a**2 / r_c) * np.sin(E)
        y_dot = (n * a**2 / r_c) * np.sqrt(1 - e**2) * np.cos(E)
        
        vel_perifocal = np.array([x_dot, y_dot, 0])
        pos_perifocal = np.array([a * (np.cos(E) - e), a * np.sqrt(1-e**2) * np.sin(E), 0])
        
                         
                                     
        
                 
        cw = np.cos(om)
        sw = np.sin(om)
        R_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
        
             
        ci = np.cos(i)
        si = np.sin(i)
        R_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
        
                
        cO = np.cos(raan)
        sO = np.sin(raan)
        R_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
        
        R = R_O @ R_i @ R_w
        
        r_eci = R @ pos_perifocal
        v_eci = R @ vel_perifocal
        
        self.state = np.concatenate((r_eci, v_eci))
        
                                          
                                                             
                                  
        ts = load.timescale()
        self.t_current_jd = ts.now().tt

    def step(self, dt: float):
        """
        Perform one RK4 step.
        dt: Time step in seconds.
        """
                                   
        dt_days = dt / 86400.0
        
        y = self.state
        t = self.t_current_jd
        
        k1 = self._derivs(t, y)
        k2 = self._derivs(t + dt_days/2, y + k1 * dt/2)
        k3 = self._derivs(t + dt_days/2, y + k2 * dt/2)
        k4 = self._derivs(t + dt_days, y + k3 * dt)
        
        self.state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t_current_jd += dt_days

    def _derivs(self, t_jd, y):
                                      
                                         
        r = y[:3]
        v = y[3:]
        
        a = self.force_model.compute_acceleration(t_jd, r, v)
        
        return np.concatenate((v, a))
