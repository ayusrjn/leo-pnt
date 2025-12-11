import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from .constellation import LEOS9Constellation, StarlinkConstellation
from .orbit_propagator import NumericalPropagator
from .error_models import ClockErrorModel, ClockType, AtmosphericErrorModel, MeasurementNoiseModel
from .constants import C, TX_FREQUENCY

class LEOS9Simulator:
    def __init__(self, duration_sec=60, step_sec=1.0):
        self.duration = duration_sec
        self.step = step_sec
        
                                  
        print("Initializing Constellation (Starlink Shell 1)...")
        self.constellation = StarlinkConstellation()
        self.sats_elements = self.constellation.generate_all()
        print(f"Generated {len(self.sats_elements)} satellites.")
        
                                                    
                                                 
                                                
                                                                 
                                                                                            
        self.propagators = [NumericalPropagator(el) for el in self.sats_elements]
        
                                 
        self.clocks = [ClockErrorModel(ClockType.OCXO) for _ in range(len(self.sats_elements))]
        self.atm_model = AtmosphericErrorModel()
        self.noise_model = MeasurementNoiseModel()
        
                                        
        self.user_lat = 19.0760
        self.user_lon = 72.8777
        self.user_alt = 0.0
        self.user_pos_ecef = self._get_user_ecef()

    def _get_user_ecef(self):
                           
                                   
                                   
                                  
                                  
        RE = 6378137.0
        lat = np.radians(self.user_lat)
        lon = np.radians(self.user_lon)
        
        x = RE * np.cos(lat) * np.cos(lon)
        y = RE * np.cos(lat) * np.sin(lon)
        z = RE * np.sin(lat)
        return np.array([x, y, z])

    def run(self):
        print(f"Starting Simulation for {self.duration} seconds...")
        results = []
        
                   
        for t_sec in range(0, self.duration, int(self.step)):
            if t_sec % 10 == 0:
                print(f"Time: {t_sec}/{self.duration}")
                
                                      
            for i, prop in enumerate(self.propagators):
                prop.step(self.step)
                
                               
                r_eci = prop.state[:3]
                v_eci = prop.state[3:]
                
                                 
                                                                                      
                                       
                ts = load.timescale()
                t_skyfield = ts.tt_jd(prop.t_current_jd)
                
                                              
                                    
                                               

                                                                                     
                                                                            
                                               
                                                             
                                                                                                 
                           
                                                                 
                                                   
                
                                                            
                                                                                                
                                                    
                                                                             
                
                                                     
                                                              
                
                                                       
                                                                                                    
                                                                    
                                                 
                
                                                                                                 
                                                                                                                  
                                                
                
                                                            
                delta_t_days = 69.184 / 86400.0
                jd_ut1 = prop.t_current_jd - delta_t_days
                Tu = jd_ut1 - 2451545.0
                theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * Tu)
                theta = theta % (2 * np.pi)
                
                c = np.cos(theta)
                s = np.sin(theta)
                R_z = np.array([
                    [c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]
                ])
                
                sat_pos_ecef = R_z @ r_eci
                
                                                         
                                                           
                omega_e = np.array([0, 0, 7.2921159e-5])
                sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
                
                                       
                sat_pos = sat_pos_ecef
                sat_vel = sat_vel_ecef
                
                el, az, dist = self.atm_model._get_el_az_dist(sat_pos, self.user_pos_ecef)
                
                if el > 10.0:                 
                                           
                    
                                        
                    geom_range = dist
                    
                                    
                                                                   
                    clk_err = 0.0
                    
                                           
                    iono = self.atm_model.get_ionosphere_delay(sat_pos, self.user_pos_ecef, t_sec)
                    tropo = self.atm_model.get_troposphere_delay(sat_pos, self.user_pos_ecef)
                    
                                                        
                                                           
                    sagnac = omega_e[2] * (sat_pos[0]*self.user_pos_ecef[1] - sat_pos[1]*self.user_pos_ecef[0]) / C
                    
                                      
                    shapiro = 0.0
                    
                                
                                
                                            
                    u_los = (sat_pos - self.user_pos_ecef) / dist
                    range_rate = np.dot(sat_vel, u_los)
                    
                                                                      
                    doppler = - (range_rate / C) * TX_FREQUENCY
                    
                                       
                    pr_true = geom_range + clk_err + iono + tropo + sagnac + shapiro
                    
                               
                    pr_meas, ph_meas = self.noise_model.add_noise(pr_true)
                    
                    results.append({
                        "time": t_sec,
                        "sat_id": self.sats_elements[i].sat_id,
                        "az": az,
                        "el": el,
                        "pseudorange": pr_meas,
                        "carrier_phase": ph_meas,                            
                        "doppler": doppler,
                        "true_range": geom_range,
                        "sat_x": sat_pos[0],
                        "sat_y": sat_pos[1],
                        "sat_z": sat_pos[2],
                        "sat_vx": sat_vel[0],
                        "sat_vy": sat_vel[1],
                        "sat_vz": sat_vel[2]
                    })
                    
        return pd.DataFrame(results)

if __name__ == "__main__":
    sim = LEOS9Simulator(duration_sec=60)
    df = sim.run()
    print(df.head())
    df.to_csv("simulation_results.csv", index=False)
