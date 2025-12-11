                                     

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from .constellation import StarlinkConstellation, KeplerianElements
from .orbit_propagator import NumericalPropagator
from .error_models import AtmosphericErrorModel
from .constants import C, TX_FREQUENCY

class SatelliteIdentifier:
    def __init__(self, user_lat, user_lon, user_alt=0.0):
        self.user_lat = user_lat
        self.user_lon = user_lon
        self.user_alt = user_alt
        self.user_pos_ecef = self._get_user_ecef()

        self.constellation = StarlinkConstellation()
        self.sats_elements = self.constellation.generate_all()
        
                                                                          
        self.propagators = {el.sat_id: NumericalPropagator(el) for el in self.sats_elements}
        
        self.atm_model = AtmosphericErrorModel()

    def _get_user_ecef(self):
                                                                                
        RE = 6378137.0
        lat = np.radians(self.user_lat)
        lon = np.radians(self.user_lon)
        
        x = RE * np.cos(lat) * np.cos(lon)
        y = RE * np.cos(lat) * np.sin(lon)
        z = RE * np.sin(lat)
        return np.array([x, y, z])

    def predict_all_dopplers(self, current_jd, step_sec=1.0):
        """
        Predicts Doppler shifts for all visible satellites at a given Julian Date.
        Returns a dictionary of {sat_id: predicted_doppler}.
        """
        predicted_dopplers = {}
        ts = load.timescale()
        t_skyfield = ts.tt_jd(current_jd)                       
        
                                               
                                               
        delta_t_days = 69.184 / 86400.0                     
        jd_ut1 = current_jd - delta_t_days
        Tu = jd_ut1 - 2451545.0
        theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * Tu)
        theta = theta % (2 * np.pi)
        
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        R_z = np.array([
            [c_theta, s_theta, 0],
            [-s_theta, c_theta, 0],
            [0, 0, 1]
        ])
        
        omega_e = np.array([0, 0, 7.2921159e-5])                             

        for sat_id, prop in self.propagators.items():
                                           
                                                                           
                                                                                              
                                                                                          
            
                                                                 
                                                                        
                                                                         
                                                                                                    
            
                                                                                          
                                                                                                
                                        
                                                                                    
            
                                                                         
                                                                                             
                                                                                                
                                                                                 
            
                                                                                        
                                                                                              
                                                                                            

                                                                                                   
                                                                                                
                                                                                                             

                                                                                             
                                                                                            

                                                                                 
                                                                                       
                                                                                                       

                                                                      
                                                                           
                                                                           

                                                                                                          
                                                                                        
                                                                               

                                                                 
                                                                                
            temp_prop = NumericalPropagator(self.sats_elements[sat_id])                                                                   
                                                                                                   
                                                                                                             

                                                                 
                                                                                 
                                                                                                                               

                                                                                                 
                                                                                                 
                                                                                         
                                                                     
                                                                                                              
            
                                                         
            keplerian_element = next((el for el in self.sats_elements if el.sat_id == sat_id), None)
            if keplerian_element is None:
                continue                    

                                                                        
                                                                             
            temp_prop = NumericalPropagator(keplerian_element)
            
                                                    
            dt_seconds = (current_jd - temp_prop.t_current_jd) * 86400.0
            temp_prop.step(dt_seconds)                          

            r_eci = temp_prop.state[:3]
            v_eci = temp_prop.state[3:]
            
                             
            sat_pos_ecef = R_z @ r_eci
            sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
            
            el, az, dist = self.atm_model._get_el_az_dist(sat_pos_ecef, self.user_pos_ecef)
            
            if el > 10.0:                                
                u_los = (sat_pos_ecef - self.user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                doppler = - (range_rate / C) * TX_FREQUENCY
                predicted_dopplers[sat_id] = doppler
                
        return predicted_dopplers

    def identify_satellite(self, observed_doppler, current_jd, tolerance=100.0):
        """
        Identifies the most likely satellite based on observed Doppler shift.
        `tolerance` is the maximum allowed difference in Hz.
        """
        predicted_dopplers = self.predict_all_dopplers(current_jd)
        
        if not predicted_dopplers:
            return None, float('inf')                        
            
        best_sat_id = None
        min_diff = float('inf')
        
        for sat_id, pred_doppler in predicted_dopplers.items():
            diff = abs(observed_doppler - pred_doppler)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                best_sat_id = sat_id
                
        return best_sat_id, min_diff

                                      
if __name__ == "__main__":
                                                  
    user_lat_ex = 19.0760
    user_lon_ex = 72.8777
    current_time_ex = load.utc(2023, 10, 26, 12, 0, 0)                   
    current_jd_ex = current_time_ex.tt
    
    identifier = SatelliteIdentifier(user_lat_ex, user_lon_ex)
    
    print(f"Predicting Dopplers at {current_time_ex.utc_iso()}...")
    predicted = identifier.predict_all_dopplers(current_jd_ex)
    
    if predicted:
        print("Predicted Dopplers for visible satellites:")
        for sat_id, doppler_val in predicted.items():
            print(f"  Sat ID {sat_id}: {doppler_val:.2f} Hz")
            
                                      
                                                                           
                                              
        random_sat_id = list(predicted.keys())[0]
        true_doppler = predicted[random_sat_id]
        observed_doppler_sim = true_doppler + np.random.normal(0, 50)                 
        
        print(f"\nSimulated Observed Doppler: {observed_doppler_sim:.2f} Hz (True from Sat ID {random_sat_id})")
        
        identified_sat_id, diff = identifier.identify_satellite(observed_doppler_sim, current_jd_ex)
        
        if identified_sat_id is not None:
            print(f"Identified Satellite ID: {identified_sat_id} with difference {diff:.2f} Hz")
            print(f"Actual Sat ID was: {random_sat_id}")
        else:
            print("No satellite identified within tolerance.")
    else:
        print("No satellites visible at this time.")

                                              
    current_time_no_vis = load.utc(2023, 1, 1, 0, 0, 0)
    current_jd_no_vis = current_time_no_vis.tt
    print(f"\nPredicting Dopplers at {current_time_no_vis.utc_iso()} (expecting none visible)...")
    predicted_no_vis = identifier.predict_all_dopplers(current_jd_no_vis)
    print(f"Predicted Dopplers: {predicted_no_vis}")
    
    identified_sat_id_no_vis, diff_no_vis = identifier.identify_satellite(100.0, current_jd_no_vis)
    print(f"Identified Satellite ID: {identified_sat_id_no_vis}")
