import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skyfield.api import load, Topos, wgs84
from doppler_pkg.constellation import StarlinkConstellation, KeplerianElements
from doppler_pkg.orbit_propagator import NumericalPropagator
from doppler_pkg.error_models import AtmosphericErrorModel
from skyfield.timelib import Time

                   
C = 299792458.0                         
TX_FREQUENCY = 137.1e6                                    





class SignalSimulator:
    """
    Module C: SignalSimulator (The Digital Twin)
    Responsibility: Generate synthetic 'Measured' data.
    """
    def __init__(self, propagator: NumericalPropagator, tx_freq: float = TX_FREQUENCY, user_lat=0.0, user_lon=0.0, user_alt=0.0):
        self.propagator = propagator
        self.tx_freq = tx_freq
        self.user_lat = user_lat
        self.user_lon = user_lon
        self.user_alt = user_alt
        self.user_pos_ecef = self._get_user_ecef()
        self.atm_model = AtmosphericErrorModel()

    def _get_user_ecef(self):
        RE = 6378137.0
        lat = np.radians(self.user_lat)
        lon = np.radians(self.user_lon)
        x = RE * np.cos(lat) * np.cos(lon)
        y = RE * np.cos(lat) * np.sin(lon)
        z = RE * np.sin(lat)
        return np.array([x, y, z])

    def generate_data(self, start_time: datetime.datetime, duration_sec: int, noise_std: float):
        """
        Iterate second-by-second over the pass duration.
        Calculate exact theoretical frequency.
        Add Gaussian noise.
        """
        ts = load.timescale()
        timestamps = []
        noisy_freqs = []
        
        print(f"Simulating signal for Lat: {self.user_lat}, Lon: {self.user_lon}...")
        

        
                                                              
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

        for i in range(duration_sec):
                                              
            self.propagator.step(1.0)                   

                           
            r_eci = self.propagator.state[:3]
            v_eci = self.propagator.state[3:]

                                                         
            jd_ut1 = self.propagator.t_current_jd - delta_t_days
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
            sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
            
            el, az, dist = self.atm_model._get_el_az_dist(sat_pos_ecef, self.user_pos_ecef)
            
            if el > 10.0:                  
                                      
                u_los = (sat_pos_ecef - self.user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                
                               
                f_theoretical = self.tx_freq * (1 - range_rate / C)
                
                           
                noise = np.random.normal(0, noise_std)
                f_measured = f_theoretical + noise
                
                timestamps.append(self.propagator.t_current_jd)           
                noisy_freqs.append(f_measured)
            else:
                                                    
                pass
            
                                                                                             
                                                                                                          
                                                                 
        dt_timestamps = [ts.tt_jd(jd).utc_datetime() for jd in timestamps]
        return dt_timestamps, noisy_freqs

class GeolocationSolver:
    """
    Module D: GeolocationSolver (The Algorithm)
    Responsibility: Blindly estimate location using only TLE and noisy frequency data.
    """
    def __init__(self, propagator: NumericalPropagator, tx_freq: float = TX_FREQUENCY):
        self.propagator = propagator
        self.tx_freq = tx_freq
        self.ts = load.timescale()
        self.atm_model = AtmosphericErrorModel()

    def solve(self, timestamps: list, measured_freqs: list) -> tuple:
        """
        Estimate Lat/Lon using least squares optimization.
        """
        print("Solving for location...")
        
        measured_freqs_np = np.array(measured_freqs)
        
                                                                                  
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

        def _residuals(guess_lat_lon):
            lat, lon = guess_lat_lon
            
                                                       
            RE = 6378137.0
            user_lat_rad = np.radians(lat)
            user_lon_rad = np.radians(lon)
            user_x = RE * np.cos(user_lat_rad) * np.cos(user_lon_rad)
            user_y = RE * np.cos(user_lat_rad) * np.sin(user_lon_rad)
            user_z = RE * np.sin(user_lat_rad)
            user_pos_ecef = np.array([user_x, user_y, user_z])

            theoretical_freqs = []
            
                                                                                                             
                                                                                         
            temp_propagator = NumericalPropagator(self.propagator.keplerian_elements)
            
                                                                  
            first_jd = timestamps[0]                         
            dt_to_first = (first_jd - temp_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                temp_propagator.step(dt_to_first)

            for t_jd in timestamps:
                                                                                                       
                                                                       
                dt_step = (t_jd - temp_propagator.t_current_jd) * 86400.0
                if dt_step > 0:
                    temp_propagator.step(dt_step)
                
                r_eci = temp_propagator.state[:3]
                v_eci = temp_propagator.state[3:]

                jd_ut1 = temp_propagator.t_current_jd - delta_t_days
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
                sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
                
                el, az, dist = self.atm_model._get_el_az_dist(sat_pos_ecef, user_pos_ecef)
                
                if el > 10.0:                  
                    u_los = (sat_pos_ecef - user_pos_ecef) / dist
                    range_rate = np.dot(sat_vel_ecef, u_los)
                    f_theo = self.tx_freq * (1 - range_rate / C)
                    theoretical_freqs.append(f_theo)
                else:
                                                                                           
                                                                                              
                                                                                               
                                                                                        
                                                                                                          
                    theoretical_freqs.append(0.0)                                                          
                                                                                                        
                                                                                                         
                                                                                              
                                                                              
            
            return measured_freqs_np - np.array(theoretical_freqs)

                               
        initial_guess = [0.0, 0.0]
        
                                                
        bounds = ([-90, -180], [90, 180])

        result = least_squares(_residuals, initial_guess, bounds=bounds, verbose=1)
        
        return result.x[0], result.x[1]

class Visualizer:
    """
    Module E: Visualizer (Analysis)
    Responsibility: Plot the results.
    """
    @staticmethod
    def plot_results(timestamps_dt, measured_freqs, solved_lat, solved_lon, true_lat, true_lon, propagator: NumericalPropagator, tx_freq):
        ts = load.timescale()
        atm_model = AtmosphericErrorModel()                                    
        
                                     
        plt.figure(figsize=(10, 6))
        plt.scatter(timestamps_dt, measured_freqs, s=10, label='Noisy Measured Data', color='blue', alpha=0.5)
        
                                
        fitted_freqs = []
        
                                                                                                  
        first_jd = ts.from_datetime(timestamps_dt[0]).tt
        dt_to_first = (first_jd - propagator.t_current_jd) * 86400.0
        if dt_to_first > 0:
            propagator.step(dt_to_first)                                           
            
                                                                                  
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

                                           
        RE = 6378137.0
        user_lat_rad = np.radians(solved_lat)
        user_lon_rad = np.radians(solved_lon)
        user_x = RE * np.cos(user_lat_rad) * np.cos(user_lon_rad)
        user_y = RE * np.cos(user_lat_rad) * np.sin(user_lon_rad)
        user_z = RE * np.sin(user_lat_rad)
        user_pos_ecef = np.array([user_x, user_y, user_z])

        for i in range(len(timestamps_dt)):
            propagator.step(1.0)                                       
            
            r_eci = propagator.state[:3]
            v_eci = propagator.state[3:]

            jd_ut1 = propagator.t_current_jd - delta_t_days
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
            
            sat_pos_ecef = R_z @ r_eci
            sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
            
            el, az, dist = atm_model._get_el_az_dist(sat_pos_ecef, user_pos_ecef)
            
            if el > 10.0:                  
                u_los = (sat_pos_ecef - user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                f_theo = tx_freq * (1 - range_rate / C)
                fitted_freqs.append(f_theo)
            else:
                                                                                         
                fitted_freqs.append(np.nan)                                   

                                 
        plt.figure(figsize=(8, 6))
        plt.scatter(true_lon, true_lat, color='green', marker='*', s=200, label='True Location')
        plt.scatter(solved_lon, solved_lat, color='red', marker='x', s=100, label='Estimated Location')
        
                   
        plt.plot([true_lon, solved_lon], [true_lat, solved_lat], 'k--', alpha=0.5)
        
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geolocation Result')
        plt.legend()
        plt.grid(True)
        plt.savefig('geolocation_map.png')
        print("Saved geolocation_map.png")
        plt.show()

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in km."""
    R = 6371.0                      
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def main():
                                                    
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
    print(f"Hidden Truth Location: Lat {TRUE_LAT}, Lon {TRUE_LON}")

                                          
    print("Initializing Starlink Constellation (Shell 1)...")
    constellation = StarlinkConstellation()
    sats_elements = constellation.generate_all()
    print(f"Generated {len(sats_elements)} satellites.")

                                                                     
                                                               
                                                
    np.random.seed(42)                      
    true_sat_keplerian_element = sats_elements[np.random.randint(len(sats_elements))]
    true_sat_propagator = NumericalPropagator(true_sat_keplerian_element)
    print(f"\nSimulating signal from True Sat ID: {true_sat_keplerian_element.sat_id}")

                                                                     
    ts = load.timescale()
    
                                                                                    
    t_search_start = ts.now()
    t_search_end = ts.from_datetime(t_search_start.utc_datetime() + datetime.timedelta(days=2))                                   
    
    print(f"Searching for next pass for Sat ID {true_sat_keplerian_element.sat_id} over Lat: {TRUE_LAT}, Lon: {TRUE_LON}...")
    observer_topos = wgs84.latlon(TRUE_LAT, TRUE_LON)
    
                                                                   
                                                                             
                                                                    
                                                                                                
                                                                               
                                                                                             
                                                                                 
    
                                                                                        
                                                                                             
                                                                                    
                                                                                   
                                                                                    
                                               
    
                                                                            
                                                       
    
    start_time_found = None
    search_duration_hours = 48
    time_step_minutes = 5
    
    print(f"Searching for visible pass for Sat ID {true_sat_keplerian_element.sat_id}...")
    
                                                                                                 
    temp_search_propagator = NumericalPropagator(true_sat_keplerian_element)
    
                                                      
    current_jd = ts.now().tt
    dt_to_current = (current_jd - temp_search_propagator.t_current_jd) * 86400.0
    if dt_to_current > 0:
        temp_search_propagator.step(dt_to_current)

    user_lat_rad = np.radians(TRUE_LAT)
    user_lon_rad = np.radians(TRUE_LON)
    RE = 6378137.0
    user_pos_ecef = np.array([
        RE * np.cos(user_lat_rad) * np.cos(user_lon_rad),
        RE * np.cos(user_lat_rad) * np.sin(user_lon_rad),
        RE * np.sin(user_lat_rad)
    ])
    atm_model = AtmosphericErrorModel()
    
                                                          
    delta_t_days = 69.184 / 86400.0
    omega_e = np.array([0, 0, 7.2921159e-5])

    for i in range(0, search_duration_hours * 60, time_step_minutes):
        temp_search_propagator.step(time_step_minutes * 60.0)                            
        
        r_eci = temp_search_propagator.state[:3]
        v_eci = temp_search_propagator.state[3:]

        jd_ut1 = temp_search_propagator.t_current_jd - delta_t_days
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
        sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
        
        el, az, dist = atm_model._get_el_az_dist(sat_pos_ecef, user_pos_ecef)
        
        if el > 10.0:
            start_time_found = ts.tt_jd(temp_search_propagator.t_current_jd).utc_datetime()
            print(f"Visible pass found starting at (approx) {start_time_found} with elevation {el:.2f} deg.")
            break
            
    if start_time_found is None:
        print("No visible pass found in the next 48 hours for the selected satellite. Exiting.")
        return
    
    start_time_for_sim = start_time_found - datetime.timedelta(minutes=5)                          
    
    DURATION = 1800             
    NOISE_STD = 50.0     

                                                                        
    true_sat_propagator = NumericalPropagator(true_sat_keplerian_element)                   
    sim_start_jd = ts.from_datetime(start_time_for_sim).tt
    dt_to_sim_start = (sim_start_jd - true_sat_propagator.t_current_jd) * 86400.0
    if dt_to_sim_start > 0:
        true_sat_propagator.step(dt_to_sim_start)

                                                                                       
    simulator = SignalSimulator(true_sat_propagator, TX_FREQUENCY, TRUE_LAT, TRUE_LON)
    timestamps, measured_freqs = simulator.generate_data(start_time_for_sim, DURATION, NOISE_STD)

    if not measured_freqs:
        print("No visible passes for the simulated satellite during the DURATION. This should not happen if pass finding was successful. Exiting.")
        return

                                                           
    first_measurement_jd = ts.from_datetime(timestamps[0]).tt

                                                                         
    from doppler_pkg.satellite_identifier import SatelliteIdentifier
    identifier = SatelliteIdentifier(TRUE_LAT, TRUE_LON)
    
    print("\nAttempting to identify satellite from observed Doppler...")
                                                       
    observed_doppler_for_id = measured_freqs[0] - TX_FREQUENCY                                         
    identified_sat_id, diff = identifier.identify_satellite(observed_doppler_for_id, first_measurement_jd, tolerance=1000.0)                                       

    if identified_sat_id is not None:
        print(f"Identified Satellite ID: {identified_sat_id} with difference {diff:.2f} Hz")
                                                                 
        identified_keplerian_element = next((el for el in sats_elements if el.sat_id == identified_sat_id), None)
        if identified_keplerian_element:
                                                                                                        
            identified_sat_propagator = NumericalPropagator(identified_keplerian_element)
                                                              
            dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                identified_sat_propagator.step(dt_to_first)
            
            print(f"Using Identified Satellite ID {identified_sat_id} for geolocation.")
        else:
            print(f"Error: Keplerian elements not found for identified satellite ID {identified_sat_id}. Using a default (first) satellite.")
            identified_sat_propagator = NumericalPropagator(sats_elements[0])           
            dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                identified_sat_propagator.step(dt_to_first)

    else:
        print("No satellite identified within tolerance. Using a default (first) satellite for geolocation.")
        identified_sat_propagator = NumericalPropagator(sats_elements[0])           
        dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
        if dt_to_first > 0:
            identified_sat_propagator.step(dt_to_first)


                                                                          
                                                                                                  
    jd_timestamps = [ts.from_datetime(dt).tt for dt in timestamps]
    solver = GeolocationSolver(identified_sat_propagator, TX_FREQUENCY)
    est_lat, est_lon = solver.solve(jd_timestamps, measured_freqs)

                 
    error_km = haversine_distance(TRUE_LAT, TRUE_LON, est_lat, est_lon)
    print(f"\n--- Results ---")
    print(f"True Location:      {TRUE_LAT:.4f}, {TRUE_LON:.4f}")
    print(f"Estimated Location: {est_lat:.4f}, {est_lon:.4f}")
    print(f"Error: {error_km:.2f} km")
    if identified_sat_id is not None:
        print(f"Truth Sat ID: {true_sat_keplerian_element.sat_id}")
        print(f"Identified Sat ID: {identified_sat_id}")
        
                  
                                                        
    Visualizer.plot_results(timestamps, measured_freqs, est_lat, est_lon, TRUE_LAT, TRUE_LON, identified_sat_propagator, TX_FREQUENCY)

if __name__ == "__main__":
    main()
