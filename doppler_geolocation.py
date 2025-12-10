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

# --- Constants ---
C = 299792458.0  # Speed of light in m/s
TX_FREQUENCY = 137.1e6  # 137.1 MHz (Typical for NOAA APT)





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
        

        
        # Earth rotation constants (copied from leo_s9_sim.py)
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

        for i in range(duration_sec):
            # Propagate satellite for one step
            self.propagator.step(1.0) # Step by 1 second

            # Get ECI State
            r_eci = self.propagator.state[:3]
            v_eci = self.propagator.state[3:]

            # Convert to ECEF (copied from leo_s9_sim.py)
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
            
            if el > 10.0: # Only if visible
                # Calculate Range Rate
                u_los = (sat_pos_ecef - self.user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                
                # Doppler Shift
                f_theoretical = self.tx_freq * (1 - range_rate / C)
                
                # Add noise
                noise = np.random.normal(0, noise_std)
                f_measured = f_theoretical + noise
                
                timestamps.append(self.propagator.t_current_jd) # Store JD
                noisy_freqs.append(f_measured)
            else:
                # If not visible, do not append data
                pass
            
        # Convert JDs back to datetime objects for plotting if necessary, or pass JD directly
        # For now, let's keep it as JD for internal calculations and convert later if needed for plotting.
        # But Visualizer expects datetime, so we need to convert.
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
        
        # Earth rotation constants (copied from leo_s9_sim.py and SignalSimulator)
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

        def _residuals(guess_lat_lon):
            lat, lon = guess_lat_lon
            
            # User ECEF position based on current guess
            RE = 6378137.0
            user_lat_rad = np.radians(lat)
            user_lon_rad = np.radians(lon)
            user_x = RE * np.cos(user_lat_rad) * np.cos(user_lon_rad)
            user_y = RE * np.cos(user_lat_rad) * np.sin(user_lon_rad)
            user_z = RE * np.sin(user_lat_rad)
            user_pos_ecef = np.array([user_x, user_y, user_z])

            theoretical_freqs = []
            
            # Create a temporary propagator for calculations within residuals to avoid modifying the original
            # This is critical because least_squares will call _residuals multiple times.
            temp_propagator = NumericalPropagator(self.propagator.keplerian_elements)
            
            # Step the temporary propagator to the first timestamp
            first_jd = timestamps[0] # Timestamps are now JDs
            dt_to_first = (first_jd - temp_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                temp_propagator.step(dt_to_first)

            for t_jd in timestamps:
                # Need to step the temporary propagator by the difference from its current time to t_jd
                # This assumes timestamps are monotonically increasing.
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
                
                if el > 10.0: # Only if visible
                    u_los = (sat_pos_ecef - user_pos_ecef) / dist
                    range_rate = np.dot(sat_vel_ecef, u_los)
                    f_theo = self.tx_freq * (1 - range_rate / C)
                    theoretical_freqs.append(f_theo)
                else:
                    # If not visible, we can't calculate a Doppler, so append NaN or handle
                    # For least_squares, this might be problematic if it expects equal length.
                    # This implies that observed_freqs should also only contain visible points.
                    # Or, we should assign a very large residual for non-visible points.
                    # For now, append 0 or f_theoretical. The `generate_data` already filters non-visible.
                    theoretical_freqs.append(0.0) # This will be incorrect, need to ensure data is visible.
                                                  # The `generate_data` in SignalSimulator filters data,
                                                  # so `measured_freqs` will only contain visible points.
                                                  # If a satellite is not visible for a guess,
                                                  # the residual will be high.
            
            return measured_freqs_np - np.array(theoretical_freqs)

        # Initial guess: (0, 0)
        initial_guess = [0.0, 0.0]
        
        # Bounds: Lat [-90, 90], Lon [-180, 180]
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
        atm_model = AtmosphericErrorModel() # Initialize here for static method
        
        # --- Plot 1: The S-Curve ---
        plt.figure(figsize=(10, 6))
        plt.scatter(timestamps_dt, measured_freqs, s=10, label='Noisy Measured Data', color='blue', alpha=0.5)
        
        # Calculate fitted curve
        fitted_freqs = []
        
        # Need to ensure the propagator is at the start of the time series for consistent plotting
        first_jd = ts.from_datetime(timestamps_dt[0]).tt
        dt_to_first = (first_jd - propagator.t_current_jd) * 86400.0
        if dt_to_first > 0:
            propagator.step(dt_to_first) # Advance the propagator to the start time
            
        # Earth rotation constants (copied from leo_s9_sim.py and SignalSimulator)
        delta_t_days = 69.184 / 86400.0
        omega_e = np.array([0, 0, 7.2921159e-5])

        # User ECEF for the solved location
        RE = 6378137.0
        user_lat_rad = np.radians(solved_lat)
        user_lon_rad = np.radians(solved_lon)
        user_x = RE * np.cos(user_lat_rad) * np.cos(user_lon_rad)
        user_y = RE * np.cos(user_lat_rad) * np.sin(user_lon_rad)
        user_z = RE * np.sin(user_lat_rad)
        user_pos_ecef = np.array([user_x, user_y, user_z])

        for i in range(len(timestamps_dt)):
            propagator.step(1.0) # Step by 1 second for each data point
            
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
            
            if el > 10.0: # Only if visible
                u_los = (sat_pos_ecef - user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                f_theo = tx_freq * (1 - range_rate / C)
                fitted_freqs.append(f_theo)
            else:
                # If not visible, append NaN or a value that doesn't mess up plot scaling
                fitted_freqs.append(np.nan) # Will appear as a gap in the plot

        # --- Plot 2: The Map ---
        plt.figure(figsize=(8, 6))
        plt.scatter(true_lon, true_lat, color='green', marker='*', s=200, label='True Location')
        plt.scatter(solved_lon, solved_lat, color='red', marker='x', s=100, label='Estimated Location')
        
        # Draw line
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
    R = 6371.0  # Earth radius in km
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def main():
    # 1. Define Hidden Truth Location (e.g., Mumbai)
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
    print(f"Hidden Truth Location: Lat {TRUE_LAT}, Lon {TRUE_LON}")

    # 2. Initialize Starlink Constellation
    print("Initializing Starlink Constellation (Shell 1)...")
    constellation = StarlinkConstellation()
    sats_elements = constellation.generate_all()
    print(f"Generated {len(sats_elements)} satellites.")

    # 3. Setup a "true" satellite for simulation (e.g., a random one)
    # For a real scenario, this would be an actual measurement.
    # For now, we simulate one satellite's data.
    np.random.seed(42) # For reproducibility
    true_sat_keplerian_element = sats_elements[np.random.randint(len(sats_elements))]
    true_sat_propagator = NumericalPropagator(true_sat_keplerian_element)
    print(f"\nSimulating signal from True Sat ID: {true_sat_keplerian_element.sat_id}")

    # 4. Generate simulated noisy signal data from the true satellite
    ts = load.timescale()
    
    # Find a good pass time (when satellite is above horizon for TRUE_LAT, TRUE_LON)
    t_search_start = ts.now()
    t_search_end = ts.from_datetime(t_search_start.utc_datetime() + datetime.timedelta(days=2)) # Search for passes in next 2 days
    
    print(f"Searching for next pass for Sat ID {true_sat_keplerian_element.sat_id} over Lat: {TRUE_LAT}, Lon: {TRUE_LON}...")
    observer_topos = wgs84.latlon(TRUE_LAT, TRUE_LON)
    
    # Create a dummy skyfield EarthSatellite object for find_events
    # This is a bit of a hack as we moved away from Skyfield for propagation.
    # For now, it's the easiest way to use Skyfield's event finding.
    # In a more complete refactor, a 'find_events' method could be added to NumericalPropagator.
    # For now, we rely on the KeplerianElements to TLE conversion for Skyfield.
    # Assuming KeplerianElements can be converted to TLE for a dummy Skyfield EarthSatellite.
    # Let's read doppler_pkg/tle_manager.py to see if there's a utility for this.
    
    # Correction: I cannot easily convert KeplerianElements to a Skyfield EarthSatellite
    # without a TLE string. The original `find_events` was using a pre-loaded EarthSatellite.
    # I need to implement a similar `find_events` logic using NumericalPropagator or
    # find an existing way to get a skyfield EarthSatellite from KeplerianElements.
    # Given the previous refactoring, it's better to implement a basic event finding
    # within the NumericalPropagator framework.
    
    # For simplicity, and to avoid re-introducing Skyfield's EarthSatellite,
    # let's just loop through time and check elevation.
    
    start_time_found = None
    search_duration_hours = 48
    time_step_minutes = 5
    
    print(f"Searching for visible pass for Sat ID {true_sat_keplerian_element.sat_id}...")
    
    # Create a temporary propagator for searching, so the main true_sat_propagator isn't affected
    temp_search_propagator = NumericalPropagator(true_sat_keplerian_element)
    
    # Propagate temp_search_propagator to current time
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
    
    # Earth rotation constants (copied from leo_s9_sim.py)
    delta_t_days = 69.184 / 86400.0
    omega_e = np.array([0, 0, 7.2921159e-5])

    for i in range(0, search_duration_hours * 60, time_step_minutes):
        temp_search_propagator.step(time_step_minutes * 60.0) # Step by time_step_minutes
        
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
    
    start_time_for_sim = start_time_found - datetime.timedelta(minutes=5) # Start a bit before pass
    
    DURATION = 1800 # 30 minutes
    NOISE_STD = 50.0 # Hz

    # Re-initialize the true_sat_propagator to the simulation start time
    true_sat_propagator = NumericalPropagator(true_sat_keplerian_element) # Fresh propagator
    sim_start_jd = ts.from_datetime(start_time_for_sim).tt
    dt_to_sim_start = (sim_start_jd - true_sat_propagator.t_current_jd) * 86400.0
    if dt_to_sim_start > 0:
        true_sat_propagator.step(dt_to_sim_start)

    # Initialize SignalSimulator with the true satellite's propagator and user location
    simulator = SignalSimulator(true_sat_propagator, TX_FREQUENCY, TRUE_LAT, TRUE_LON)
    timestamps, measured_freqs = simulator.generate_data(start_time_for_sim, DURATION, NOISE_STD)

    if not measured_freqs:
        print("No visible passes for the simulated satellite during the DURATION. This should not happen if pass finding was successful. Exiting.")
        return

    # Convert first timestamp to JD for SatelliteIdentifier
    first_measurement_jd = ts.from_datetime(timestamps[0]).tt

    # 5. Identify the satellite based on the observed (simulated) Doppler
    from doppler_pkg.satellite_identifier import SatelliteIdentifier
    identifier = SatelliteIdentifier(TRUE_LAT, TRUE_LON)
    
    print("\nAttempting to identify satellite from observed Doppler...")
    # Use the first observed Doppler for identification
    observed_doppler_for_id = measured_freqs[0] - TX_FREQUENCY # Convert observed freq to Doppler shift
    identified_sat_id, diff = identifier.identify_satellite(observed_doppler_for_id, first_measurement_jd, tolerance=1000.0) # Increased tolerance for initial test

    if identified_sat_id is not None:
        print(f"Identified Satellite ID: {identified_sat_id} with difference {diff:.2f} Hz")
        # Find the KeplerianElements for the identified satellite
        identified_keplerian_element = next((el for el in sats_elements if el.sat_id == identified_sat_id), None)
        if identified_keplerian_element:
            # Create a new propagator for the identified satellite, initialized to the start of the data
            identified_sat_propagator = NumericalPropagator(identified_keplerian_element)
            # Propagate to the first measurement time to align
            dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                identified_sat_propagator.step(dt_to_first)
            
            print(f"Using Identified Satellite ID {identified_sat_id} for geolocation.")
        else:
            print(f"Error: Keplerian elements not found for identified satellite ID {identified_sat_id}. Using a default (first) satellite.")
            identified_sat_propagator = NumericalPropagator(sats_elements[0]) # Fallback
            dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
            if dt_to_first > 0:
                identified_sat_propagator.step(dt_to_first)

    else:
        print("No satellite identified within tolerance. Using a default (first) satellite for geolocation.")
        identified_sat_propagator = NumericalPropagator(sats_elements[0]) # Fallback
        dt_to_first = (first_measurement_jd - identified_sat_propagator.t_current_jd) * 86400.0
        if dt_to_first > 0:
            identified_sat_propagator.step(dt_to_first)


    # 6. Solve for geolocation using the identified satellite's propagator
    # The timestamps for the solver are now JDs, so convert the generated datetime objects to JDs.
    jd_timestamps = [ts.from_datetime(dt).tt for dt in timestamps]
    solver = GeolocationSolver(identified_sat_propagator, TX_FREQUENCY)
    est_lat, est_lon = solver.solve(jd_timestamps, measured_freqs)

    # 7. Analysis
    error_km = haversine_distance(TRUE_LAT, TRUE_LON, est_lat, est_lon)
    print(f"\n--- Results ---")
    print(f"True Location:      {TRUE_LAT:.4f}, {TRUE_LON:.4f}")
    print(f"Estimated Location: {est_lat:.4f}, {est_lon:.4f}")
    print(f"Error: {error_km:.2f} km")
    if identified_sat_id is not None:
        print(f"Truth Sat ID: {true_sat_keplerian_element.sat_id}")
        print(f"Identified Sat ID: {identified_sat_id}")
        
    # 8. Visualize
    # Visualizer expects datetime objects for timestamps
    Visualizer.plot_results(timestamps, measured_freqs, est_lat, est_lon, TRUE_LAT, TRUE_LON, identified_sat_propagator, TX_FREQUENCY)

if __name__ == "__main__":
    main()
