import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.optimize import least_squares
from skyfield.api import load, Topos, EarthSatellite, wgs84
from skyfield.timelib import Time

# --- Constants ---
C = 299792458.0  # Speed of light in m/s
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle"
TLE_FILENAME = "weather.txt" 
SATELLITE_NAME = "NOAA 20 (JPSS-1)"
TX_FREQUENCY = 137.1e6  # 137.1 MHz (Typical for NOAA APT)

class TLEManager:
    """
    Module A: TLEManager (Data Handling)
    Responsibility: Fetch and manage orbital data.
    """
    def __init__(self, filename: str = TLE_FILENAME, url: str = TLE_URL):
        self.filename = filename
        self.url = url

    def _is_file_fresh(self) -> bool:
        """Check if local TLE file exists and is < 24 hours old."""
        if not os.path.exists(self.filename):
            return False
        file_age = time.time() - os.path.getmtime(self.filename)
        return file_age < 86400  # 24 hours in seconds

    def _download_tle(self):
        """Download fresh TLE data from CelesTrak."""
        print(f"Downloading fresh TLE data from {self.url}...")
        response = requests.get(self.url)
        response.raise_for_status()
        with open(self.filename, 'w') as f:
            f.write(response.text)
        print("Download complete.")

    def get_satellite(self, sat_name: str) -> EarthSatellite:
        """
        Parse the file to find a specific satellite object by name.
        Returns a Skyfield Satellite object.
        """
        if not self._is_file_fresh():
            self._download_tle()

        ts = load.timescale()
        satellites = load.tle_file(self.filename)
        by_name = {sat.name: sat for sat in satellites}
        
        if sat_name not in by_name:
            raise ValueError(f"Satellite '{sat_name}' not found in TLE file.")
            
        return by_name[sat_name]

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
        
        # We need relative position and velocity vectors in the observer's frame (or ECEF)
        # Ideally for Doppler, we want the range rate.
        # Let's get the position and velocity in GCRS first, then relative.
        # Actually, Skyfield's position_and_velocity() on the difference vector gives GCRS difference.
        # But we want the range rate (scalar).
        
        # Let's stick to the prompt's request: "returns the satellite's position relative to a specific ground location"
        # We will return the position and velocity vectors in the GCRS frame for the *difference* vector.
        # This allows calculating the dot product for range rate.
        
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

class SignalSimulator:
    """
    Module C: SignalSimulator (The Digital Twin)
    Responsibility: Generate synthetic 'Measured' data.
    """
    def __init__(self, propagator: SatellitePropagator, tx_freq: float = TX_FREQUENCY):
        self.propagator = propagator
        self.tx_freq = tx_freq

    def generate_data(self, true_lat: float, true_lon: float, start_time: datetime.datetime, duration_sec: int, noise_std: float):
        """
        Iterate second-by-second over the pass duration.
        Calculate exact theoretical frequency.
        Add Gaussian noise.
        """
        ts = load.timescale()
        timestamps = []
        noisy_freqs = []
        
        print(f"Simulating signal for Lat: {true_lat}, Lon: {true_lon}...")
        
        for i in range(duration_sec):
            curr_time_dt = start_time + datetime.timedelta(seconds=i)
            t = ts.from_datetime(curr_time_dt)
            
            # Calculate theoretical frequency
            # f_obs = f_tx * (1 - v_rel / c)
            v_rel = self.propagator.calculate_range_rate(t, true_lat, true_lon)
            f_theoretical = self.tx_freq * (1 - v_rel / C)
            
            # Add noise
            noise = np.random.normal(0, noise_std)
            f_measured = f_theoretical + noise
            
            timestamps.append(curr_time_dt)
            noisy_freqs.append(f_measured)
            
        return timestamps, noisy_freqs

class GeolocationSolver:
    """
    Module D: GeolocationSolver (The Algorithm)
    Responsibility: Blindly estimate location using only TLE and noisy frequency data.
    """
    def __init__(self, propagator: SatellitePropagator, tx_freq: float = TX_FREQUENCY):
        self.propagator = propagator
        self.tx_freq = tx_freq
        self.ts = load.timescale()

    def solve(self, timestamps: list, measured_freqs: list) -> tuple:
        """
        Estimate Lat/Lon using least squares optimization.
        """
        print("Solving for location...")
        
        # Convert timestamps to Skyfield Time objects once for efficiency
        ts_objects = [self.ts.from_datetime(t) for t in timestamps]
        measured_freqs_np = np.array(measured_freqs)

        def _residuals(guess_lat_lon):
            lat, lon = guess_lat_lon
            theoretical_freqs = []
            
            for t in ts_objects:
                v_rel = self.propagator.calculate_range_rate(t, lat, lon)
                f_theo = self.tx_freq * (1 - v_rel / C)
                theoretical_freqs.append(f_theo)
            
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
    def plot_results(timestamps, measured_freqs, solved_lat, solved_lon, true_lat, true_lon, propagator, tx_freq):
        ts = load.timescale()
        
        # --- Plot 1: The S-Curve ---
        plt.figure(figsize=(10, 6))
        plt.scatter(timestamps, measured_freqs, s=10, label='Noisy Measured Data', color='blue', alpha=0.5)
        
        # Calculate fitted curve
        fitted_freqs = []
        for t_dt in timestamps:
            t = ts.from_datetime(t_dt)
            v_rel = propagator.calculate_range_rate(t, solved_lat, solved_lon)
            f_theo = tx_freq * (1 - v_rel / C)
            fitted_freqs.append(f_theo)
            
        plt.plot(timestamps, fitted_freqs, label=f'Solved Fit (Lat:{solved_lat:.2f}, Lon:{solved_lon:.2f})', color='red', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Doppler Curve: Measured vs Solved')
        plt.legend()
        plt.grid(True)
        plt.savefig('doppler_curve.png')
        print("Saved doppler_curve.png")

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
    # 1. Initialize TLEManager and get Satellite
    tle_manager = TLEManager()
    try:
        sat = tle_manager.get_satellite(SATELLITE_NAME)
        print(f"Loaded satellite: {sat.name}")
    except Exception as e:
        print(f"Error loading satellite: {e}")
        return

    # 2. Define Hidden Truth Location (e.g., Mumbai)
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
    print(f"Hidden Truth Location: Lat {TRUE_LAT}, Lon {TRUE_LON}")

    # 3. Setup Propagator and Simulator
    propagator = SatellitePropagator(sat)
    simulator = SignalSimulator(propagator)

    # Find a good pass time (when satellite is above horizon for Mumbai)
    # For simulation purposes, we'll just pick a time and hope it's visible or just simulate physics anyway.
    # Ideally we should find_events, but to keep it simple as requested, we'll just pick 'now' or a known recent time.
    # Let's search for the next pass to make it realistic.
    ts = load.timescale()
    t0 = ts.now()
    t1 = ts.from_datetime(t0.utc_datetime() + datetime.timedelta(days=1))
    
    print("Searching for next pass over Mumbai...")
    observer = wgs84.latlon(TRUE_LAT, TRUE_LON)
    t, events = sat.find_events(observer, t0, t1, altitude_degrees=10.0)
    
    if len(t) == 0:
        print("No pass found in next 24 hours. Using current time (might be below horizon).")
        start_time = datetime.datetime.now(datetime.timezone.utc)
    else:
        # Find the first rise event
        for ti, event in zip(t, events):
            if event == 0: # 0 is rise
                start_time = ti.utc_datetime()
                print(f"Pass found starting at {start_time}")
                break
        else:
             start_time = datetime.datetime.now(datetime.timezone.utc)

    # Generate 10 minutes (600 seconds) of data
    DURATION = 600
    NOISE_STD = 50.0 # Hz

    timestamps, measured_freqs = simulator.generate_data(TRUE_LAT, TRUE_LON, start_time, DURATION, NOISE_STD)

    # 4. Solve
    solver = GeolocationSolver(propagator)
    est_lat, est_lon = solver.solve(timestamps, measured_freqs)

    # 5. Analysis
    error_km = haversine_distance(TRUE_LAT, TRUE_LON, est_lat, est_lon)
    print(f"\n--- Results ---")
    print(f"True Location:      {TRUE_LAT:.4f}, {TRUE_LON:.4f}")
    print(f"Estimated Location: {est_lat:.4f}, {est_lon:.4f}")
    print(f"Error: {error_km:.2f} km")

    # 6. Visualize
    Visualizer.plot_results(timestamps, measured_freqs, est_lat, est_lon, TRUE_LAT, TRUE_LON, propagator, TX_FREQUENCY)

if __name__ == "__main__":
    main()
