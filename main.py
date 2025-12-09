import datetime
import random
from skyfield.api import wgs84, load
from doppler_pkg import TLEManager, SatellitePropagator, SignalSimulator, Visualizer, haversine_distance, TX_FREQUENCY
from doppler_pkg.selector import SatelliteSelector

def main():
    # 1. Initialize TLEManager and get all satellites
    tle_manager = TLEManager()
    
    # Load all satellites from TLE
    # We need to access the raw list from skyfield, TLEManager helper gets one by name.
    # Let's use the file directly via skyfield as we did in TLEManager but for all.
    ts = load.timescale()
    satellites = load.tle_file(tle_manager.filename)
    
    # Filter for NOAA satellites (to keep the list reasonable and relevant to APT)
    # The prompt says "Load all NOAA satellites".
    candidates = [sat for sat in satellites if "NOAA" in sat.name]
    
    print(f"Loaded {len(candidates)} NOAA candidates.")
    if not candidates:
        print("No NOAA satellites found. Exiting.")
        return

    # 2. Randomly select "Mystery Satellite" (The Truth)
    mystery_sat = random.choice(candidates)
    print(f"Mystery Satellite Selected: {mystery_sat.name} (Hidden from Solver)")

    # 3. Define Hidden Truth Location (e.g., Mumbai)
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
    print(f"Hidden Truth Location: Lat {TRUE_LAT}, Lon {TRUE_LON}")

    # 4. Generate Signal Data for Mystery Satellite
    propagator = SatellitePropagator(mystery_sat)
    simulator = SignalSimulator(propagator)

    # Find a pass
    t0 = ts.now()
    t1 = ts.from_datetime(t0.utc_datetime() + datetime.timedelta(days=1))
    
    print(f"Searching for next pass of {mystery_sat.name} over Mumbai...")
    observer = wgs84.latlon(TRUE_LAT, TRUE_LON)
    t, events = mystery_sat.find_events(observer, t0, t1, altitude_degrees=10.0)
    
    if len(t) == 0:
        print("No pass found in next 24 hours. Using current time (might be below horizon).")
        start_time = datetime.datetime.now(datetime.timezone.utc)
    else:
        for ti, event in zip(t, events):
            if event == 0: # Rise
                start_time = ti.utc_datetime()
                print(f"Pass found starting at {start_time}")
                break
        else:
             start_time = datetime.datetime.now(datetime.timezone.utc)

    DURATION = 600
    NOISE_STD = 50.0 # Hz
    timestamps, measured_freqs = simulator.generate_data(TRUE_LAT, TRUE_LON, start_time, DURATION, NOISE_STD)

    # 5. Blind Identification
    print("\n--- Starting Blind Identification ---")
    selector = SatelliteSelector()
    
    # Pass ONLY the data, not the mystery_sat
    identified_sat, est_location, min_cost = selector.identify_satellite(candidates, timestamps, measured_freqs)
    
    if identified_sat is None:
        print("Failed to identify any satellite.")
        return

    est_lat, est_lon = est_location
    
    # 6. Analysis & Results
    print(f"\n--- Results ---")
    print(f"Mystery Satellite:   {mystery_sat.name}")
    print(f"Identified Satellite: {identified_sat.name}")
    
    is_correct = mystery_sat.name == identified_sat.name
    print(f"Identification:       {'SUCCESS' if is_correct else 'FAILURE'}")
    
    error_km = haversine_distance(TRUE_LAT, TRUE_LON, est_lat, est_lon)
    print(f"True Location:        {TRUE_LAT:.4f}, {TRUE_LON:.4f}")
    print(f"Estimated Location:   {est_lat:.4f}, {est_lon:.4f}")
    print(f"Location Error:       {error_km:.2f} km")
    print(f"Residual Cost:        {min_cost:.2f}")

    # 7. Visualize (using the identified satellite's propagator)
    id_propagator = SatellitePropagator(identified_sat)
    Visualizer.plot_results(timestamps, measured_freqs, est_lat, est_lon, TRUE_LAT, TRUE_LON, id_propagator, TX_FREQUENCY)

if __name__ == "__main__":
    main()
