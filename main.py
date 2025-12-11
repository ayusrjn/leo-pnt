import sys
import datetime
import random
import time
from skyfield.api import wgs84, load
from doppler_pkg import TLEManager, SatellitePropagator, SignalSimulator, Visualizer, haversine_distance, TX_FREQUENCY
from doppler_pkg.selector import SatelliteSelector
from doppler_pkg.sdr_interface import SDRInterface, MockSDR, HAS_RTLSDR
from doppler_pkg.qt_visualizer import QApplication, DashboardWindow, SDRWorker

               
USE_LIVE_SDR = False                                  
USE_MOCK_SDR = True                                                                   

def main():
                                                     
    tle_manager = TLEManager()
    ts = load.timescale()
    satellites = load.tle_file(tle_manager.filename)
    candidates = [sat for sat in satellites if "NOAA" in sat.name]
    
    print(f"Loaded {len(candidates)} NOAA candidates.")
    if not candidates:
        print("No NOAA satellites found. Exiting.")
        return

                         
    sdr = None
    mystery_sat = None
    
    if USE_LIVE_SDR and HAS_RTLSDR:
        print("\n--- LIVE SDR CAPTURE MODE ---")
        sdr = SDRInterface()
        print(f"SDR Initialized. Center Freq: {sdr.sdr.center_freq/1e6} MHz")
    elif USE_MOCK_SDR:
        print("\n--- MOCK SDR MODE ---")
        mystery_sat = random.choice(candidates)
        print(f"Mocking signal from: {mystery_sat.name}")
        sdr = MockSDR()
    else:
        print("No SDR source selected.")
        return

                               
    app = QApplication(sys.argv)
    window = DashboardWindow(sdr.center_freq if hasattr(sdr, 'center_freq') else sdr.sdr.center_freq, 
                             sdr.sample_rate)
    window.show()
    
                            
                                           
    worker = SDRWorker(sdr, chunks=60)
    worker.data_ready.connect(window.update_data)
    
                                         
    timestamps = []
    measured_freqs = []
    
    def on_data(ts, freq, spectrum):
        timestamps.append(ts)
        measured_freqs.append(freq)
        print(f"Freq: {freq:.2f} Hz")
        
    worker.data_ready.connect(on_data)
    
                                          
    def on_finished():
        print("Capture finished.")
        sdr.close()
        window.close()
        
                           
        run_solver(candidates, timestamps, measured_freqs, mystery_sat)
        app.quit()
        
    worker.finished.connect(on_finished)
    worker.start()
    
                    
    sys.exit(app.exec_())

def run_solver(candidates, timestamps, measured_freqs, mystery_sat=None):
                             
    print("\n--- Starting Blind Identification ---")
    
                                                                                                
    if USE_MOCK_SDR and mystery_sat:
                                                                      
                                                                                                  
        print("Generating high-fidelity mock data for solver...")
        propagator = SatellitePropagator(mystery_sat)
        simulator = SignalSimulator(propagator)
        
                     
        ts = load.timescale()
        t0 = ts.now()
        t1 = ts.from_datetime(t0.utc_datetime() + datetime.timedelta(days=1))
        TRUE_LAT = 19.0760
        TRUE_LON = 72.8777
        observer = wgs84.latlon(TRUE_LAT, TRUE_LON)
        t, events = mystery_sat.find_events(observer, t0, t1, altitude_degrees=10.0)
        
        pass_start = t[0].utc_datetime() if len(t) > 0 else datetime.datetime.now(datetime.timezone.utc)
        timestamps, measured_freqs = simulator.generate_data(TRUE_LAT, TRUE_LON, pass_start, 600, 50.0)
    
    selector = SatelliteSelector()
    identified_sat, est_location, min_cost = selector.identify_satellite(candidates, timestamps, measured_freqs)
    
    if identified_sat is None:
        print("Failed to identify any satellite.")
        return

    est_lat, est_lon = est_location
    
                           
    print(f"\n--- Results ---")
    print(f"Identified Satellite: {identified_sat.name}")
    
    if mystery_sat:
        print(f"Mystery Satellite:   {mystery_sat.name}")
        is_correct = mystery_sat.name == identified_sat.name
        print(f"Identification:       {'SUCCESS' if is_correct else 'FAILURE'}")
        
    print(f"Estimated Location:   {est_lat:.4f}, {est_lon:.4f}")
    print(f"Residual Cost:        {min_cost:.2f}")

                                           
                                                               
                                     
                                                                                               
                         
    id_propagator = SatellitePropagator(identified_sat)
    Visualizer.plot_results(timestamps, measured_freqs, est_lat, est_lon, 19.0760, 72.8777, id_propagator, TX_FREQUENCY)

if __name__ == "__main__":
    main()
