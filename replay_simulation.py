import time
import zmq
import pickle
import pandas as pd
import numpy as np
from doppler_pkg.position_engine import PositionEngine
from doppler_pkg.constants import TX_FREQUENCY

# CONFIGURATION
ZMQ_PORT = 5555
REPLAY_SPEED = 2.0 # 1.0 = Realtime, 2.0 = 2x speed

def main():
    print("--- REPLAY SIMULATION STARTED ---")
    
    # 1. Setup ZMQ Publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"Publishing on port {ZMQ_PORT}...")
    
    # 2. Load Data
    try:
        df = pd.read_csv("leo_s9_results.csv")
    except FileNotFoundError:
        print("Error: leo_s9_results.csv not found.")
        return

    # 3. Initialize Solver
    engine = PositionEngine()
    
    # True Position (for reference)
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
    RE = 6378137.0
    
    epochs = df['time'].unique()
    epochs.sort()
    
    print("Waiting 5 seconds for Display Node to connect...")
    time.sleep(5)
    
    for t in epochs:
        print(f"Processing Epoch {t}...")
        
        # 1. Send Log
        log_msg = {'type': 'log', 'msg': f"--- Epoch {t} ---"}
        socket.send(pickle.dumps(log_msg))
        
        epoch_data = df[df['time'] == t]
        observations = []
        
        # 2. Send Spectrum (Fake)
        # We don't have raw spectrum, but we have Doppler.
        # Let's generate a fake spectrum with peaks at the Doppler frequencies.
        fft_size = 2048
        sample_rate = 2.048e6
        spectrum = np.random.normal(0, 0.1, fft_size) + 1j * np.random.normal(0, 0.1, fft_size)
        # Add peaks
        freqs = np.linspace(-sample_rate/2, sample_rate/2, fft_size)
        
        visible_sats = []
        
        for _, row in epoch_data.iterrows():
            doppler = row['doppler']
            sat_id = row['sat_id']
            visible_sats.append(str(sat_id))
            
            # Add peak to spectrum
            # Find bin
            idx = (np.abs(freqs - doppler)).argmin()
            spectrum[idx] += 10.0 # Strong peak
            
            observations.append({
                'sat_x': row['sat_x'],
                'sat_y': row['sat_y'],
                'sat_z': row['sat_z'],
                'pseudorange': row['pseudorange']
            })
            
        # Send Spectrum
        spec_msg = {
            'type': 'spectrum',
            'timestamp': time.time(),
            'freq': TX_FREQUENCY,
            'spectrum': spectrum
        }
        socket.send(pickle.dumps(spec_msg))
        
        # Log Visible Sats
        socket.send(pickle.dumps({'type': 'log', 'msg': f"Visible Sats: {', '.join(visible_sats)}"}))
        
        # 2.5 Send Doppler Data
        doppler_list = []
        for _, row in epoch_data.iterrows():
            doppler_list.append({
                'sat_id': int(row['sat_id']),
                'doppler': row['doppler']
            })
        
        socket.send(pickle.dumps({
            'type': 'doppler_data',
            'timestamp': t,
            'sats': doppler_list
        }))
        
        # 3. Solve Position
        socket.send(pickle.dumps({'type': 'log', 'msg': "Solving Position..."}))
        est_pos, bias, dop, res = engine.solve_epoch(observations, prior_pos=None)
        
        if est_pos is not None:
            # Convert ECEF to Lat/Lon
            x, y, z = est_pos
            p = np.sqrt(x**2 + y**2)
            lon = np.arctan2(y, x)
            lat = np.arctan2(z, p * (1 - 0.00669437999014)) # Simple approx
            
            lat_deg = np.degrees(lat)
            lon_deg = np.degrees(lon)
            
            socket.send(pickle.dumps({
                'type': 'position',
                'lat': lat_deg,
                'lon': lon_deg,
                'true_lat': TRUE_LAT,
                'true_lon': TRUE_LON
            }))
            
            socket.send(pickle.dumps({'type': 'log', 'msg': f"Solved: {lat_deg:.4f}, {lon_deg:.4f} (PDOP: {dop['PDOP']:.2f})"}))
        else:
            socket.send(pickle.dumps({'type': 'log', 'msg': "Solver Failed (Singular Matrix)"}))
            
        time.sleep(1.0 / REPLAY_SPEED)

if __name__ == "__main__":
    main()
