import zmq
import pickle
import time
import datetime
import random
from doppler_pkg.sdr_interface import SDRInterface, MockSDR, HAS_RTLSDR

# CONFIGURATION
USE_LIVE_SDR = False # Set to True to use real RTL-SDR
USE_MOCK_SDR = True  # Set to True to simulate SDR if Live is False
ZMQ_PORT = 5555

def main():
    print(f"--- COMPUTE NODE STARTED (Port {ZMQ_PORT}) ---")
    
    # 1. Setup ZMQ Publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"ZMQ Publisher bound to tcp://*:{ZMQ_PORT}")
    
    # 2. Setup SDR
    sdr = None
    if USE_LIVE_SDR and HAS_RTLSDR:
        print("Initializing Live SDR...")
        sdr = SDRInterface()
    elif USE_MOCK_SDR:
        print("Initializing Mock SDR...")
        sdr = MockSDR()
    else:
        print("No SDR source available.")
        return

    print(f"SDR Center Freq: {sdr.center_freq/1e6} MHz")
    
    try:
        while True:
            # Capture
            samples = sdr.capture_samples(1.0) # 1 sec chunk
            freq, fft_data = sdr.measure_frequency(samples)
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Downsample FFT for network efficiency
            # Original ~2M points -> 2048 points
            factor = len(fft_data) // 2048
            if factor > 1:
                fft_downsampled = fft_data[::factor]
            else:
                fft_downsampled = fft_data
                
            # Prepare Payload
            payload = {
                'timestamp': now,
                'freq': freq,
                'spectrum': fft_downsampled,
                'center_freq': sdr.center_freq,
                'sample_rate': sdr.sample_rate
            }
            
            # Serialize and Send
            data = pickle.dumps(payload)
            socket.send(data)
            
            print(f"Sent data: Freq={freq:.2f} Hz, Size={len(data)} bytes")
            
            # Rate limiting for Mock SDR
            if USE_MOCK_SDR:
                time.sleep(0.1) # Simulate real-time
                
    except KeyboardInterrupt:
        print("\nStopping Compute Node...")
    finally:
        sdr.close()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
