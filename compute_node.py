import zmq
import pickle
import time
import datetime
import random
from doppler_pkg.sdr_interface import SDRInterface, MockSDR, HAS_RTLSDR

               
USE_LIVE_SDR = False                                  
USE_MOCK_SDR = True                                                
ZMQ_PORT = 5555

def main():
    print(f"--- COMPUTE NODE STARTED (Port {ZMQ_PORT}) ---")
    
                            
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"ZMQ Publisher bound to tcp://*:{ZMQ_PORT}")
    
                  
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
                     
            samples = sdr.capture_samples(1.0)              
            freq, fft_data = sdr.measure_frequency(samples)
            now = datetime.datetime.now(datetime.timezone.utc)
            
                                                   
                                                
            factor = len(fft_data) // 2048
            if factor > 1:
                fft_downsampled = fft_data[::factor]
            else:
                fft_downsampled = fft_data
                
                             
            payload = {
                'timestamp': now,
                'freq': freq,
                'spectrum': fft_downsampled,
                'center_freq': sdr.center_freq,
                'sample_rate': sdr.sample_rate
            }
            
                                
            data = pickle.dumps(payload)
            socket.send(data)
            
            print(f"Sent data: Freq={freq:.2f} Hz, Size={len(data)} bytes")
            
                                        
            if USE_MOCK_SDR:
                time.sleep(0.1)                     
                
    except KeyboardInterrupt:
        print("\nStopping Compute Node...")
    finally:
        sdr.close()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
