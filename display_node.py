import sys
import zmq
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from doppler_pkg.qt_visualizer import DashboardWindow

# CONFIGURATION
# IP of the Compute Node (Pi 1). Use 'localhost' for testing on same machine.
COMPUTE_NODE_IP = "localhost" 
ZMQ_PORT = 5555

class ZMQSubscriberThread(QThread):
    """
    Background thread to receive data from Compute Node via ZMQ.
    """
    data_ready = pyqtSignal(object, float, object) # timestamp, freq, spectrum
    log_ready = pyqtSignal(str)
    pos_ready = pyqtSignal(float, float, float, float) # lat, lon, true_lat, true_lon
    doppler_ready = pyqtSignal(object) # list of dicts
    
    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = True

    def run(self):
        print(f"[ZMQSubscriber] Connecting to tcp://{self.ip}:{self.port}...")
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.ip}:{self.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics
        
        while self.running:
            try:
                # Receive
                data = socket.recv()
                payload = pickle.loads(data)
                
                # Extract
                msg_type = payload.get('type', 'spectrum')
                
                if msg_type == 'spectrum':
                    timestamp = payload['timestamp']
                    freq = payload['freq']
                    spectrum = payload['spectrum']
                    self.data_ready.emit(timestamp, freq, spectrum)
                    
                elif msg_type == 'log':
                    self.log_ready.emit(payload['msg'])
                    
                elif msg_type == 'position':
                    self.pos_ready.emit(payload['lat'], payload['lon'], payload.get('true_lat', 0.0), payload.get('true_lon', 0.0))
                    
                elif msg_type == 'doppler_data':
                    self.doppler_ready.emit(payload['sats'])
                    
            except Exception as e:
                print(f"[ZMQSubscriber] Error: {e}")
                break
                
            except Exception as e:
                print(f"[ZMQSubscriber] Error: {e}")
                break
        
        socket.close()
        context.term()

    def stop(self):
        self.running = False

def main():
    print("--- DISPLAY NODE STARTED ---")
    
    app = QApplication(sys.argv)
    
    # We need initial params to setup window. 
    # Ideally we wait for first packet, or just use defaults.
    # Let's use defaults and update later if needed.
    default_center_freq = 137.5e6
    default_sample_rate = 2.048e6
    
    window = DashboardWindow(default_center_freq, default_sample_rate)
    window.setWindowTitle(f"Display Node - Connected to {COMPUTE_NODE_IP}")
    window.show()
    
    # Setup Subscriber
    subscriber = ZMQSubscriberThread(COMPUTE_NODE_IP, ZMQ_PORT)
    subscriber.data_ready.connect(window.update_data)
    subscriber.log_ready.connect(window.update_log)
    subscriber.pos_ready.connect(window.update_position)
    subscriber.doppler_ready.connect(window.update_doppler)
    subscriber.start()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()

if __name__ == "__main__":
    main()
