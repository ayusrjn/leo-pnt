import sys
import time
import datetime
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QTextEdit, QSplitter
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

# Configure pyqtgraph
pg.setConfigOptions(antialias=True)

class SDRWorker(QThread):
    """                                                                                                                                                                                     
    Background thread to handle SDR capture.
    """
    data_ready = pyqtSignal(object, float, object) # timestamp, freq, spectrum
    finished = pyqtSignal()

    def __init__(self, sdr_interface, duration=None, chunks=None):
        super().__init__()
        self.sdr = sdr_interface
        self.duration = duration
        self.chunks = chunks
        self.running = True

    def run(self):
        print("[SDRWorker] Started")
        count = 0
        try:
            while self.running:
                # Check termination conditions
                if self.chunks and count >= self.chunks:
                    break
                
                # Capture
                samples = self.sdr.capture_samples(1.0) # 1 sec chunk
                freq, fft_data = self.sdr.measure_frequency(samples)
                now = datetime.datetime.now(datetime.timezone.utc)
                
                # Downsample FFT for visualization (avoid OOM)
                # Original size ~2M points. Target ~2048.
                # Simple decimation or averaging
                factor = len(fft_data) // 2048
                if factor > 1:
                    # Use mean of chunks to preserve power info roughly, or just decimate
                    # Decimation is faster: fft_data[::factor]
                    # Resizing using mean is better for waterfall to not miss peaks, but decimate is okay for visual
                    fft_downsampled = fft_data[::factor]
                else:
                    fft_downsampled = fft_data
                
                # Emit data
                self.data_ready.emit(now, freq, fft_downsampled)
                
                count += 1
                
                # If mock, sleep a bit to simulate real-time if capture is too fast
                # (MockSDR is instant, real SDR blocks for 1s)
                # But SDRInterface.capture_samples is blocking for real SDR.
                # For MockSDR, we need to slow it down.
                if "MockSDR" in str(type(self.sdr)):
                    time.sleep(0.1) # 10x speed for mock
                    
        except Exception as e:
            print(f"[SDRWorker] Error: {e}")
        finally:
            print("[SDRWorker] Finished")
            self.finished.emit()

    def stop(self):
        self.running = False

class DashboardWindow(QMainWindow):
    def __init__(self, center_freq, sample_rate):
        super().__init__()
        self.setWindowTitle(f"Cognitive LEO-PNT Tracker | Live Dashboard ({center_freq/1e6} MHz)")
        self.resize(1000, 800)
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.fft_size = 2048 # Target size after downsampling
        
        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # --- NEW: Solver Visualization Layout ---
        # Splitter to resize between Map and Log
        self.main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.main_splitter)
        
        # Top Widget for Map
        self.map_plot = pg.PlotWidget(title="Estimated Position (Lat/Lon)")
        self.map_plot.setLabel('left', 'Latitude')
        self.map_plot.setLabel('bottom', 'Longitude')
        self.map_plot.showGrid(x=True, y=True)
        # Add True Position Marker
        self.true_pos_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 255), symbol='x')
        self.map_plot.addItem(self.true_pos_scatter)
        # Add Estimated Position Marker
        self.est_pos_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255), symbol='o')
        self.map_plot.addItem(self.est_pos_scatter)
        # Add History Trace
        self.pos_history_curve = self.map_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DotLine))
        self.pos_history_x = []
        self.pos_history_y = []
        
        self.main_splitter.addWidget(self.map_plot)
        
        # Middle Widget for Doppler Tracks
        self.doppler_plot = pg.PlotWidget(title="Satellite Doppler Tracks")
        self.doppler_plot.setLabel('left', 'Doppler Shift', units='Hz')
        self.doppler_plot.setLabel('bottom', 'Time Step')
        self.doppler_plot.showGrid(x=True, y=True)
        self.doppler_plot.addLegend()
        
        self.sat_curves = {} # {sat_id: {'curve': PlotCurveItem, 'x': [], 'y': []}}
        self.time_step = 0
        
        self.main_splitter.addWidget(self.doppler_plot)
        
        # Bottom Widget for Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: black; color: lime; font-family: Monospace;")
        
        self.main_splitter.addWidget(self.log_text)
        
        # Set initial sizes
        self.main_splitter.setSizes([400, 300, 100])

    def update_log(self, message):
        self.log_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
        # Auto scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_position(self, lat, lon, true_lat=None, true_lon=None):
        self.est_pos_scatter.setData([lon], [lat])
        
        self.pos_history_x.append(lon)
        self.pos_history_y.append(lat)
        self.pos_history_curve.setData(self.pos_history_x, self.pos_history_y)
        
        if true_lat is not None and true_lon is not None:
            self.true_pos_scatter.setData([true_lon], [true_lat])
            
        # Auto range?
        # self.map_plot.autoRange()

    def update_doppler(self, sats_data):
        """
        sats_data: list of {'sat_id': int, 'doppler': float}
        """
        self.time_step += 1
        
        # Assign colors based on sat_id
        # Simple hash to color
        def get_color(sat_id):
            np.random.seed(sat_id)
            return tuple(np.random.randint(50, 255, 3))
            
        current_sats = set()
        
        for sat in sats_data:
            sat_id = sat['sat_id']
            doppler = sat['doppler']
            current_sats.add(sat_id)
            
            if sat_id not in self.sat_curves:
                # Create new curve
                color = get_color(sat_id)
                curve = self.doppler_plot.plot(pen=pg.mkPen(color, width=2), name=f"Sat {sat_id}")
                self.sat_curves[sat_id] = {'curve': curve, 'x': [], 'y': []}
            
            # Update data
            self.sat_curves[sat_id]['x'].append(self.time_step)
            self.sat_curves[sat_id]['y'].append(doppler)
            
            # Limit history
            if len(self.sat_curves[sat_id]['x']) > 100:
                self.sat_curves[sat_id]['x'].pop(0)
                self.sat_curves[sat_id]['y'].pop(0)
                
            self.sat_curves[sat_id]['curve'].setData(self.sat_curves[sat_id]['x'], self.sat_curves[sat_id]['y'])

    def update_data(self, timestamp, freq, spectrum):
        pass # Spectrum visualization removed
            
