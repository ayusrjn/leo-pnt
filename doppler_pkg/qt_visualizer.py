import sys
import time
import datetime
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QTextEdit, QSplitter
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

                     
pg.setConfigOptions(antialias=True)

class SDRWorker(QThread):
    """                                                                                                                                                                                     
    Background thread to handle SDR capture.
    """
    data_ready = pyqtSignal(object, float, object)                            
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
                                              
                if self.chunks and count >= self.chunks:
                    break
                
                         
                samples = self.sdr.capture_samples(1.0)              
                freq, fft_data = self.sdr.measure_frequency(samples)
                now = datetime.datetime.now(datetime.timezone.utc)
                
                                                              
                                                         
                                                
                factor = len(fft_data) // 2048
                if factor > 1:
                                                                                         
                                                              
                                                                                                                    
                    fft_downsampled = fft_data[::factor]
                else:
                    fft_downsampled = fft_data
                
                           
                self.data_ready.emit(now, freq, fft_downsampled)
                
                count += 1
                
                                                                                   
                                                              
                                                                            
                                                       
                if "MockSDR" in str(type(self.sdr)):
                    time.sleep(0.1)                     
                    
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
        self.fft_size = 2048                                 
        
                  
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
                                                  
                                                
        self.main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.main_splitter)
        
                            
        self.map_plot = pg.PlotWidget(title="Estimated Position (Lat/Lon)")
        self.map_plot.setLabel('left', 'Latitude')
        self.map_plot.setLabel('bottom', 'Longitude')
        self.map_plot.showGrid(x=True, y=True)
                                  
        self.true_pos_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 255), symbol='x')
        self.map_plot.addItem(self.true_pos_scatter)
                                       
        self.est_pos_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255), symbol='o')
        self.map_plot.addItem(self.est_pos_scatter)
                           
        self.pos_history_curve = self.map_plot.plot(pen=pg.mkPen('r', width=1, style=Qt.DotLine))
        self.pos_history_x = []
        self.pos_history_y = []
        
        self.main_splitter.addWidget(self.map_plot)
        
                                          
        self.doppler_plot = pg.PlotWidget(title="Satellite Doppler Tracks")
        self.doppler_plot.setLabel('left', 'Doppler Shift', units='Hz')
        self.doppler_plot.setLabel('bottom', 'Time Step')
        self.doppler_plot.showGrid(x=True, y=True)
        self.doppler_plot.addLegend()
        
        self.sat_curves = {}                                                       
        self.time_step = 0
        
        self.main_splitter.addWidget(self.doppler_plot)
        
                               
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: black; color: lime; font-family: Monospace;")
        
        self.main_splitter.addWidget(self.log_text)
        
                           
        self.main_splitter.setSizes([400, 300, 100])

    def update_log(self, message):
        self.log_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
                     
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_position(self, lat, lon, true_lat=None, true_lon=None):
        self.est_pos_scatter.setData([lon], [lat])
        
        self.pos_history_x.append(lon)
        self.pos_history_y.append(lat)
        self.pos_history_curve.setData(self.pos_history_x, self.pos_history_y)
        
        if true_lat is not None and true_lon is not None:
            self.true_pos_scatter.setData([true_lon], [true_lat])
            
                     
                                   

    def update_doppler(self, sats_data):
        """
        sats_data: list of {'sat_id': int, 'doppler': float}
        """
        self.time_step += 1
        
                                       
                              
        def get_color(sat_id):
            np.random.seed(sat_id)
            return tuple(np.random.randint(50, 255, 3))
            
        current_sats = set()
        
        for sat in sats_data:
            sat_id = sat['sat_id']
            doppler = sat['doppler']
            current_sats.add(sat_id)
            
            if sat_id not in self.sat_curves:
                                  
                color = get_color(sat_id)
                curve = self.doppler_plot.plot(pen=pg.mkPen(color, width=2), name=f"Sat {sat_id}")
                self.sat_curves[sat_id] = {'curve': curve, 'x': [], 'y': []}
            
                         
            self.sat_curves[sat_id]['x'].append(self.time_step)
            self.sat_curves[sat_id]['y'].append(doppler)
            
                           
            if len(self.sat_curves[sat_id]['x']) > 100:
                self.sat_curves[sat_id]['x'].pop(0)
                self.sat_curves[sat_id]['y'].pop(0)
                
            self.sat_curves[sat_id]['curve'].setData(self.sat_curves[sat_id]['x'], self.sat_curves[sat_id]['y'])

    def update_data(self, timestamp, freq, spectrum):
        pass                                 
            
