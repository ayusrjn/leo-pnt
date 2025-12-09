import sys
import time
import datetime
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
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
        
        # --- PLOT 1: Live Spectrum ---
        self.spec_plot = pg.PlotWidget(title="Real-Time Spectrum (Zoomed)")
        self.spec_plot.setLabel('left', 'Power', units='dB')
        self.spec_plot.setLabel('bottom', 'Frequency Offset', units='Hz')
        self.spec_plot.showGrid(x=True, y=True)
        
        # ** FIX 1: ZOOM IN ** # Only show +/- 25 kHz around the center. 
        self.spec_plot.setXRange(-25000, 25000)
        
        self.spec_curve = self.spec_plot.plot(pen='y') # Yellow line
        layout.addWidget(self.spec_plot)
        
        # --- PLOT 2: Waterfall ---
        self.waterfall_view = pg.PlotWidget(title="Waterfall History (Doppler Trace)")
        self.waterfall_view.setLabel('left', 'Time', units='scans')
        self.waterfall_view.setLabel('bottom', 'Frequency Bin')
        
        self.img_item = pg.ImageItem()
        self.waterfall_view.addItem(self.img_item)
        
        # Colormap (Blue background -> Yellow signal)
        pos = np.array([0.0, 0.2, 1.0])
        color = np.array([[0,0,30,255], [0,0,255,255], [255,255,0,255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        
        layout.addWidget(self.waterfall_view)
        
        # Buffer for Waterfall
        self.history_size = 200
        self.waterfall_data = np.zeros((self.history_size, self.fft_size)) 

    def update_data(self, timestamp, freq, spectrum):
        # spectrum is the magnitude from SDRWorker
        # We need to convert to dB for visualization
        
        # Ensure spectrum size matches buffer (handle potential mismatch from simple downsampling)
        if len(spectrum) != self.fft_size:
            # Resample to match fft_size if needed, or just slice/pad
            # For now, let's assume SDRWorker sends roughly the right size or we adjust
            # Simple linear interpolation to resize if needed would be better, but let's just clip/pad for speed
            if len(spectrum) > self.fft_size:
                spectrum = spectrum[:self.fft_size]
            else:
                spectrum = np.pad(spectrum, (0, self.fft_size - len(spectrum)))
        
        # Convert to dB
        psd = 20 * np.log10(np.abs(spectrum) + 1e-12)
        
        # 1. Update Spectrum Plot
        # Create freq axis centered at 0
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate))
        # But wait, SDRWorker sends us `fft_downsampled`. 
        # If we downsampled by factor N, effective sample rate for axis is sample_rate / N?
        # No, downsampling in time domain reduces bandwidth. 
        # Downsampling in frequency domain (decimating FFT output) just reduces resolution.
        # My SDRWorker code `fft_downsampled = fft_data[::factor]` decimates the FFT bins.
        # So the frequency span is the same (sample_rate), but fewer points.
        
        # Re-calculate freqs for the plot
        freqs = np.linspace(-self.sample_rate/2, self.sample_rate/2, self.fft_size)
        
        self.spec_curve.setData(freqs, psd)
        
        # 2. Update Waterfall
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1] = psd
        
        # ** FIX 2: CONTRAST **
        # Set levels manually to hide noise. 
        # Noise floor is usually around 0-10, Signal is 20-40.
        self.img_item.setImage(self.waterfall_data.T, autoLevels=False, levels=[10, 50])
        
        # Update title with latest frequency
        self.spec_plot.setTitle(f"Real-Time Spectrum (Peak: {freq:.2f} Hz)")
            
