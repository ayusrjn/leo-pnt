import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load
from .constants import C
from .utils import haversine_distance

class Visualizer:
    """
    Module E: Visualizer (Analysis)
    Responsibility: Plot the results.
    """
    @staticmethod
    def plot_dashboard(timestamps, measured_freqs, solved_lat, solved_lon, true_lat, true_lon, propagator, tx_freq, spectrogram_data, sample_rate):
        """
        Generates a comprehensive dashboard with:
        1. Waterfall Plot (Spectrogram)
        2. Doppler Curve (Measured vs Solved)
        3. Geolocation Map
        """
        ts = load.timescale()
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        # --- 1. Waterfall Plot ---
        ax1 = fig.add_subplot(gs[0, :]) # Top row spanning both columns
        
        if spectrogram_data is not None and len(spectrogram_data) > 0:
            # spectrogram_data is list of 1D arrays (FFT magnitudes)
            # Convert to 2D array: Time x Frequency
            spec_matrix = np.array(spectrogram_data)
            
            # Create frequency axis (centered)
            num_bins = spec_matrix.shape[1]
            freqs = np.fft.fftshift(np.fft.fftfreq(num_bins, d=1/sample_rate))
            
            # Create time axis
            # We assume uniform sampling for the plot roughly
            duration = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 1
            extent = [freqs[0]/1000, freqs[-1]/1000, duration, 0] # kHz vs Seconds
            
            # Shift zero freq to center
            spec_matrix_shifted = np.fft.fftshift(spec_matrix, axes=1)
            
            # Log scale for better visibility
            spec_db = 10 * np.log10(np.abs(spec_matrix_shifted) + 1e-9)
            
            im = ax1.imshow(spec_db, aspect='auto', extent=extent, cmap='inferno')
            plt.colorbar(im, ax=ax1, label='Power (dB)')
            ax1.set_title(f'Waterfall Plot (Center: {tx_freq/1e6:.3f} MHz)')
            ax1.set_xlabel('Frequency Offset (kHz)')
            ax1.set_ylabel('Time (s)')
        else:
            ax1.text(0.5, 0.5, "No Spectrogram Data", ha='center', va='center')

        # --- 2. Doppler Curve ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(timestamps, measured_freqs, s=10, label='Measured', color='blue', alpha=0.5)
        
        fitted_freqs = []
        for t_dt in timestamps:
            t = ts.from_datetime(t_dt)
            v_rel = propagator.calculate_range_rate(t, solved_lat, solved_lon)
            f_theo = tx_freq * (1 - v_rel / C)
            fitted_freqs.append(f_theo)
            
        ax2.plot(timestamps, fitted_freqs, label='Solved Fit', color='red', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Doppler Curve')
        ax2.legend()
        ax2.grid(True)

        # --- 3. Map ---
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(true_lon, true_lat, color='green', marker='*', s=200, label='True')
        ax3.scatter(solved_lon, solved_lat, color='red', marker='x', s=100, label='Est')
        ax3.plot([true_lon, solved_lon], [true_lat, solved_lat], 'k--', alpha=0.5)
        ax3.set_xlim(-180, 180)
        ax3.set_ylim(-90, 90)
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title(f'Geolocation Error: {haversine_distance(true_lat, true_lon, solved_lat, solved_lon):.2f} km')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('dashboard.png')
        print("Saved dashboard.png")
        
    @staticmethod
    def plot_results(timestamps, measured_freqs, solved_lat, solved_lon, true_lat, true_lon, propagator, tx_freq):
        # Keep legacy method wrapper or just redirect
        # For now, we'll just keep it but the main script will call plot_dashboard
        pass

class LiveVisualizer:
    """
    Real-time visualization of SDR data.
    """
    def __init__(self, center_freq, sample_rate, history_size=100):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.history_size = history_size
        
        self.timestamps = []
        self.measured_freqs = []
        self.spectrogram_data = [] # List of 1D arrays
        
        plt.ion() # Enable interactive mode
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1])
        
        # Waterfall
        self.ax_waterfall = self.fig.add_subplot(gs[0])
        self.ax_waterfall.set_title(f"Real-Time Waterfall ({center_freq/1e6} MHz)")
        self.ax_waterfall.set_ylabel("Time (latest at bottom)")
        self.ax_waterfall.set_xlabel("Frequency Offset (kHz)")
        
        # Frequency Plot
        self.ax_freq = self.fig.add_subplot(gs[1])
        self.ax_freq.set_title("Measured Frequency")
        self.ax_freq.set_ylabel("Frequency (Hz)")
        self.ax_freq.set_xlabel("Time (s)")
        self.ax_freq.grid(True)
        
        self.line_freq, = self.ax_freq.plot([], [], 'b.-')
        
        # Placeholder for image
        self.im = None
        
        plt.show()
        plt.pause(0.1)

    def update(self, timestamp, measured_freq, spectrum_chunk):
        self.timestamps.append(timestamp)
        self.measured_freqs.append(measured_freq)
        self.spectrogram_data.append(spectrum_chunk)
        
        # Keep history limited
        if len(self.timestamps) > self.history_size:
            self.timestamps.pop(0)
            self.measured_freqs.pop(0)
            self.spectrogram_data.pop(0)
            
        # Update Frequency Plot
        # Use relative time for x-axis
        t0 = self.timestamps[0]
        rel_times = [(t - t0).total_seconds() for t in self.timestamps]
        
        self.line_freq.set_data(rel_times, self.measured_freqs)
        self.ax_freq.relim()
        self.ax_freq.autoscale_view()
        
        # Update Waterfall
        if len(self.spectrogram_data) > 0:
            spec_matrix = np.array(self.spectrogram_data)
            # Log scale
            spec_db = 10 * np.log10(np.abs(spec_matrix) + 1e-9)
            
            num_bins = spec_matrix.shape[1]
            freqs = np.fft.fftshift(np.fft.fftfreq(num_bins, d=1/self.sample_rate))
            extent = [freqs[0]/1000, freqs[-1]/1000, len(self.spectrogram_data), 0]
            
            if self.im is None:
                 self.im = self.ax_waterfall.imshow(spec_db, aspect='auto', extent=extent, cmap='inferno', origin='upper')
            else:
                 self.im.set_data(spec_db)
                 self.im.set_extent(extent)
                 self.im.set_clim(vmin=np.min(spec_db), vmax=np.max(spec_db))
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.ioff()
        plt.close(self.fig)

