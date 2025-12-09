# doppler_pkg/visualizer.py
import matplotlib.pyplot as plt
from skyfield.api import load
from .constants import C

class Visualizer:
    """
    Module E: Visualizer (Analysis)
    Responsibility: Plot the results.
    """
    @staticmethod
    def plot_results(timestamps, measured_freqs, solved_lat, solved_lon, true_lat, true_lon, propagator, tx_freq):
        ts = load.timescale()
        
        # --- Plot 1: The S-Curve ---
        plt.figure(figsize=(10, 6))
        plt.scatter(timestamps, measured_freqs, s=10, label='Noisy Measured Data', color='blue', alpha=0.5)
        
        # Calculate fitted curve
        fitted_freqs = []
        for t_dt in timestamps:
            t = ts.from_datetime(t_dt)
            v_rel = propagator.calculate_range_rate(t, solved_lat, solved_lon)
            f_theo = tx_freq * (1 - v_rel / C)
            fitted_freqs.append(f_theo)
            
        plt.plot(timestamps, fitted_freqs, label=f'Solved Fit (Lat:{solved_lat:.2f}, Lon:{solved_lon:.2f})', color='red', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Doppler Curve: Measured vs Solved')
        plt.legend()
        plt.grid(True)
        plt.savefig('doppler_curve.png')
        print("Saved doppler_curve.png")

        # --- Plot 2: The Map ---
        plt.figure(figsize=(8, 6))
        plt.scatter(true_lon, true_lat, color='green', marker='*', s=200, label='True Location')
        plt.scatter(solved_lon, solved_lat, color='red', marker='x', s=100, label='Estimated Location')
        
        # Draw line
        plt.plot([true_lon, solved_lon], [true_lat, solved_lat], 'k--', alpha=0.5)
        
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geolocation Result')
        plt.legend()
        plt.grid(True)
        plt.savefig('geolocation_map.png')
        print("Saved geolocation_map.png")
        # plt.show() # Non-interactive backend might fail
