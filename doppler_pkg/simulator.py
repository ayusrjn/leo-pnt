                          
import datetime
import numpy as np
from skyfield.api import load
from .constants import C, TX_FREQUENCY
from .propagator import SatellitePropagator

class SignalSimulator:
    """
    Module C: SignalSimulator (The Digital Twin)
    Responsibility: Generate synthetic 'Measured' data.
    """
    def __init__(self, propagator: SatellitePropagator, tx_freq: float = TX_FREQUENCY):
        self.propagator = propagator
        self.tx_freq = tx_freq

    def generate_data(self, true_lat: float, true_lon: float, start_time: datetime.datetime, duration_sec: int, noise_std: float):
        """
        Iterate second-by-second over the pass duration.
        Calculate exact theoretical frequency.
        Add Gaussian noise.
        """
        ts = load.timescale()
        timestamps = []
        noisy_freqs = []
        
        print(f"Simulating signal for Lat: {true_lat}, Lon: {true_lon}...")
        
        for i in range(duration_sec):
            curr_time_dt = start_time + datetime.timedelta(seconds=i)
            t = ts.from_datetime(curr_time_dt)
            
                                             
                                            
            v_rel = self.propagator.calculate_range_rate(t, true_lat, true_lon)
            f_theoretical = self.tx_freq * (1 - v_rel / C)
            
                       
            noise = np.random.normal(0, noise_std)
            f_measured = f_theoretical + noise
            
            timestamps.append(curr_time_dt)
            noisy_freqs.append(f_measured)
            
        return timestamps, noisy_freqs
