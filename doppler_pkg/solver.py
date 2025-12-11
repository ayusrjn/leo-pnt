                       
import numpy as np
from scipy.optimize import least_squares
from skyfield.api import load
from .constants import C, TX_FREQUENCY
from .propagator import SatellitePropagator

class GeolocationSolver:
    """
    Module D: GeolocationSolver (The Algorithm)
    Responsibility: Blindly estimate location using only TLE and noisy frequency data.
    """
    def __init__(self, propagator: SatellitePropagator, tx_freq: float = TX_FREQUENCY):
        self.propagator = propagator
        self.tx_freq = tx_freq
        self.ts = load.timescale()

    def solve(self, timestamps: list, measured_freqs: list) -> tuple:
        """
        Estimate Lat/Lon using least squares optimization.
        
        """
        print("Solving for location...")
        
                                                                         
        ts_objects = [self.ts.from_datetime(t) for t in timestamps]
        measured_freqs_np = np.array(measured_freqs)

        def _residuals(guess_lat_lon):
            lat, lon = guess_lat_lon
            theoretical_freqs = []
            
            for t in ts_objects:
                v_rel = self.propagator.calculate_range_rate(t, lat, lon)
                f_theo = self.tx_freq * (1 - v_rel / C)
                theoretical_freqs.append(f_theo)
            
            return measured_freqs_np - np.array(theoretical_freqs)

                               
        initial_guess = [0.0, 0.0]
        
                                                
        bounds = ([-90, -180], [90, 180])

        result = least_squares(_residuals, initial_guess, bounds=bounds, verbose=0)
        
        return result.x[0], result.x[1], result.cost
