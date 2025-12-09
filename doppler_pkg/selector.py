# doppler_pkg/selector.py
import numpy as np
from .solver import GeolocationSolver
from .propagator import SatellitePropagator

class SatelliteSelector:
    """
    Module: SatelliteSelector
    Responsibility: Identify the correct satellite from a list of candidates.
    """
    def __init__(self):
        pass

    def identify_satellite(self, candidates: list, timestamps: list, measured_freqs: list):
        """
        Iterate through candidates, solve for location, and find the one with minimum residual cost.
        
        Args:
            candidates: List of Skyfield Satellite objects.
            timestamps: List of timestamps.
            measured_freqs: List of measured frequencies.
            
        Returns:
            best_sat: The identified Satellite object.
            best_location: (lat, lon) tuple of the estimated location.
            min_cost: The minimum residual cost found.
        """
        best_sat = None
        best_location = (0.0, 0.0)
        min_cost = float('inf')
        
        print(f"Scanning {len(candidates)} candidates...")
        
        for i, sat in enumerate(candidates):
            try:
                propagator = SatellitePropagator(sat)
                solver = GeolocationSolver(propagator)
                
                # Solve and get cost
                lat, lon, cost = solver.solve(timestamps, measured_freqs)
                
                # print(f"[{i+1}/{len(candidates)}] {sat.name}: Cost={cost:.2f}, Loc=({lat:.2f}, {lon:.2f})")
                
                if cost < min_cost:
                    min_cost = cost
                    best_sat = sat
                    best_location = (lat, lon)
                    
            except Exception as e:
                # Satellite might be below horizon or other math error
                # print(f"[{i+1}/{len(candidates)}] {sat.name}: Failed ({e})")
                continue
                
        return best_sat, best_location, min_cost
