# doppler_pkg/satellite_identifier.py

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from .constellation import StarlinkConstellation, KeplerianElements
from .orbit_propagator import NumericalPropagator
from .error_models import AtmosphericErrorModel
from .constants import C, TX_FREQUENCY

class SatelliteIdentifier:
    def __init__(self, user_lat, user_lon, user_alt=0.0):
        self.user_lat = user_lat
        self.user_lon = user_lon
        self.user_alt = user_alt
        self.user_pos_ecef = self._get_user_ecef()

        self.constellation = StarlinkConstellation()
        self.sats_elements = self.constellation.generate_all()
        
        # Initialize one propagator per satellite for independent stepping
        self.propagators = {el.sat_id: NumericalPropagator(el) for el in self.sats_elements}
        
        self.atm_model = AtmosphericErrorModel()

    def _get_user_ecef(self):
        # Simple ECEF conversion (can be moved to a utility if needed elsewhere)
        RE = 6378137.0
        lat = np.radians(self.user_lat)
        lon = np.radians(self.user_lon)
        
        x = RE * np.cos(lat) * np.cos(lon)
        y = RE * np.cos(lat) * np.sin(lon)
        z = RE * np.sin(lat)
        return np.array([x, y, z])

    def predict_all_dopplers(self, current_jd, step_sec=1.0):
        """
        Predicts Doppler shifts for all visible satellites at a given Julian Date.
        Returns a dictionary of {sat_id: predicted_doppler}.
        """
        predicted_dopplers = {}
        ts = load.timescale()
        t_skyfield = ts.tt_jd(current_jd) # Need TT for Skyfield
        
        # Approximate UT1 = UTC for simulation.
        # We need UT1 for Earth rotation angle.
        delta_t_days = 69.184 / 86400.0 # From leo_s9_sim.py
        jd_ut1 = current_jd - delta_t_days
        Tu = jd_ut1 - 2451545.0
        theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * Tu)
        theta = theta % (2 * np.pi)
        
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        R_z = np.array([
            [c_theta, s_theta, 0],
            [-s_theta, c_theta, 0],
            [0, 0, 1]
        ])
        
        omega_e = np.array([0, 0, 7.2921159e-5]) # Earth rotation rate vector

        for sat_id, prop in self.propagators.items():
            # Step propagator to current JD
            # NOTE: This will advance the internal state of the propagator.
            # If we need to predict without advancing, we'd need a copy or a different method.
            # For this use case, we assume we want to step it forward to the current time.
            
            # Reset and step, or step from current internal time?
            # For identification, we need to know state *at* current_jd.
            # Let's assume prop.step(dt) brings it to current_jd for now.
            # A better approach would be to have a predict_state_at_jd method in NumericalPropagator
            
            # For now, let's just get the state directly from the propagator at current_jd
            # This requires NumericalPropagator to have a function to get state at a specific JD
            # rather than just stepping.
            # Let's assume NumericalPropagator.get_state_at_jd(jd) exists or add it.
            
            # For simplicity, and aligning with how LEOS9Simulator works,
            # we'll use prop.state and assume it's been propagated externally to `current_jd`
            # or that `prop.step` updates `prop.t_current_jd` appropriately if called with `dt`.
            # However, for a one-shot prediction, it needs to be `at` current_jd.
            
            # Let's temporarily assume `NumericalPropagator.propagate_to_jd(jd)` exists.
            # Or, we can re-initialize a propagator for each prediction, which is inefficient.
            # The LEOS9Simulator advances all propagators in a loop. Here, we want one-shot.

            # Re-reading NumericalPropagator: it steps. It doesn't have a direct 'get state at JD'.
            # This means `NumericalPropagator` itself is not ideal for this one-shot prediction.
            # We need a way to get a satellite's state at a specific JD without modifying its internal state.

            # Alternative: Use skyfield directly or modify NumericalPropagator to allow this.
            # For now, I will modify NumericalPropagator to have a `get_state_at_jd` method.

            # For now, let's simulate the ECI position and velocity at current_jd
            # This requires a new method in NumericalPropagator or direct skyfield use.
            # Let's consider a simpler approach first, using the propagation as done in LEOS9Simulator.

            # To avoid modifying propagator state for each prediction,
            # we can create a temporary propagator or use the original TLE.
            # Or, we make sure the input `prop` is already at `current_jd`.

            # Let's assume for now that we have a way to get (r_eci, v_eci) at current_jd for each sat_id.
            # This would likely involve using skyfield or modifying NumericalPropagator.
            # For this first pass, let's focus on the Doppler calculation part.

            # For the purpose of getting ECI state at current_jd,
            # we will create a dummy NumericalPropagator to avoid state changes.
            temp_prop = NumericalPropagator(self.sats_elements[sat_id]) # THIS IS WRONG, needs sat_id to KeplerianElements object mapping.
            # The self.propagators dict maps sat_id to the already initialized NumericalPropagator.
            # But we need to *propagate it to current_jd* without modifying its current state if it's reused.

            # Let's re-read NumericalPropagator again to be sure.
            # It has `step(dt)`. It updates `self.t_current_jd` and `self.state`.
            # So, we cannot just use `self.propagators[sat_id].state` directly unless we are sure it's already at `current_jd`.

            # A more robust solution: modify NumericalPropagator to predict without state change.
            # For now, let's keep it simple and assume a way to get r_eci, v_eci at `current_jd`.
            # We need to simulate the propagation for each satellite at the `current_jd`.
            # The NumericalPropagator init takes a KeplerianElements.
            # We need to map `sat_id` back to `KeplerianElements` to create a fresh propagator for prediction.
            
            # Map sat_id back to KeplerianElements object
            keplerian_element = next((el for el in self.sats_elements if el.sat_id == sat_id), None)
            if keplerian_element is None:
                continue # Should not happen

            # Create a temporary propagator for prediction at current_jd
            # This is inefficient but avoids state management issues for now.
            temp_prop = NumericalPropagator(keplerian_element)
            
            # Propagate from its epoch to current_jd
            dt_seconds = (current_jd - temp_prop.t_current_jd) * 86400.0
            temp_prop.step(dt_seconds) # Propagate to current_jd

            r_eci = temp_prop.state[:3]
            v_eci = temp_prop.state[3:]
            
            # Convert to ECEF
            sat_pos_ecef = R_z @ r_eci
            sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
            
            el, az, dist = self.atm_model._get_el_az_dist(sat_pos_ecef, self.user_pos_ecef)
            
            if el > 10.0: # 10 degree mask for visibility
                u_los = (sat_pos_ecef - self.user_pos_ecef) / dist
                range_rate = np.dot(sat_vel_ecef, u_los)
                doppler = - (range_rate / C) * TX_FREQUENCY
                predicted_dopplers[sat_id] = doppler
                
        return predicted_dopplers

    def identify_satellite(self, observed_doppler, current_jd, tolerance=100.0):
        """
        Identifies the most likely satellite based on observed Doppler shift.
        `tolerance` is the maximum allowed difference in Hz.
        """
        predicted_dopplers = self.predict_all_dopplers(current_jd)
        
        if not predicted_dopplers:
            return None, float('inf') # No visible satellites
            
        best_sat_id = None
        min_diff = float('inf')
        
        for sat_id, pred_doppler in predicted_dopplers.items():
            diff = abs(observed_doppler - pred_doppler)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                best_sat_id = sat_id
                
        return best_sat_id, min_diff

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example: User at some location, current time
    user_lat_ex = 19.0760
    user_lon_ex = 72.8777
    current_time_ex = load.utc(2023, 10, 26, 12, 0, 0) # Example UTC time
    current_jd_ex = current_time_ex.tt
    
    identifier = SatelliteIdentifier(user_lat_ex, user_lon_ex)
    
    print(f"Predicting Dopplers at {current_time_ex.utc_iso()}...")
    predicted = identifier.predict_all_dopplers(current_jd_ex)
    
    if predicted:
        print("Predicted Dopplers for visible satellites:")
        for sat_id, doppler_val in predicted.items():
            print(f"  Sat ID {sat_id}: {doppler_val:.2f} Hz")
            
        # Simulate an observed Doppler
        # Let's say we observe a Doppler close to one of the predicted ones
        # Pick one at random for demonstration
        random_sat_id = list(predicted.keys())[0]
        true_doppler = predicted[random_sat_id]
        observed_doppler_sim = true_doppler + np.random.normal(0, 50) # Add some noise
        
        print(f"\nSimulated Observed Doppler: {observed_doppler_sim:.2f} Hz (True from Sat ID {random_sat_id})")
        
        identified_sat_id, diff = identifier.identify_satellite(observed_doppler_sim, current_jd_ex)
        
        if identified_sat_id is not None:
            print(f"Identified Satellite ID: {identified_sat_id} with difference {diff:.2f} Hz")
            print(f"Actual Sat ID was: {random_sat_id}")
        else:
            print("No satellite identified within tolerance.")
    else:
        print("No satellites visible at this time.")

    # Another test case: no satellites visible
    current_time_no_vis = load.utc(2023, 1, 1, 0, 0, 0)
    current_jd_no_vis = current_time_no_vis.tt
    print(f"\nPredicting Dopplers at {current_time_no_vis.utc_iso()} (expecting none visible)...")
    predicted_no_vis = identifier.predict_all_dopplers(current_jd_no_vis)
    print(f"Predicted Dopplers: {predicted_no_vis}")
    
    identified_sat_id_no_vis, diff_no_vis = identifier.identify_satellite(100.0, current_jd_no_vis)
    print(f"Identified Satellite ID: {identified_sat_id_no_vis}")
