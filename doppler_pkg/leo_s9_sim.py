import numpy as np
import pandas as pd
from skyfield.api import load, wgs84
from .constellation import LEOS9Constellation, StarlinkConstellation
from .orbit_propagator import NumericalPropagator
from .error_models import ClockErrorModel, ClockType, AtmosphericErrorModel, MeasurementNoiseModel
from .constants import C, TX_FREQUENCY

class LEOS9Simulator:
    def __init__(self, duration_sec=60, step_sec=1.0):
        self.duration = duration_sec
        self.step = step_sec
        
        # Initialize Constellation
        print("Initializing Constellation (Starlink Shell 1)...")
        self.constellation = StarlinkConstellation()
        self.sats_elements = self.constellation.generate_all()
        print(f"Generated {len(self.sats_elements)} satellites.")
        
        # Initialize Propagators (One per satellite)
        # WARNING: 441 propagators might be slow.
        # For demo, maybe limit to visible ones?
        # But we don't know which are visible until we propagate.
        # Let's propagate ALL for now, but maybe just a subset for testing if it's too slow.
        self.propagators = [NumericalPropagator(el) for el in self.sats_elements]
        
        # Initialize Error Models
        self.clocks = [ClockErrorModel(ClockType.OCXO) for _ in range(len(self.sats_elements))]
        self.atm_model = AtmosphericErrorModel()
        self.noise_model = MeasurementNoiseModel()
        
        # User Location (Static for now)
        self.user_lat = 19.0760
        self.user_lon = 72.8777
        self.user_alt = 0.0
        self.user_pos_ecef = self._get_user_ecef()

    def _get_user_ecef(self):
        # Simple conversion
        # x = (N+h) cos lat cos lon
        # y = (N+h) cos lat sin lon
        # z = (N(1-e^2)+h) sin lat
        # Approximation: spherical
        RE = 6378137.0
        lat = np.radians(self.user_lat)
        lon = np.radians(self.user_lon)
        
        x = RE * np.cos(lat) * np.cos(lon)
        y = RE * np.cos(lat) * np.sin(lon)
        z = RE * np.sin(lat)
        return np.array([x, y, z])

    def run(self):
        print(f"Starting Simulation for {self.duration} seconds...")
        results = []
        
        # Time loop
        for t_sec in range(0, self.duration, int(self.step)):
            if t_sec % 10 == 0:
                print(f"Time: {t_sec}/{self.duration}")
                
            # Propagate all satellites
            for i, prop in enumerate(self.propagators):
                prop.step(self.step)
                
                # Get ECI State
                r_eci = prop.state[:3]
                v_eci = prop.state[3:]
                
                # Convert to ECEF
                # We need the rotation matrix from ECI (GCRS) to ECEF (ITRS) at time t
                # Skyfield can do this.
                ts = load.timescale()
                t_skyfield = ts.tt_jd(prop.t_current_jd)
                
                # Rotation matrix GCRS -> ITRS
                # r_ecef = R * r_eci
                # R = t_skyfield.r  <-- REMOVED

                # Skyfield's .r is the rotation matrix? No, .M is usually the matrix.
                # Actually, let's use skyfield's position object to be safe.
                # But we have raw numpy arrays.
                # t.gast is Greenwich Apparent Sidereal Time?
                # Let's use a simple rotation for now if we want speed, or skyfield for accuracy.
                # Skyfield:
                # p = positionlib.Geocentric(r_eci, t=t_skyfield)
                # p_itrs = p.frame_latlon(itrs) ...
                
                # Faster: Earth rotation angle theta = w * t
                # This ignores nutation/precession but is consistent with our simple propagator?
                # No, our propagator uses J2000 ECI.
                # We should use Skyfield to get the rotation matrix properly.
                
                # t.ut1 is needed for earth rotation.
                # We can approximate UT1 = UTC for simulation.
                
                # Get rotation matrix from GCRS to ITRS
                # Skyfield doesn't expose a simple 3x3 matrix easily without computing nutation etc.
                # But we can use `t.M` which is the rotation matrix?
                # t.M is for precession/nutation?
                
                # Let's use the `era` (Earth Rotation Angle) from Skyfield functions if possible.
                # Or just use the standard ERA formula: theta = 2pi * (0.7790572732640 + 1.00273781191135448 * Tu)
                # Tu = (Julian Date - 2451545.0)
                
                # Correction: TT is ahead of UT1 by ~69.184s
                delta_t_days = 69.184 / 86400.0
                jd_ut1 = prop.t_current_jd - delta_t_days
                Tu = jd_ut1 - 2451545.0
                theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * Tu)
                theta = theta % (2 * np.pi)
                
                c = np.cos(theta)
                s = np.sin(theta)
                R_z = np.array([
                    [c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]
                ])
                
                sat_pos_ecef = R_z @ r_eci
                
                # Velocity conversion (transport theorem)
                # v_ecef = R * v_eci - cross(omega, r_ecef)
                omega_e = np.array([0, 0, 7.2921159e-5])
                sat_vel_ecef = R_z @ v_eci - np.cross(omega_e, sat_pos_ecef)
                
                # Use ECEF for geometry
                sat_pos = sat_pos_ecef
                sat_vel = sat_vel_ecef
                
                el, az, dist = self.atm_model._get_el_az_dist(sat_pos, self.user_pos_ecef)
                
                if el > 10.0: # 10 degree mask
                    # Calculate Observables
                    
                    # 1. Geometric Range
                    geom_range = dist
                    
                    # 2. Clock Error
                    # clk_err = self.clocks[i].get_error(t_sec) * C
                    clk_err = 0.0
                    
                    # 3. Atmospheric Delays
                    iono = self.atm_model.get_ionosphere_delay(sat_pos, self.user_pos_ecef, t_sec)
                    tropo = self.atm_model.get_troposphere_delay(sat_pos, self.user_pos_ecef)
                    
                    # 4. Relativistic (Sagnac + Shapiro)
                    # Sagnac: omega_e * (xs*yu - ys*xu) / c
                    sagnac = omega_e[2] * (sat_pos[0]*self.user_pos_ecef[1] - sat_pos[1]*self.user_pos_ecef[0]) / C
                    
                    # Shapiro (approx)
                    shapiro = 0.0
                    
                    # 5. Doppler
                    # Range Rate
                    # v_rel projected on LOS
                    u_los = (sat_pos - self.user_pos_ecef) / dist
                    range_rate = np.dot(sat_vel, u_los)
                    
                    # Doppler Shift: f_obs = f_tx * (1 - range_rate/c)
                    doppler = - (range_rate / C) * TX_FREQUENCY
                    
                    # Total Pseudorange
                    pr_true = geom_range + clk_err + iono + tropo + sagnac + shapiro
                    
                    # Add Noise
                    pr_meas, ph_meas = self.noise_model.add_noise(pr_true)
                    
                    results.append({
                        "time": t_sec,
                        "sat_id": self.sats_elements[i].sat_id,
                        "az": az,
                        "el": el,
                        "pseudorange": pr_meas,
                        "carrier_phase": ph_meas, # Ambiguity not modeled yet
                        "doppler": doppler,
                        "true_range": geom_range,
                        "sat_x": sat_pos[0],
                        "sat_y": sat_pos[1],
                        "sat_z": sat_pos[2],
                        "sat_vx": sat_vel[0],
                        "sat_vy": sat_vel[1],
                        "sat_vz": sat_vel[2]
                    })
                    
        return pd.DataFrame(results)

if __name__ == "__main__":
    sim = LEOS9Simulator(duration_sec=60)
    df = sim.run()
    print(df.head())
    df.to_csv("simulation_results.csv", index=False)
