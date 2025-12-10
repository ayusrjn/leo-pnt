import numpy as np
from .constants import C

class PositionEngine:
    """
    Solves for user position using Pseudorange measurements.
    """
    def __init__(self):
        self.max_iter = 10
        self.convergence_thresh = 1e-3 # meters

    def solve_epoch(self, observations: list, prior_pos=None):
        """
        Solve for (x, y, z, bias) for a single epoch.
        
        Args:
            observations: List of dicts with keys:
                'sat_x', 'sat_y', 'sat_z', 'pseudorange'
            prior_pos: Initial guess [x, y, z]. If None, uses (0,0,0).
            
        Returns:
            est_pos: [x, y, z] (meters)
            clock_bias: Receiver clock bias (meters)
            dop: Dilution of Precision (GDOP, PDOP, HDOP, VDOP)
            residuals: Final measurement residuals
        """
        if len(observations) < 4:
            # Need at least 4 satellites for 3D pos + time
            return None, None, None, None
            
        # Initial State: [x, y, z, b]
        if prior_pos is not None:
            state = np.array([prior_pos[0], prior_pos[1], prior_pos[2], 0.0])
        else:
            # Cold Start Strategy: Centroid of satellites
            # This puts us roughly in the right part of the world
            sx_list = [o['sat_x'] for o in observations]
            sy_list = [o['sat_y'] for o in observations]
            sz_list = [o['sat_z'] for o in observations]
            
            avg_x = np.mean(sx_list)
            avg_y = np.mean(sy_list)
            avg_z = np.mean(sz_list)
            
            # Project to Earth Surface
            r = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
            RE = 6378137.0
            scale = RE / r
            
            state = np.array([avg_x * scale, avg_y * scale, avg_z * scale, 0.0])
        
        for i in range(self.max_iter):
            # Formulate H matrix and y vector (pre-fit residuals)
            H = []
            y = []
            
            rx, ry, rz, b = state
            
            for obs in observations:
                sx, sy, sz = obs['sat_x'], obs['sat_y'], obs['sat_z']
                pr_meas = obs['pseudorange']
                
                # Geometric range
                dist = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)
                
                # Predicted Pseudorange
                pr_pred = dist + b
                
                # Residual (Measured - Predicted)
                # Note: Convention varies. Let's use dy = Meas - Pred
                dy = pr_meas - pr_pred
                y.append(dy)
                
                # Partial derivatives (Design Matrix row)
                # d(pr)/dx = - (sx - rx) / dist
                dr_dx = -(sx - rx) / dist
                dr_dy = -(sy - ry) / dist
                dr_dz = -(sz - rz) / dist
                dr_db = 1.0
                
                H.append([dr_dx, dr_dy, dr_dz, dr_db])
                
            H = np.array(H)
            y = np.array(y)
            
            # Least Squares Solution
            # dx = (H^T H)^-1 H^T y
            try:
                # Normal equations
                HtH = H.T @ H
                HtH_inv = np.linalg.inv(HtH)
                dx = HtH_inv @ H.T @ y
                
                state += dx
                
                if np.linalg.norm(dx[:3]) < self.convergence_thresh:
                    break
                    
            except np.linalg.LinAlgError:
                print("Singular matrix in Position Solver.")
                return None, None, None, None
                
        # Compute DOPs
        # Q = (H^T H)^-1
        # GDOP = sqrt(trace(Q))
        # PDOP = sqrt(Qxx + Qyy + Qzz)
        try:
            Q = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(Q))
            pdop = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
            tdop = np.sqrt(Q[3,3])
            
            # For HDOP/VDOP need rotation to ENU, skipping for now
            dop = {"GDOP": gdop, "PDOP": pdop, "TDOP": tdop}
            
        except:
            dop = None
            
        return state[:3], state[3], dop, y
