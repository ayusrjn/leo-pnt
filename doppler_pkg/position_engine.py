import numpy as np
from .constants import C

class PositionEngine:
    """
    Solves for user position using Pseudorange measurements.
    """
    def __init__(self):
        self.max_iter = 10
        self.convergence_thresh = 1e-3         

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
                                                          
            return None, None, None, None
            
                                     
        if prior_pos is not None:
            state = np.array([prior_pos[0], prior_pos[1], prior_pos[2], 0.0])
        else:
                                                         
                                                                 
            sx_list = [o['sat_x'] for o in observations]
            sy_list = [o['sat_y'] for o in observations]
            sz_list = [o['sat_z'] for o in observations]
            
            avg_x = np.mean(sx_list)
            avg_y = np.mean(sy_list)
            avg_z = np.mean(sz_list)
            
                                      
            r = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
            RE = 6378137.0
            scale = RE / r
            
            state = np.array([avg_x * scale, avg_y * scale, avg_z * scale, 0.0])
        
        for i in range(self.max_iter):
                                                                 
            H = []
            y = []
            
            rx, ry, rz, b = state
            
            for obs in observations:
                sx, sy, sz = obs['sat_x'], obs['sat_y'], obs['sat_z']
                pr_meas = obs['pseudorange']
                
                                 
                dist = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)
                
                                       
                pr_pred = dist + b
                
                                                 
                                                                     
                dy = pr_meas - pr_pred
                y.append(dy)
                
                                                         
                                               
                dr_dx = -(sx - rx) / dist
                dr_dy = -(sy - ry) / dist
                dr_dz = -(sz - rz) / dist
                dr_db = 1.0
                
                H.append([dr_dx, dr_dy, dr_dz, dr_db])
                
            H = np.array(H)
            y = np.array(y)
            
                                    
                                   
            try:
                                  
                HtH = H.T @ H
                HtH_inv = np.linalg.inv(HtH)
                dx = HtH_inv @ H.T @ y
                
                state += dx
                
                if np.linalg.norm(dx[:3]) < self.convergence_thresh:
                    break
                    
            except np.linalg.LinAlgError:
                print("Singular matrix in Position Solver.")
                return None, None, None, None
                
                      
                        
                               
                                      
        try:
            Q = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(Q))
            pdop = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
            tdop = np.sqrt(Q[3,3])
            
                                                                  
            dop = {"GDOP": gdop, "PDOP": pdop, "TDOP": tdop}
            
        except:
            dop = None
            
        return state[:3], state[3], dop, y
