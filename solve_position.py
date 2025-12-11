import pandas as pd
import numpy as np
from doppler_pkg.position_engine import PositionEngine

def main():
                  
    try:
        df = pd.read_csv("leo_s9_results.csv")
    except FileNotFoundError:
        print("Error: leo_s9_results.csv not found. Run simulation first.")
        return

    print(f"Loaded {len(df)} observations.")
    
                          
    engine = PositionEngine()
    
                             
    epochs = df['time'].unique()
    epochs.sort()
    
    results = []
    
                                     
    TRUE_LAT = 19.0760
    TRUE_LON = 72.8777
                     
    RE = 6378137.0
    lat_rad = np.radians(TRUE_LAT)
    lon_rad = np.radians(TRUE_LON)
    true_x = RE * np.cos(lat_rad) * np.cos(lon_rad)
    true_y = RE * np.cos(lat_rad) * np.sin(lon_rad)
    true_z = RE * np.sin(lat_rad)
    true_pos = np.array([true_x, true_y, true_z])
    
    print(f"True Position (ECEF): {true_pos}")
    
    for t in epochs:
        epoch_data = df[df['time'] == t]
        
        observations = []
        for _, row in epoch_data.iterrows():
            observations.append({
                'sat_x': row['sat_x'],
                'sat_y': row['sat_y'],
                'sat_z': row['sat_z'],
                'pseudorange': row['pseudorange']
            })
            
                                                             
                                                                                          
                                                                                          
        
                                          
        est_pos, bias, dop, res = engine.solve_epoch(observations, prior_pos=None)
        
        if est_pos is not None:
                             
            error_vec = est_pos - true_pos
            error_3d = np.linalg.norm(error_vec)
            
            results.append({
                "time": t,
                "error_3d": error_3d,
                "pdop": dop['PDOP'] if dop else np.nan,
                "sats": len(observations)
            })
            
            if t % 10 == 0:
                print(f"Time {t}: Error={error_3d:.2f}m, PDOP={dop['PDOP']:.2f}, Sats={len(observations)}")
        else:
            print(f"Time {t}: Failed to solve (Not enough sats?)")

                
    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Positioning Performance ---")
        print(f"Mean 3D Error: {res_df['error_3d'].mean():.2f} m")
        print(f"RMS 3D Error:  {np.sqrt((res_df['error_3d']**2).mean()):.2f} m")
        print(f"Mean PDOP:     {res_df['pdop'].mean():.2f}")
        
        res_df.to_csv("positioning_solution.csv", index=False)
        print("Solution saved to positioning_solution.csv")

if __name__ == "__main__":
    main()
