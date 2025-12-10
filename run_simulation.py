import sys
import os

# Add current directory to path so we can import doppler_pkg
sys.path.append(os.getcwd())

from doppler_pkg.leo_s9_sim import LEOS9Simulator
import time

def main():
    print("--- LEO-S9 Simulation Runner ---")
    
    # Run for a short duration to test
    duration = 60 # seconds
    step = 1.0
    
    start_time = time.time()
    sim = LEOS9Simulator(duration_sec=duration, step_sec=step)
    
    print(f"Initialization took {time.time() - start_time:.2f} seconds.")
    
    run_start = time.time()
    df = sim.run()
    run_time = time.time() - run_start
    
    print(f"Simulation completed in {run_time:.2f} seconds.")
    print(f"Generated {len(df)} observations.")
    
    output_file = "leo_s9_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Basic Analysis
    if not df.empty:
        print("\n--- Statistics ---")
        print(f"Unique Satellites Visible: {df['sat_id'].nunique()}")
        print(f"Avg Satellites per Epoch: {len(df) / duration:.2f}")
        print(f"Min Elevation: {df['el'].min():.2f}")
        print(f"Max Elevation: {df['el'].max():.2f}")

if __name__ == "__main__":
    main()
