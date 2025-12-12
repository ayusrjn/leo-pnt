# LEO-PNT Simulation & Positioning Stack

This project implements a high-fidelity simulation of a LEO-PNT constellation (Walker Delta 441 satellites) and a software-defined receiver positioning engine.

## Prerequisites

- Python 3.10+
- Dependencies: `numpy`, `pandas`, `scipy`, `skyfield`, `pyqt5`, `pyqtgraph`, `pyzmq`

```bash
pip install numpy pandas scipy skyfield PyQt5 pyqtgraph pyzmq
```

## 1. Run Simulation (Batch Mode)

To generate synthetic satellite data and solve for position in batch mode:

```bash
# 1. Run the simulation (Generates leo_s9_results.csv)
python3.10 run_simulation.py

# 2. Run the Position Solver (Reads CSV, outputs positioning_solution.csv)
python3.10 solve_position.py
```

## 2. Run Visualization (Real-Time Replay)

To visualize the satellite selection, Doppler spectrum, and position convergence in real-time:

**Terminal 1 (Display Node):**
Starts the GUI dashboard.
```bash
python3.10 display_node.py
```

**Terminal 2 (Replay Node):**
Streams the simulation data to the dashboard.
```bash
python3.10 replay_simulation.py
```

**To run Simulation & Visualization:**

```bash
cd viz
python3.10 -m http-server
```

## Project Structure

- `doppler_pkg/`: Core package
    - `constellation.py`: Walker Delta constellation generator.
    - `orbit_propagator.py`: Numerical propagator (J2, Drag, SRP).
    - `leo_s9_sim.py`: Main simulation engine.
    - `position_engine.py`: Least Squares positioning solver.
    - `qt_visualizer.py`: PyQt5 dashboard widgets.
- `run_simulation.py`: Script to execute the simulation.
- `solve_position.py`: Script to calculate position from simulation results.
- `display_node.py`: ZMQ Subscriber + GUI.
- `replay_simulation.py`: ZMQ Publisher (Replay).
