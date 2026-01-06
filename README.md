# CORTEX

## Causal Omics Reasoning in Temporal Experiments

## Table of contents

- [Setup](#setup)
- [Description](#description)
- [Usage](#usage)
  - [Simulator](#simulator)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager before running the setup

```bash
# Clone repository
git clone https://github.com/AmoghSingh25/GeneProtSim.git

# Create venv and download packages
uv sync
```

## Code Structure

- `src/`
  - `simulator/` - Core simulation engine
    - `dynamics/` - Mathematical models for simulator dynamics
    - `noise_models/` - Noise processes, random seeds
    - `utils/` - Utility functions for simulator
    - `dynSim.py` - Simulator code
  - `causal/` - Causal discovery and inference modules
    - `identification/` - Algorithms for structure/parameter learning
    - `evaluation/` - Evaluation scripts
  - `experiments/` - Scripts for running experiments
  - `visualization/` - Plotting, analysis, dashboards
  - `tests/` - Unit & integration tests
- `configs/` - Config files for runs
- `notebooks/` - Exploratory analyses and demos
  - `testing.py` - Notebook to check working of GeneProtSim
- `pyproject.toml` - For package/dependency management
- `README.md`

## Usage

### Simulator

```py
## Gene-Protein Sim
uv run src/simulator/run_gpsim.py

## Spatial Sim
uv run src/simulator/spatial/main.py
```
### Visualization

Currently the visualization server, `src/visualization/testing_visualization.py` only supports GridMesh data visualization. To run the visualization, pass the log file name of a previously run simulation run with the `-f` flag. A sample log file is provided under `src/simulator/spatial/logs/1767656604` and the server defaults to using the sample log file.

Performing a simulation run and running the visualization

```py
# From DynamicSim/
uv run src/simulator/spatial/main.py # Runs the spatial sim with default `test_config.yaml` configuration

uv run src/visualization/testing_visualization.py -f FILE_NAME # FILE_NAME is the timestamp under src/simulator/spatial/logs/
```