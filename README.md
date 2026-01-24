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
    - `grn/` - Code and logs of the GRN simulator
    - `spatial/` -  Code, model definitions and logs of the Spatial Sim
    - `spatial_vec/` -  Code, model definitions and logs of the vectorized version of Spatial Sim
    - `run_sim.py` - Run the GRN and Spatial Sim
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

```bash
## Runs the GRN Sim and the spatial sim with the default config 'freemesh_config.yaml'
uv run src/simulator/run_sim.py
```
### Visualization

The visualization can be run by the running the two commands in seperate terminal windows.

```bash
## Runs the API for fetching the data from TileDB
uv run src/visualization/retrieve_logger_api.py

## Runs the website to visualize the data
cd src/visualization/visualization_web/
npm install # If the packages are not installed
npm run dev
```

After running these two commands, open the link `http://localhost:3000/vis?file_name={FILE_NAME}` and inserting the name of the log file to be visualized at `{FILE_NAME}`. This log file should be present inside `src/simulator/grn/logs/` and `src/simulator/spatial_vec/logs/`.