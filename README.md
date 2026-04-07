# CORTEX

## Causal Omics Reasoning in Temporal Experiments

## Table of contents

- [Setup](#setup)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Visualization](#visualization)

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

Examples for running the simulator are given in a [Marimo notebook](examples/marimo_example.py) and a [IPYNB notebook](examples/ipynb_example.ipynb). The simulator uses Hydra configs for the simulation parameters and the description of the config files are given in [Config Description](docs/config_desc.md).

## Visualization

The visualization can be run by the running the two commands in seperate terminal windows.

```bash
## Runs the API for fetching the data from TileDB
uv run src/nexus_sim/dashboad/retrieve_logger_api.py

## Runs the website to visualize the data
cd src/nexus_sim/dashboard/web/
npm install # If the packages are not installed
npm run dev
```

After running these two commands, open the link `http://localhost:3000` and selecting a log file from the dropdown. This log file should be present inside `src/simulator/grn/logs/` and `src/simulator/spatial_vec/logs/`.

The colors indicate the states of the cell, yellow indicating live cells, blue indicating cells undergoing programmed cell death and red are the cells undergoing sudden cell death.

Further instructions on using the dashboard is given in [README](src/nexus_sim/dashboard/web/README.md)