# Nexus

Nexus is a JAX-accelerated multi-scale simulator for biological control and causal discovery, capable of generating biologically-realistic data.

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

- `src/nexus/`
  - `simulator/` - Core simulation engine
    - `dynamics/` - Mathematical models for simulator dynamics
    - `noise_models/` - Noise processes, random seeds
    - `utils/` - Utility functions for simulator
    - `grn/` - Code and logs of the GRN simulator
    - `spatial/` -  Code, model definitions and logs of the Spatial Sim
    - `spatial_vec/` -  Code, model definitions and logs of the vectorized version of Spatial Sim
    - `run_sim.py` - Run the GRN and Spatial Sim
    - `run_linked_sim.py` - Run the GRN and Spatial Sim together, allowing diffusion between cells and fields
  - `causal/` - Causal discovery and inference modules
    - `identification/` - Algorithms for structure/parameter learning
    - `evaluation/` - Evaluation scripts
  - `experiments/` - Scripts for running experiments
  - `dashboard/` - Plotting, analysis, dashboards
    - `web/` - Next.JS website for visualization of simulator outputs
    - `retrieve_logger_api.py` - Flask API to serve data to the website
  - `tests/` - Unit & integration tests
- `configs/` - Config files for runs
- `notebooks/` - Exploratory analyses and demos
- `docs/` - Documentation
  - `config_desc.md` - Configuration parameters and their description
  - `intervention_api.md` - Working of intervention API and its configuration
  - `simulator.md` - Working of the GRN simulator
  - `spatial_simulator.md` - Working of the spatial simulator
- `examples/` - Examples scripts
  - `marimo_example.py` - Marimo notebook containing example scripts
  - `ipynb_example.py` - IPYNB notebook containing example scripts
- `pyproject.toml` - For package/dependency management
- `README.md`

## Usage

A short code snippet for using the simulator is given below.
```py
from nexus.simulator.grn.grnSim import GRNSim
from nexus.simulator.spatial.spatialSim import SpatialSim
from hydra import initialize_config_dir, compose

 def get_config(config_name="test_config"): # Get config from Hydra config file
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg

cfg = get_config()

sim = GRNSim(cfg.grn)
sim.run_sim()

spatial_sim = SpatialSim(cfg.spatial)
spatial_sim.run_sim()
```

Examples for running the simulator are given in a [Marimo notebook](examples/marimo_example.py) and a [IPYNB notebook](examples/ipynb_example.ipynb). The simulator uses Hydra configs for the simulation parameters and the description of the config files are given in [Config Description](docs/config_desc.md).

## Visualization

The visualization can be run by the running the two commands in seperate terminal windows.

```bash
## Runs the API for fetching the data from TileDB
uv run src/dashboard/retrieve_logger_api.py

## Runs the website to visualize the data
cd dashboard/web/
npm install # If the packages are not installed
npm run dev
```

After running these two commands, open the link `http://localhost:3000` and select a log file from the dropdown. This log file should be present inside the relative directory `logs/`.

Further instructions on using the dashboard is given in [README](dashboard/web/README.md)