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
# Import simulator
from simulator import gpsim

# Initialize
sim = gpsim.simulator(
  config_file="configs/sample_data/sample_network_2cell.yaml", n_cells=2
)

# Run for 10 steps and return the progression of gene and protein concentration
gene_conc, prot_conc = sim.run_sim(n_steps=10)
```
