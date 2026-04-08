# Configuration description

The config consists of two base-level keys, `grn` and `spatial_sim`. A sample config is given below with their descriptions.

```yaml
# Config for GRN set to simulate SERGIO configuration
defaults:
  - _self_
grn:
  ## Simulation related
  n_cells: 8 # Number of cells to simulate. Configuration must be given in config_file for each cell or a single cell which is copied to multiple cells.
  delta: 0.01 # Delta between successive timesteps in the simulation
  n_steps: 10 # Number of steps
  config_file: "configs/sample_data/sample_network_1cell.yaml" # Path to the config file. Can be a .yaml consisting of RNA and protein parameters or two files like SERGIO for MR and Gene parameters.
  random_key: 42 # Random key for jax.random
  logging: true # Flag to enable logging of the data to a TileDB array
  learn_params: false # Flag to perform backpropagation on a target_gene and target_prot values to learn the parameters.
  epochs: 0 # Number of epochs for backpropagation
  
  ## Gene related
  non_mr_basal: false # Flag to indicate if non Master Regulators have a basal rate
  noise: false # Add noise to gene concentrations
  noise_amplitude: 0.1 # Amplitude of the noise to be added
  decay: # Decay rates. Can be a single global value or specified for each gene
    - 0.8
  hill_coeffs: # Hill Coefficients. can be a single global value or specific for each pair.
    - 1.0

  ## Protein related
  protein_sim: true # Simulate protein concentrations

# Lattice free setup
spatial_sim:
  ## Mesh related
  height: 2 # Height dimension in the spatial simulator
  width: 2 # Width dimension in the spatial simulator
  depth: 2 # Depth dimension in the spatial simulator
  mesh_type: "lattice-free" # Type of mesh for the spatial simulation. Supports "lattice-free" and "grid"
  field_resolution: 2 # Divides the field into X divisions along each axis
  cell_concentration: 1 # Number of cells per field
  field_vol: 1.0 # Volume of each field
  n_cell_type: 2 # Number of types of cell. Each type of cell is characterized by a different set of parameters.
  cell_qty: # Proportion of each type of cell
    cell1: 0.8
    cell2: 0.2

  ## Diffusion related
  D: 0.5 # Value of the diffusion constant to be used
  diffusion_bool: true # Flag to enable diffusion
  n_neighbours: 1 # Number of fields to consider for computing field neighbours for diffusion. Selects the N closest fields.
  
  ## Simulation Related
  delta: 0.01 # Delta between successive timesteps in the simulation
  n_steps: 10 # Number of steps
  random_key: 42 # Random key for jax.random
  logging: true # Flag to enable logging of the data to a TileDB array

  ## Movement related
  movement_bool: true # Flag to enable movement of cells in field
  movement: # Movement parameters for each type of cell
    cell1:
      attraction_coeff: 1 # Coefficient for the attractive force between the cells
      repulsion_coeff: 0.1 # Coefficient for repulsive force between cells for overlaps and at short distances
      drift_vel_coeff: 0.01 # Coefficient for the drift velocity, added to the velocity for t+1
      random_vel_coeff: 0.01 # Coefficient for additive random velocity
    cell2:
      attraction_coeff: 2
      repulsion_coeff: 0.2
      drift_vel_coeff: 0.02
      random_vel_coeff: 0.05

  ## Cycle related
  cycle_bool: true # Flag for enabling cell cycling
  cycle: # Parameters for each cell type for cell cycling
    cell1:
      cycle_len: 10 # Length of the cell cycle
      interphase_len: 0.9 # Length of the cell's interphase
      necrosis_death_prob: 0.0001 # Probability of sudden death of the cell at each timestep
      apoptosis_death_prob: 0.001 # Probability 
      cell_target_vol: 1.0 # Target volume for the cell
      cell_density: 1.0 # Density of the cell. Used to compute mass of the cell from the volume
      cell_vol_growth_rate: 1.0 # Growth rate of the cell volume
      cell_decay_rate: 0.5 # Decay rate of the cell for programmed cell death
    cell2:
      cycle_len: 20
      interphase_len: 0.8
      necrosis_death_prob: 0.0001
      apoptosis_death_prob: 0.001
      cell_target_vol: 2.0
      cell_density: 2.0
      cell_vol_growth_rate: 1.5
      cell_decay_rate: 0.5
  
  ## Chemicals
  chemical: # Chemicals to simulate concentrations
    name: # Names of the chemicals
      - "chem1"
      - "chem2"
      - "chem3"
    mol_mass: # Molecular masses of the chemicals
      - 1
      - 2
      - 3
  
  ## Reactions
  reaction_bool: true # Flag to enable reactions
  reaction_prob: false # Use a probability-based reaction computation. Disabling it performs all reactions every timestep.
  reaction:
    reac_3: # Name of the reaction. This is a decay reaction
      order: 0 # Order of reaction
      rate_coeff: 1 # Rate coefficient of the reaction
      reactants: # Reactants names
        - "chem2"
      reactants_exponent: # Exponent of the reactant concentrations for reaction computation. In the same order as the reactants names
        - 3
      products: # Products names
      products_exponent: # Exponent of product concentrations for reaction computation. In the same order as the product names.
  
intervention: # Config for intervention. Leave empty if there is no intervention to perform. Look at intervention_api.md for more information
    spatial_sim: # Simulator to target
      - - "D"
        - 2
        - ["loop", 1, 2, 4]  # Type(Loop), Loop start time, loop length, gap between loops
        - ["global", [1,2,3]]
    grn:
      - - "decay"
        - 0.8
        - ["scheduled", 1]  # Type(Loop), Loop start time, loop length, gap between loops
        - ["global"]
```