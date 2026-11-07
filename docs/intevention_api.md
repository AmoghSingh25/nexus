# Intervention API usage

The intervention API supports changing the parameters of the simulator by specifying the parameter, the value to be used, the spatial scope and the temporal scope. This document explains the parameters that are currently supported and the corresponding spatial and temporal scopes it supports. An example of the usage is given below

```python
- - "D" # Parameter name
    - ["hard", 2] # Type of shift and new value of the parameter
    - ["loop", 1, 2, 4]  # Temporal scope - Type(Loop), Loop start time, loop length, gap between loops
    - ["index", [1,2,3]] # Spatial scope - Cell, list of cells to apply change on
```

## Temporal scopes

- `scheduled`
  - `["scheduled", a]`: `a` is the timestep that the intervention is scheduled for
- `loop`
  - `["loop", a, b, c]`: `a` - Loop start time, `b` - Loop length, `c` - Gap between successive loops
- `pulse`
  - `["pulse", a, b]`: `a` - Pulse start time, `b` - Pulse end time

Note: For parameters like `cell_positions` which change during the simulation, using a `loop` or `pulse` will modify the value to the given intervention value and then revert back to the **value at initialisation**. Using `scheduled` will modify the value at the specified timestep and not revert the values back.

## Spatial scopes

- `global`
  - `["global"]`
- `index`
  - `["index", [a,b,c...]]`: Index based scope that allows passing indices of entities (cells, RNAs, proteins) to target

## Intervention Types
- `hard`
  - `["hard", 0.2]` - Sets the value of the parameter to 0.2, performing a hard intervention
- `soft`
  - `["soft", "shift", 0.2]` - Performs a soft intervention where the value of the parameter is shifted by 0.2 as $x_t = x_{t-1} + 0.2$

## Spatial Sim interventions

| Parameter                | Description                                          | Temporal Scope | Spatial Scope     |
| ------------------------ | ---------------------------------------------------- | -------------- | ----------------- |
| `D`                      | Diffusion constant D                                 | All            | `global`          |
| `cell_positions`         | Coordinates of cells                                 | `scheduled`    | `global`, `index` |
| `cell_death_prob`        | Probability of cell death                            | `scheduled`    | `global`, `index` |
| `cell_prg_death_prob`    | Programmed cell death probability                    | All            | `global`, `index` |
| `cell_target_vol`        | Target volume for cells                              | `scheduled`    | `global`, `index` |
| `cell_vol_growth_rate`   | Rate of cell volume growth                           | All            | `global`, `index` |
| `cell_attraction_coeff`  | Cell attraction coefficient                          | All            | `global`, `index` |
| `cell_repulsion_coeff`   | Cell repulsion coefficient                           | All            | `global`, `index` |
| `cell_drift_vel_coeff`   | Drift velocity coefficient                           | All            | `global`, `index` |
| `cell_random_vel_coeff`  | Random velocity coefficient                          | All            | `global`, `index` |
| `cell_death_decay_coeff` | Decay coefficient when cell is undergoing cell death | All            | `global`, `index` |
| `cell_vel`               | Cell velocity (For interventions on `cell_vel`, the parameter `cell_residual_vel` can also be used which adds the components to the computed velocity )                                        | All            | `global`, `index` |
| `cell_states`            | Cell states                                          | All            | `global`, `index` |
| `cell_radius`            | Cell radius                                          | All            | `global`, `index` |
| `cell_mass`              | Cell mass                                            | All            | `global`, `index` |

## GRN Sim interventions

| Parameter         | Description                                                   | Temporal Scope | Spatial Scope      |
| ----------------- | ------------------------------------------------------------- | -------------- | ------------------ |
| `decay`           | Decay rates of the RNAs                                       | All            | `global`           |
| `prot_decay`      | Decay rates of the proteins                                   | All            | `global`, `index`  |
| `prot_tran_rates` | Transcription rates of the proteins                           | All            | `global`, `index`  |
| `prot_half_lives` | Half lives of the proteins                                    | All            | `global`, `index` |
| `ki_matrix`       | K_i matrix containing the transcription factors for the genes | All            | `global`          |

## Other interventions

The simulator can process other types of intervention and are as detailed below. For the usages, see the example config given below.

| Parameter         | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| `add_cell`        | Adds a new cell to the spatial simulator, with the parameters defined. | 
| `ablate`      | Removes a cell. The cell idx and type of cell death can be specified                                   |
| `add_particle` | Adds a particle that increases the presence of a chemical in a region. The increase is proportional to the existing concentration of chemical. |
| `set_particle` | Modifies the parameters of a chemical generating particle present in the simulator. |
| `remove_particle` | Removes the chemical generating particle. |
| `modify_edge` | Modify the GRN network. Can be used to modify the $ki_matrix$ or add in a new edge. |
| `block_edge` | Blockes the edge in the GRN network of random cells, the probability of selecting a cell specified with the random parameter.  |
| `add_reaction` | Add a new reaction to the simulator. |
| `modify_reaction` | Modify an existing reaction in the simulator |

## Random interventions

The intervention api also supports adding random interventions, which perform the given intervention with a random probability. The configuration for specifying the random intervention is given below.

```yaml
intervention:
  random_key: 42
  spatial_sim:
    - - "random" ## Random event
      - 0.2 ## Probability of event (computed over n_steps of the simulator)
      - - "cell_vel" # Definition of the intervention to perform.
        - ["soft", "shift", -0.2]
        - ["scheduled", null] # Timestep is null as the timestep of the event is decided randomly.
        - ["global"]
```

## Sample config

```yaml
defaults:
  - _self_
grn:
  ## Simulation related
  n_cells: 8
  delta: 0.01
  n_steps: 10
  config_file: "configs/sample_data/sample_network_1cell.yaml"
  random_key: 42
  logging: true
  log_dir: "logs"
  learn_params: false
  epochs: 0
  lr: 0.01

  ## Gene related
  non_mr_basal: false
  noise: false
  noise_amplitude: 0.1
  decay:
    - 0.8
  hill_coeffs:
    - 1.0

  ## Protein related
  protein_sim: true

# Lattice free setup
spatial_sim:
  ## Mesh related
  height: 2
  width: 2
  depth: 2
  mesh_type: "lattice-free"
  field_resolution: 2 # Divide into X divisions along each axis
  cell_concentration: 1 # Default value is 10
  field_vol: 1.0
  n_cell_type: 2

  ## Diffusion related
  # D: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  D: 0.2
  diffusion_bool: false
  n_neighbours: 1

  ## Simulation Related
  delta: 0.01
  n_steps: 10
  random_key: 42
  logging: true
  log_dir: "logs"

  ## Movement related
  movement_bool: true
  movement:
    cell1:
      qty_ratio: 0.6
      attraction_coeff: 1
      repulsion_coeff: 0.1
      drift_vel_coeff: 0.01
      random_vel_coeff: 0.01
    cell2:
      qty_ratio: 0.4
      attraction_coeff: 2
      repulsion_coeff: 0.2
      drift_vel_coeff: 0.02
      random_vel_coeff: 0.05

  ## Cycle related
  cycle_bool: true
  cycle:
    cell1:
      cycle_len: 3
      interphase_len: 0.9
      necrosis_death_prob: 0.0001
      apoptosis_death_prob: 0.001
      cell_target_vol: 1.0
      cell_density: 1.0
      cell_vol_growth_rate: 1.0
      cell_decay_rate: 0.5
      resource_limit:
        limits: [["chem1", 0.1], ["chem2", 0.2]] # Chemical name and minimum value of the chemical for cell division
        type: "hill" # Other option is hard cutoff
        hill_coeff: 1
        combine_func: "min" # Or "mult"
      contact_limit:
        limits: [3, 0.5] # Number of neighbours in the surrounding to (perform cell division)(for half rate cell division). Higher number of neighbours prevent cell division, radial limit
        type: "hill" # Other option is hard cutoff
        hill_coeff: 1 # Other option is hard cutoff
    cell2:
      cycle_len: 20
      interphase_len: 0.8
      necrosis_death_prob: 0.0001
      apoptosis_death_prob: 0.001
      cell_target_vol: 2.0
      cell_density: 2.0
      cell_vol_growth_rate: 1.5
      resource_limit:
        limits: [["chem1", 0.1]] # Chemical name and minimum value of the chemical for cell division
        type: "hard" # Other option is hard cutoff
      cell_decay_rate: 0.5

  ## Chemicals
  chemical:
    non_zero_init: True
    name:
      - "chem1"
      - "chem2"
      - "chem3"
      - "chem4"
    mol_mass:
      - 1
      - 2
      - 3
      - 4

  ## Reactions
  reaction_bool: true
  reaction_prob: false
  reaction:
    reac_3: # Decay of chemical
      order: 0
      rate_coeff: 1
      reactants:
        - "chem2"
      reactants_exponent:
        - 3
      products:
      products_exponent:

intervention:
  random_key: 42
  spatial_sim:
    - - "cell_target_vol"
      - ["soft", "shift", 0.1]
      - ["loop", 1, 2, 4] # Type(Loop), Loop start time, loop length, gap between loops
      - ["index", [1, 2, 3]]
    - - "cell_residual_vel"
      - ["hard", [[0.1, 0.2, 0.3]]]
      - ["scheduled", 2]
      - ["index", [1, 2, 3]]
  grn:
    - - "decay"
      - ["hard", 0.8]
      - ["scheduled", 1] # Type(Loop), Loop start time, loop length, gap between loops
      - ["global"]
  other: ## Add_cell intervention - Data required - Time, cell config
    - - "add_cell"
      - ["scheduled", 2] # Temporal param
      - [[1, 1, 1], 0, 0.4, 0] ## Other Params - Position, cell_state of new cell, radius of new cell, parent cell id (to inherit other cell related parameters)
    - - "ablate" ## alias remove_cell
      - ["scheduled", 2]
      - [[2], -1] # Cell id, Death type (-1 for sudden death, -2 for programmed cell death)
    - - "add_particle"
      - ["scheduled", 3]
      - [[0.1, 0.1, 0.1], 0, 0.1] # Pos, Chemical, Ratio of increase/decrease
    - - "remove_particle"
      - ["scheduled", 8]
      - [[0.1, 0.1, 0.1], 0] # Location, chemical_id
    - - "set_particle"
      - ["scheduled", 5]
      - [[0.1, 0.1, 0.1], 0, 0.4] # Location, chemical_id, new rate
    - - "modify_edge" # Also adds edge
      - ["scheduled", 3]
      - [0, 1, -5.0, 0.2] # Target gene, Regulator Gene, Weight of new edge, probability of adding edge (computed as probability of selecting eac cell)
    - - "block_edge"
      - ["scheduled", 3]
      - [0, 10, 0.3] # Target gene, Regulator gene, Probability (k) of selecting each cell
    - - "add_reaction"
      - ["scheduled", 1]
      - - reac_4: # Reaction name, all details in a dict again
            order: 2
            rate_coeff: 1
            reactants:
              - "chem2"
            reactants_exponent:
              - 3
            products:
              - "chem1"
            products_exponent:
              - 1
```