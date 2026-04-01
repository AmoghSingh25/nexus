# Intervention API usage

The intervention API supports changing the parameters of the simulator by specifying the parameter, the value to be used, the spatial scope and the temporal scope. This document explains the parameters that are currently supported and the corresponding spatial and temporal scopes it supports. An example of the usage is given below

```python
- - "D" # Parameter name
    - 2 # New value of the parameter
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

## Spatial scopes

- `global`
  - `["global"]`
- `index`
  - `["index", [a,b,c...]]`: Index based scope that allows passing indices of entities (cells, RNAs, proteins) to target

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
| `cell_vel`               | Cell velocity                                        | All            | `global`, `index` |
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
