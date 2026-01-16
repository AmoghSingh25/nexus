import numpy as np
from simulator.spatial_vec.utils.random_generators import (
    generate_permutation,
)


def check_cell_type_data(key, sub_key, n_cells, cfg):
    # Add qty ratios and should sum to 1
    # Check all the required sub-keys
    # Return a mask indicating the types in terms of n_cells - Ex, n_cells=10, n_types=2, prop=0.8,0.2, mask = [1,2,1,2,1,1,1,1,1,1]

    ## TODO: Perform checks
    req_keys_movement = [
        "qty_ratio",
        "attraction_coeff",
        "repulsion_coeff",
        "drift_vel_coeff",
        "random_vel_coeff",
    ]
    req_keys_cycle = [
        "cycle_len",
        "interphase_len",
        "cell_death_prob",
        "cell_target_vol",
        "cell_density",
        "cell_vol_growth_rate",
    ]
    n_cell_types = cfg.n_cell_type

    if len(cfg.movement) != n_cell_types or len(cfg.cycle) != n_cell_types:
        raise ValueError(
            "Incorrect number of entries for movement/cell config. Must be equal to number of cell types."
        )

    check_movement_flag = False
    check_cycle_flag = False
    cell_types = []
    param_dict = {}
    for i in range(len(cfg.movement)):
        cell_types.append(list(cfg.movement.keys())[i])
        if list(cfg.movement[cell_types[-1]].keys()) != req_keys_movement:
            check_movement_flag = True
            break
        if list(cfg.cycle[cell_types[-1]].keys()) != req_keys_cycle:
            check_cycle_flag = True
            break
        _cell_params = dict(cfg.cycle.get(cell_types[-1]))
        _cell_params.update(cfg.movement.get(cell_types[-1]))
        param_dict[cell_types[-1]] = _cell_params

    for i in cfg.cycle:
        if list(cfg.cycle.get(i).keys()) != req_keys_cycle:
            check_cycle_flag = True
            break
    if check_movement_flag:
        raise ValueError("Incorrect keys for movement config.")
    if check_cycle_flag:
        raise ValueError("Incorrect keys for cycle config.")

    ## Generate mask and arrays
    proportions = []
    interphase_len = np.ones((n_cells, 1))
    mitosis_len = np.ones((n_cells, 1))
    cell_death_prob = np.zeros((n_cells, 1))
    cell_density = np.zeros((n_cells, 1))
    cell_vol_growth_rate = np.zeros((n_cells, 1))
    cell_target_vol = np.zeros((n_cells, 1))

    cell_attraction_coeff = np.zeros((n_cells, 1))
    cell_repulsion_coeff = np.zeros((n_cells, 1))
    cell_drift_vel_coeff = np.zeros((n_cells, 1))
    cell_random_vel_coeff = np.zeros((n_cells, 1))

    cell_mask = np.zeros((n_cells, 1), dtype=np.int16)
    total_qty = 0.0
    cell_type = 0
    for i in cfg.cell_qty:
        proportions.append(total_qty + cfg.cell_qty.get(i))
        start, end = (
            int(total_qty * n_cells),
            int((total_qty + proportions[-1]) * n_cells),
        )

        cell_mask[start:end] = cell_type
        interphase_len[start:end] = (
            param_dict[i]["interphase_len"] * param_dict[i]["cycle_len"]
        )
        mitosis_len[start:end] = (1 - param_dict[i]["interphase_len"]) * param_dict[i][
            "cycle_len"
        ]
        cell_death_prob[start:end] = param_dict[i]["cell_death_prob"]
        cell_density[start:end] = param_dict[i]["cell_density"]
        cell_target_vol[start:end] = param_dict[i]["cell_target_vol"]
        cell_vol_growth_rate[start:end] = param_dict[i]["cell_vol_growth_rate"]

        cell_attraction_coeff[start:end] = param_dict[i]["attraction_coeff"]
        cell_repulsion_coeff[start:end] = param_dict[i]["repulsion_coeff"]
        cell_drift_vel_coeff[start:end] = param_dict[i]["drift_vel_coeff"]
        cell_random_vel_coeff[start:end] = param_dict[i]["random_vel_coeff"]

        cell_type += 1
        total_qty = proportions[-1]

    if total_qty != 1.0:
        raise ValueError("Incorrect ratios for cell qty. Total must be 1.0")

    ## Random permutation of mask
    key, sub_key, cell_mask = generate_permutation(
        key=key, sub_key=sub_key, x=cell_mask
    )

    return (
        key,
        sub_key,
        cell_mask,
        interphase_len,
        mitosis_len,
        cell_death_prob,
        cell_target_vol,
        cell_density,
        cell_vol_growth_rate,
        cell_attraction_coeff,
        cell_repulsion_coeff,
        cell_drift_vel_coeff,
        cell_random_vel_coeff,
    )
