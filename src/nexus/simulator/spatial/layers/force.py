import jax.numpy as jnp

# Drift velocity
# Attraction-repulsion forces
# Random velocity


def calc_inter_cell_force(
    dists,
    source_cell_size,
    neighbour_cell_sizes,
    source_cell_mass,
    neighbour_cell_masses,
    source_cell_pos,
    neigh_cell_pos,
    attraction_coeff,
    repulsion_coeff,
    delta,
):
    """
    Calculate the forces between two cells. The force is repulsive if there is overlap and attractive otherwise.

    :param dists: Distances of neighbours
    :param source_cell_size: Sizes of the source cell
    :param neighbour_cell_sizes: Sizes of the neighbour cells
    :param source_cell_mass: Mass of the source cell
    :param neighbour_cell_masses: Mass of the neighbour cells
    :param source_cell_pos: Positions of the source cell
    :param neigh_cell_pos: Positions of the neighbour cells
    :param attraction_coeff: Attraction coefficient in the configuration
    :param repulsion_coeff: Repulsion coefficient in the configuration
    :param delta: Delta value
    """
    combined_cell_sizes = (neighbour_cell_sizes + source_cell_size).reshape(-1)
    combined_cell_masses = (source_cell_mass + neighbour_cell_masses).reshape(-1)

    overlap_cells_mask = dists < combined_cell_sizes

    # Gives attraction force vector, -1* for repulsion
    force_vectors = neigh_cell_pos - source_cell_pos
    unit_force_vectors = (
        force_vectors / jnp.linalg.norm(force_vectors, axis=1, ord=2)[:, None]
    )

    # Calculate force so cells do not overlap in 1 simulation step after delta multiplication
    repulsion_acc = overlap_cells_mask * (combined_cell_sizes - dists) * 2.0 / delta
    repulsion_force = repulsion_coeff * repulsion_acc * combined_cell_masses

    attraction_force = (
        jnp.invert(overlap_cells_mask)
        * attraction_coeff
        * (source_cell_mass * neighbour_cell_masses).reshape(-1)
        / dists**2
    )

    repulsion_force_component = repulsion_force @ (-1 * unit_force_vectors)
    attraction_force_component = attraction_force @ unit_force_vectors
    inter_cell_vel = (
        attraction_force_component + repulsion_force_component
    ) / source_cell_mass
    return inter_cell_vel


def calc_drift_velocity(prev_vel, drift_vel_coeff):
    """
    Compute the drift velocity

    :param prev_vel: Previous velocity of the cell
    :param drift_vel_coeff: Drift velocity coefficient
    """
    return prev_vel * drift_vel_coeff


def calc_vel(
    dists,
    source_cell_size,
    neighbour_cell_sizes,
    source_cell_mass,
    neighbour_cell_masses,
    source_cell_pos,
    neigh_cell_pos,
    prev_vel,
    attraction_coeff=1,
    repulsion_coeff=1,
    drift_vel_coeff=0.1,
    delta=0.01,
):
    """
    Compute the new velocity of the cell

    :param dists: Distances of neighbours
    :param source_cell_size: Sizes of the source cell
    :param neighbour_cell_sizes: Sizes of the neighbour cells
    :param source_cell_mass: Mass of the source cell
    :param neighbour_cell_masses: Mass of the neighbour cells
    :param source_cell_pos: Positions of the source cell
    :param neigh_cell_pos: Positions of the neighbour cells
    :param prev_vel: Previous velocity of the cell
    :param attraction_coeff: Attraction coefficient in the configuration
    :param repulsion_coeff: Repulsion coefficient in the configuration
    :param drift_vel_coeff: Drift velocity coefficient in the configuration
    :param delta: Delta value
    """
    inter_cell_vel = calc_inter_cell_force(
        dists,
        source_cell_size,
        neighbour_cell_sizes,
        source_cell_mass,
        neighbour_cell_masses,
        source_cell_pos,
        neigh_cell_pos,
        attraction_coeff,
        repulsion_coeff,
        delta=delta,
    )
    drift_vel = calc_drift_velocity(prev_vel=prev_vel, drift_vel_coeff=drift_vel_coeff)
    total_cell_vel = inter_cell_vel + drift_vel
    return total_cell_vel
