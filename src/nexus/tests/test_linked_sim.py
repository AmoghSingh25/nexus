import copy
from nexus.tests.utils import log_cleanup, get_config
import jax.numpy as jnp
from nexus.simulator.spatial.spatialSim import SpatialSim
import random
from nexus.simulator import run_linked_sim


class TestLinkedSim:
    base_config = get_config("test_config")
    cell_params = [
        "cell_positions",
        "cell_death_prob",
        "cell_prg_death_prob",
        "cell_density",
        "cell_target_vol",
        "cell_vol_growth_rate",
        "cell_attraction_coeff",
        "cell_repulsion_coeff",
        "cell_drift_vel_coeff",
        "cell_random_vel_coeff",
        "cell_death_decay_coeff",
        "cell_vel",
        # "cell_time",  ## Should not be modified
        "cell_radius",
        "cell_mass",
    ]
    grn_params = [
        "decay",
        "prot_decay",
        "prot_tran_rates",
        "prot_half_lives",
        "ki_matrix",
    ]
    n_steps = 5
    interven_step = 1
    n_cell = 3

    base_config["grn"]["n_steps"] = n_steps
    base_config["spatial_sim"]["n_steps"] = n_steps
    base_config["grn"]["logging"] = False
    base_config["spatial_sim"]["logging"] = False
    base_config["intervention"]["spatial_sim"] = []
    base_config["intervention"]["grn"] = []
    base_config["intervention"]["other"] = []
    config = base_config
    random.seed(42)

    @log_cleanup
    def test_diffusion_linked_sim(self):
        base_config = copy.deepcopy(self.config)
        base_config["grn"]["n_steps"] = 10
        base_config["spatial_sim"]["n_steps"] = 10
        base_config["spatial_sim"]["reaction"] = {}
        base_config["spatial_sim"]["diffusion_bool"] = False

        field_concs, cell_concs = run_linked_sim.run_sim(base_config)

        assert not jnp.allclose(field_concs[0], field_concs[-1]) and not jnp.allclose(
            cell_concs[0], cell_concs[-1]
        )
        spatial_sim = SpatialSim(base_config.spatial_sim)
        cell_vols = spatial_sim.mesh.cell_vol
        field_vols = spatial_sim.mesh.field_vol
        associated_fields = spatial_sim.mesh.get_assigned_fields(
            norm_ord=2, clustering_type="k_mean"
        )

        associated_field_map = {}
        for j in range(spatial_sim.mesh.n_cells):
            field_idx = associated_fields[j].item()
            if field_idx not in associated_field_map:
                associated_field_map[field_idx] = []
            associated_field_map[field_idx].append(j)

        n_chemicals = spatial_sim.mesh.n_chemicals

        for field_idx, cell_indices in associated_field_map.items():
            for c in range(n_chemicals):
                field_conc_t0 = field_concs[0][field_idx, c, 0] / field_vols[field_idx]
                field_conc_t1 = field_concs[1][field_idx, c, 0] / field_vols[field_idx]

                cell_concs_t0 = jnp.array(
                    [cell_concs[0][c, j] / cell_vols[j] for j in cell_indices]
                )

                if jnp.all(cell_concs_t0 > field_conc_t0):
                    assert field_conc_t1 > field_conc_t0
                elif jnp.all(cell_concs_t0 < field_conc_t0):
                    assert field_conc_t1 < field_conc_t0
