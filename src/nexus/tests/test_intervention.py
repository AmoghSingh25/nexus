import random
import copy
from tqdm import tqdm
import jax.numpy as jnp
from nexus.simulator.run_sim import run_sim, check_config
import os
from hydra import initialize_config_dir, compose
from nexus.simulator.grn.grnSim import GRNSim
from nexus.simulator.spatial.spatialSim import SpatialSim
from nexus.simulator.intervention.intervention import InterventionManager


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


class TestInterventions:
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
        # "cell_states", ## TODO: Create seperate test
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
    config = base_config
    random.seed(42)

    def test_cell_scheduled_interventions(self):
        val_set = random.random()
        cell_range = list(range(3))
        base_config = copy.deepcopy(self.config)

        interventions = []
        for i in self.cell_params:
            interventions.append(
                [i, val_set, ["scheduled", self.interven_step], ["index", cell_range]]
            )
        base_config["intervention"]["spatial_sim"] = interventions
        grn_sim, spatial_sim = run_sim(base_config)

        grn_sim = GRNSim(base_config.grn)
        spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config, spatial_obj=spatial_sim, grn_obj=grn_sim
        )
        intervention_flag = True
        check_config(spatial_sim=spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        for i in tqdm(range(self.n_steps)):
            intervention_flag and interven_manager.check(i)
            if i == self.interven_step:
                for i in self.cell_params:
                    _param = getattr(spatial_sim.mesh, i)
                    assert jnp.all(
                        _param[jnp.array(cell_range)] == val_set
                    ) and not jnp.all(
                        _param[jnp.array(range(self.n_cell + 1, 8))] == val_set
                    )
                break
            else:
                grn_sim.run_sim(step=i)
                spatial_sim.run_sim(step=i)

    def test_cell_pulse_intervention(self):
        base_config = copy.deepcopy(self.config)
        interven_start = 1
        interven_end = 3

        interventions = []
        pulse_val = random.random()
        cell_range = [0, 1]

        for i in self.cell_params:
            interventions.append(
                [
                    i,
                    pulse_val,
                    ["pulse", interven_start, interven_end],
                    ["index", cell_range],
                ]
            )

        base_config["intervention"]["spatial_sim"] = interventions

        grn_sim = GRNSim(base_config.grn)
        spatial_sim = SpatialSim(base_config.spatial_sim)
        interven_manager = InterventionManager(
            cfg=base_config, spatial_obj=spatial_sim, grn_obj=grn_sim
        )
        intervention_flag = True

        for t in range(self.n_steps):
            intervention_flag and interven_manager.check(t)
            if t == interven_start or t == interven_end or t < interven_start:
                for i in self.cell_params:
                    current_param = getattr(spatial_sim.mesh, i)
                    vals_at_indices = current_param[jnp.array(cell_range)]
                    if t == interven_start:
                        assert jnp.all(vals_at_indices == pulse_val), (
                            f"Failed to start pulse at t={t}"
                        )
                        assert not jnp.all(
                            current_param[jnp.array(range(self.n_cell + 1, 8))]
                            == pulse_val
                        ), f"Modified other cell states at t={t}"
                    elif t == interven_end or t < interven_start:
                        assert jnp.all(vals_at_indices != pulse_val), (
                            f"Failed to revert pulse at t={t}"
                        )

    def test_grn_interventions(self):
        val_set = random.random()
        cell_range = list(range(3))
        base_config = copy.deepcopy(self.config)

        interventions = []
        for i in self.grn_params:
            interventions.append(
                [i, val_set, ["scheduled", self.interven_step], ["index", cell_range]]
            )
        base_config["intervention"]["grn"] = interventions
        grn_sim, spatial_sim = run_sim(base_config)

        grn_sim = GRNSim(base_config.grn)
        spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config, spatial_obj=spatial_sim, grn_obj=grn_sim
        )
        intervention_flag = True
        check_config(spatial_sim=spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        for i in tqdm(range(self.n_steps)):
            intervention_flag and interven_manager.check(i)
            if i == self.interven_step:
                for i in self.grn_params:
                    _param = getattr(grn_sim, i)
                    assert jnp.all(
                        _param[jnp.array(cell_range)] == val_set
                    ) and not jnp.all(
                        _param[jnp.array(range(self.n_cell + 1, 8))] == val_set
                    )
                break
            else:
                grn_sim.run_sim(step=i)
                spatial_sim.run_sim(step=i)
