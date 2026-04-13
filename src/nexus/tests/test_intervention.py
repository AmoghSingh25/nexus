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
    def test_cell_interventions(self):
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
            "cell_states",
            # "cell_time",  ## Should not be modified
            "cell_radius",
            "cell_mass",
        ]
        n_steps = 2
        interven_step = 1

        base_config["grn"]["n_steps"] = n_steps
        base_config["spatial_sim"]["n_steps"] = n_steps
        val_set = 10
        n_cell_ = 3
        cell_range = list(range(3))

        interventions = []
        for i in cell_params:
            interventions.append(
                [i, val_set, ["scheduled", interven_step], ["index", cell_range]]
            )
        base_config["intervention"]["spatial_sim"] = interventions
        grn_sim, spatial_sim = run_sim(base_config)

        base_config["grn"]["logging"] = False
        base_config["spatial_sim"]["logging"] = False

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
        for i in tqdm(range(n_steps)):
            intervention_flag and interven_manager.check(i)
            if i == interven_step:
                for i in cell_params:
                    _param = getattr(spatial_sim.mesh, i)
                    assert jnp.all(
                        _param[jnp.array(cell_range)] == val_set
                    ) and not jnp.all(
                        _param[jnp.array(range(n_cell_ + 1, 8))] == val_set
                    )
                break
            else:
                grn_sim.run_sim(step=i)
                spatial_sim.run_sim(step=i)
