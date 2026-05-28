from utils import log_cleanup
import jax.numpy as jnp
import jax
from nexus.simulator.spatial.spatialSim import SpatialSim
import numpy as np
import os
from hydra import initialize_config_dir, compose
import random


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


class TestSpatial:
    @log_cleanup
    def test_diffusion_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.reaction_bool = False
        base_config.spatial_sim.cycle_bool = False

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()
        if self.sim.mesh.n_fields < 2:
            assert False
        field_id_1 = 0
        field_id_2 = self.sim.mesh.field_neighbours[field_id_1][0]

        chem_before = self.sim.logger.retrieve_chem_data(
            step=0, field_id=[field_id_1, field_id_2]
        )["conc"]
        chem_after = self.sim.logger.retrieve_chem_data(
            step=-1, field_id=[field_id_1, field_id_2]
        )["conc"]

        min_chem_idx = np.argmin(chem_before)
        max_chem_idx = np.argmax(chem_after)

        if (
            chem_before[min_chem_idx] <= chem_after[min_chem_idx]
            and chem_before[max_chem_idx] >= chem_after[max_chem_idx]
        ):
            assert True
        else:
            assert False

    @log_cleanup
    def test_mass_preserving_freemesh(self):
        base_config = get_config("test_config")
        base_config.spatial_sim.reaction_bool = False
        base_config.spatial_sim.cycle_bool = False
        base_config.spatial_sim.n_steps = 30

        self.sim = SpatialSim(base_config.spatial_sim)

        def calc_mass():
            mass = 0
            for i in self.sim.mesh.field_id:
                mass += np.sum(self.sim.mesh.field_chem[i])
            return mass

        mass_before = calc_mass()
        self.sim.run_sim()
        mass_after = calc_mass()

        print("Before - ", mass_before)
        print("After - ", mass_after)
        if self.sim.mesh.n_fields < 2:
            assert False
        assert (mass_after - mass_before) <= 0.01 * mass_before

    @log_cleanup
    def test_reaction_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False
        base_config.spatial_sim.cycle_bool = False

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()
        chem_id = [1, 2]
        field_id = random.randint(a=0, b=self.sim.mesh.n_fields - 1)
        chem_before = self.sim.logger.retrieve_chem_data(
            step=0, field_id=field_id, chem_id=chem_id
        )["conc"]
        chem_after = self.sim.logger.retrieve_chem_data(
            step=-1, field_id=field_id, chem_id=chem_id
        )["conc"]

        if chem_before[0] >= chem_after[0] and chem_before[1] <= chem_after[1]:
            assert True
        else:
            assert False

    @log_cleanup
    def test_consistent_noise(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False
        base_config.spatial_sim.cycle_bool = False
        base_config.spatial_sim.random_key = 100

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()

        self.sim2 = SpatialSim(base_config.spatial_sim)
        self.sim2.run_sim()

        for attr in self.sim.mesh.__dict__:
            if isinstance(getattr(self.sim.mesh, attr), jax.Array):
                arr1 = getattr(self.sim.mesh, attr)
                arr2 = getattr(self.sim2.mesh, attr)
                assert jnp.all(arr1 == arr2)
