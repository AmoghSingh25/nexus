from simulator.spatial_vec.spatialSim import SpatialSimVec
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
    def test_diffusion_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.reaction_bool = False
        base_config.spatial_sim.cycle_bool = False

        try:
            s = SpatialSimVec(base_config.spatial_sim)
            s.run_sim()
            if s.mesh.n_fields < 2:
                s.cleanup()
                assert False
            field_id_1 = random.randint(a=0, b=s.mesh.n_fields - 1)
            field_id_2 = random.randint(a=0, b=s.mesh.n_fields - 1)
            while field_id_2 == field_id_1:
                field_id_1 = random.randint(a=0, b=s.mesh.n_fields - 1)
                field_id_2 = random.randint(a=0, b=s.mesh.n_fields - 1)

            chem_before = s.logger.retrieve_chem_data(
                step=0, field_id=[field_id_1, field_id_2]
            )["conc"]
            chem_after = s.logger.retrieve_chem_data(
                step=-1, field_id=[field_id_1, field_id_2]
            )["conc"]

            min_chem_idx = np.argmin(chem_before)
            max_chem_idx = np.argmax(chem_after)
            s.cleanup()

            print("DEBUG - - ")
            print(field_id_1, field_id_2)
            print(chem_before)
            print(chem_after)

            if (
                chem_before[min_chem_idx] <= chem_after[min_chem_idx]
                and chem_before[max_chem_idx] >= chem_after[max_chem_idx]
            ):
                assert True
            else:
                assert False
        except Exception as e:
            s.cleanup()
            raise e

    def test_reaction_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False
        base_config.spatial_sim.cycle_bool = False

        s = SpatialSimVec(base_config.spatial_sim)
        s.run_sim()
        chem_id = [1, 2]
        field_id = random.randint(a=0, b=s.mesh.n_fields - 1)
        chem_before = s.logger.retrieve_chem_data(
            step=0, field_id=field_id, chem_id=chem_id
        )["conc"]
        chem_after = s.logger.retrieve_chem_data(
            step=-1, field_id=field_id, chem_id=chem_id
        )["conc"]
        s.cleanup()

        if chem_before[0] >= chem_after[0] and chem_before[1] <= chem_after[1]:
            assert True
        else:
            assert False
