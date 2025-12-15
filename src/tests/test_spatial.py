from simulator.spatial import spatialSim
import numpy as np
import os
from hydra import initialize_config_dir, compose


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


class TestSpatial:
    def test_diffusion(self):
        base_config = get_config()
        base_config.spatial_sim.reaction_bool = False

        s = spatialSim.SpatialSim(base_config.spatial_sim)
        s.run_sim()
        chem_id = 0
        cells = [0, 1]
        chem_before = s.logger.retrieve_chem_data(
            step=0, cell_id=cells, chem_id=chem_id
        )["conc"]
        chem_after = s.logger.retrieve_chem_data(
            step=-1, cell_id=cells, chem_id=chem_id
        )["conc"]
        s.cleanup()
        if np.argmin(chem_before) == np.argmin(chem_after) and np.argmax(
            chem_before
        ) == np.argmax(chem_after):
            assert True
        else:
            assert False

    def test_reaction(self):
        base_config = get_config()
        base_config.spatial_sim.diffusion_bool = False

        s = spatialSim.SpatialSim(base_config.spatial_sim)
        s.run_sim()
        chem_id = [1, 2]
        cells = 0
        chem_before = s.logger.retrieve_chem_data(
            step=0, cell_id=cells, chem_id=chem_id
        )["conc"]
        chem_after = s.logger.retrieve_chem_data(
            step=-1, cell_id=cells, chem_id=chem_id
        )["conc"]
        s.cleanup()
        if chem_before[0] >= chem_after[0] and chem_before[1] <= chem_after[1]:
            assert True
        else:
            assert False

    # TODO: Fix test
    # def test_diffusion_freemesh(self):
    #     base_config = get_config("freemesh_config")
    #     base_config.spatial_sim.reaction_bool = False

    #     s = spatialSim.SpatialSim(base_config.spatial_sim)
    #     s.run_sim()
    #     chem_id = 0
    #     cells = [0, 1]
    #     chem_before = s.logger.retrieve_chem_data(
    #         step=0, cell_id=cells, chem_id=chem_id
    #     )["conc"]
    #     chem_after = s.logger.retrieve_chem_data(
    #         step=-1, cell_id=cells, chem_id=chem_id
    #     )["conc"]
    # min_chem_idx = np.argmin(chem_before)
    # max_chem_idx = np.argmax(chem_after)

    # if np.argmin(chem_before) == np.argmin(chem_after) and np.argmax(
    #     chem_before
    # ) == np.argmax(chem_after):
    #     assert True
    # else:
    #     assert False

    def test_reaction_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False

        s = spatialSim.SpatialSim(base_config.spatial_sim)
        s.run_sim()
        chem_id = [1, 2]
        cells = 0
        chem_before = s.logger.retrieve_chem_data(
            step=0, cell_id=cells, chem_id=chem_id
        )["conc"]
        chem_after = s.logger.retrieve_chem_data(
            step=-1, cell_id=cells, chem_id=chem_id
        )["conc"]
        s.cleanup()
        if chem_before[0] >= chem_after[0] and chem_before[1] <= chem_after[1]:
            assert True
        else:
            assert False
