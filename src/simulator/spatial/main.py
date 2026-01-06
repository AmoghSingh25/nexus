import os
from simulator.spatial.spatialSim import SpatialSim
from hydra import (
    initialize_config_dir,
    compose,
)
from omegaconf import OmegaConf
import matplotlib

matplotlib.style.use("ggplot")


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


def main():
    _config = get_config()
    _config.spatial_sim.diffusion_bool = True
    _config.spatial_sim.reaction_bool = True
    _config.spatial_sim.logging = True
    print("Configuration being used - ")
    print(OmegaConf.to_yaml(_config))
    s1 = SpatialSim(_config.spatial_sim)
    s1.run_sim()


if __name__ == "__main__":
    main()
