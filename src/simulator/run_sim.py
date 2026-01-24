import time
from simulator.grn.grnSim import GRNSim
from simulator.spatial_vec.spatialSim import SpatialSimVec
import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(
    version_base=None, config_path="../../configs", config_name="freemesh_config.yaml"
)
def run_sim(cfg: DictConfig) -> None:
    ## Checking configs
    timestep = str(int(time.time()))
    with open_dict(cfg):
        cfg.spatial_sim["log_file_name"] = timestep
        cfg.grn["log_file_name"] = timestep
    spatial_sim = SpatialSimVec(cfg.spatial_sim)
    grn_sim = GRNSim(cfg.grn)
    check_config(spatial_sim=spatial_sim, cfg=cfg)

    ## Running sim
    print("Running spatial sim")
    spatial_sim.run_sim()
    print("Running GRN sim")
    grn_sim.run_sim()
    print("Completed running simulators")


def check_config(spatial_sim, cfg):
    if not cfg.grn.n_cells == spatial_sim.mesh.n_cells:
        raise ValueError(
            "Config values incorrect, no. of cells in spatial sim and GRN sim must be equal"
        )


if __name__ == "__main__":
    run_sim()
