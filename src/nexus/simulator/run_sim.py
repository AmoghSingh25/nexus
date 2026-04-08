from tqdm import tqdm
import time
from nexus.simulator.grn.grnSim import GRNSim
from nexus.simulator.spatial.spatialSim import SpatialSim
import hydra
from omegaconf import DictConfig, open_dict
from nexus.simulator.intervention.intervention import InterventionManager


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="freemesh_config.yaml",
)
def run_sim(cfg: DictConfig) -> None:
    ## Checking configs
    timestep = str(int(time.time()))
    print("Time step - ", timestep)
    n_steps = cfg.grn.n_steps
    with open_dict(cfg):
        cfg.spatial_sim["log_file_name"] = timestep
        cfg.grn["log_file_name"] = timestep
    grn_sim = GRNSim(cfg.grn)
    spatial_sim = SpatialSim(cfg.spatial_sim)
    intervention_flag = False
    if cfg.get("intervention") is not None:
        interven_manager = InterventionManager(
            cfg=cfg, spatial_obj=spatial_sim, grn_obj=grn_sim
        )
        intervention_flag = True
    check_config(spatial_sim=spatial_sim, cfg=cfg)

    ## Running sim
    print("Running simulators...")
    for i in tqdm(range(n_steps)):
        intervention_flag and interven_manager.check(i)
        grn_sim.run_sim(step=i)
        spatial_sim.run_sim(step=i)
    return grn_sim, spatial_sim


def check_config(spatial_sim, cfg):
    if not (cfg.grn.n_cells == spatial_sim.mesh.n_cells):
        raise ValueError(
            "Config values incorrect, no. of cells in spatial sim and GRN sim to be run together"
        )
    if not (cfg.spatial_sim.n_steps == cfg.grn.n_steps):
        raise ValueError("Number of steps in both simulators must be equal.")


if __name__ == "__main__":
    run_sim()
