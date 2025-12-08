from simulator.spatial.mesh.gridMesh import GridMesh
from omegaconf import DictConfig
from tqdm import tqdm
from simulator.spatial.logger.MeshLogger import DataLogger
import time
import os


class SpatialSim:
    def __init__(self, cfg: DictConfig) -> None:
        self.height = cfg.spatial_sim.height
        self.width = cfg.spatial_sim.width
        self.depth = cfg.spatial_sim.depth
        self.delta = cfg.spatial_sim.delta
        self.mesh_type = cfg.spatial_sim.mesh_type
        self.D = cfg.spatial_sim.D

        self.timestamp = str(int(time.time()))

        if self.mesh_type == "grid":
            self.mesh = GridMesh(
                height=self.height,
                width=self.width,
                depth=self.depth,
                D=self.D,
                cfg=cfg["spatial_sim"],
            )
        else:
            raise ValueError("Incorrect mesh type")

        self.n_steps = cfg.spatial_sim.n_steps

        self.logger = DataLogger(
            log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
            file_name=self.timestamp,
            n_steps=cfg.spatial_sim.n_steps,
            n_cells=self.mesh.n_cells,
            n_chems=len(cfg["spatial_sim"]["chemical"].name),
            n_reactions=len(cfg.spatial_sim.reaction),
        )

    def run_sim(self):
        self.logger.log_chem_state(step=0, cells=self.mesh.cells)
        for i in tqdm(range(self.n_steps)):
            self.mesh.step(step_i=i + 1, delta=self.delta, logger=self.logger)
        self.logger.log_chem_state(self.n_steps, self.mesh.cells)
