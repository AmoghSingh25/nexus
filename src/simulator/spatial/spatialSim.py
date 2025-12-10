from simulator.spatial.mesh.gridMesh import GridMesh
from omegaconf import DictConfig
from tqdm import tqdm
from simulator.spatial.logger.MeshLogger import DataLogger
import time
import os


class SpatialSim:
    """
    Spatial Simulator Class. Creates the mesh, initialized data using config, creates logger and runs the simulation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Docstring for __init__

        :param self: SpatialSim object
        :param cfg: uv Configuration
        """
        self.height = cfg.height
        self.width = cfg.width
        self.depth = cfg.depth
        self.delta = cfg.delta
        self.mesh_type = cfg.mesh_type
        self.D = cfg.D

        self.timestamp = str(int(time.time()))

        if self.mesh_type == "grid":
            self.mesh = GridMesh(
                cfg=cfg,
            )
        else:
            raise ValueError("Incorrect mesh type")

        self.n_steps = cfg.n_steps
        self.logging = cfg.get("logging", True)
        if self.logging:
            self.logger = DataLogger(
                log_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                ),
                file_name=self.timestamp,
                n_steps=cfg.n_steps,
                n_cells=self.mesh.n_cells,
                n_chems=len(cfg["chemical"].name),
                n_reactions=len(cfg.reaction),
            )
        else:
            self.logger = None

    def run_sim(self):
        """
        Run the spatial simulation

        :param self: SpatialSim object
        """
        self.logging and self.logger.log_chem_state(step=0, cells=self.mesh.cells)
        for i in tqdm(range(self.n_steps)):
            self.mesh.step(step_i=i + 1, delta=self.delta, logger=self.logger)
        self.logging and self.logger.log_chem_state(self.n_steps, self.mesh.cells)
