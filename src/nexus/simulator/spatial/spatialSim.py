from nexus.simulator.spatial.mesh.mesh import Mesh
from omegaconf import DictConfig
from tqdm import tqdm
from nexus.simulator.spatial.logger.mesh_logger import FieldLogger
from nexus.simulator.spatial.logger.spatial_logger import SpatialLogger
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
        self.base_log_dir = os.path.join(cfg.get("log_dir", "logs"))

        self.timestamp = cfg.get("log_file_name", str(int(time.time())))

        if not (self.mesh_type == "grid" or self.mesh_type == "lattice-free"):
            raise ValueError("Incorrect mesh type")
        self.mesh = Mesh(cfg=cfg, mesh_type=self.mesh_type)

        self.n_steps = cfg.n_steps
        self.logging = cfg.get("logging", True)
        if self.logging:
            self.logger = FieldLogger(
                log_dir=self.base_log_dir,
                file_name=self.timestamp,
                n_steps=cfg.n_steps,
                n_cells=self.mesh.n_fields,
                n_chems=len(cfg["chemical"].name),
                n_reactions=0 if not cfg.reaction_bool else len(cfg.reaction),
                n_fields=self.mesh.n_fields,
                axis_divs=[self.width, self.height, self.depth],
                field_res=cfg.field_resolution,
            )
            self.pos_logger = SpatialLogger(
                log_dir=self.base_log_dir,
                file_name=self.timestamp,
                n_steps=cfg.n_steps,
                n_cells=self.mesh.n_cells,
            )
            self.logger.log_fields_pos(self.mesh.field_positions)
        else:
            self.logger = None
            self.pos_logger = None

    def run_sim(self, step=None):
        """
        Run the spatial simulation

        :param self: SpatialSim object
        """

        self.logging and self.logger.log_chem_state(
            step=0, field_chem=self.mesh.field_chem
        )
        if step is None:
            for i in tqdm(range(self.n_steps)):
                self.mesh.step(step_i=i + 1, logger=self.logger)
                if self.logging:
                    self.pos_logger.log_cell_pos(
                        i,
                        self.mesh.cell_positions,
                        self.mesh.cell_radius,
                        self.mesh.cell_states,
                    )

            ##DEBUG: Error in mesh.field_chem for GridMesh
            self.logging and self.logger.log_chem_state(
                self.n_steps, field_chem=self.mesh.field_chem
            )
        else:
            self.mesh.step(step_i=step, logger=self.logger)
            if self.logging:
                self.pos_logger.log_cell_pos(
                    step,
                    self.mesh.cell_positions,
                    self.mesh.cell_radius,
                    self.mesh.cell_states,
                )

    def cleanup(self):
        """
        Deletes the log files generated during the run

        :param self: SpatialSim
        """
        if self.logger is not None:
            self.logger.cleanup()
