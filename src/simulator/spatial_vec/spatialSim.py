from simulator.spatial_vec.mesh.gridMesh import GridMesh
from simulator.spatial_vec.mesh.freeMesh import FreeMesh
from omegaconf import DictConfig
from tqdm import tqdm
from simulator.spatial_vec.logger.mesh_logger import FieldLogger
from simulator.spatial_vec.logger.spatial_logger import SpatialLogger
import time
import os


class SpatialSimVec:
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

        self.timestamp = cfg.get("log_file_name", str(int(time.time())))

        if self.mesh_type == "grid":
            self.mesh = GridMesh(
                cfg=cfg,
            )
        elif self.mesh_type == "lattice-free":
            self.mesh = FreeMesh(cfg=cfg)
        else:
            raise ValueError("Incorrect mesh type")

        self.n_steps = cfg.n_steps
        self.logging = cfg.get("logging", True)
        if self.logging:
            self.logger = FieldLogger(
                log_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                ),
                file_name=self.timestamp,
                n_steps=cfg.n_steps,
                n_cells=self.mesh.n_fields,
                n_chems=len(cfg["chemical"].name),
                n_reactions=len(cfg.reaction),
                n_fields=self.mesh.n_fields,
                axis_divs=[self.width, self.height, self.depth],
                field_res=cfg.field_resolution,
            )
            self.pos_logger = SpatialLogger(
                log_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                ),
                file_name=self.timestamp,
                n_steps=cfg.n_steps,
                n_cells=self.mesh.n_cells,
            )
            self.logger.log_fields_pos(self.mesh.field_positions)
        else:
            self.logger = None
            self.pos_logger = None

    def run_sim(self):
        """
        Run the spatial simulation

        :param self: SpatialSim object
        """

        self.logging and self.logger.log_chem_state(
            step=0, field_chem=self.mesh.field_chem
        )
        cell_vols = []
        for i in tqdm(range(self.n_steps)):
            cell_vols.append(
                self.mesh.step(step_i=i + 1, delta=self.delta, logger=self.logger)
            )
            self.pos_logger.log_cell_pos(
                i,
                self.mesh.cell_positions,
                self.mesh.cell_radius,
                self.mesh.cell_states,
            )
        self.mesh.pl.close()

        ##DEBUG: Error in mesh.field_chem for GridMesh
        self.logging and self.logger.log_chem_state(
            self.n_steps, field_chem=self.mesh.field_chem
        )
        return cell_vols

    def cleanup(self):
        """
        Deletes the log files generated during the run

        :param self: SpatialSimVec
        """
        if self.logger is not None:
            self.logger.cleanup()
