from simulator.spatial.mesh.gridMesh import GridMesh
from omegaconf import DictConfig
from tqdm import tqdm


class SpatialSim:
    def __init__(self, cfg: DictConfig) -> None:
        self.height = cfg.spatial_sim.height
        self.width = cfg.spatial_sim.width
        self.depth = cfg.spatial_sim.depth
        self.delta = cfg.spatial_sim.delta
        self.mesh_type = cfg.spatial_sim.mesh_type
        self.D = cfg.spatial_sim.D

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

    def run_sim(self, n_steps):
        init_mass = self.mesh.get_masses()
        for i in tqdm(range(n_steps)):
            # Run chemical diffusion
            self.mesh.calc_conc_change(self.delta)
            # Run reaction
        final_mass = self.mesh.get_masses()
        return init_mass, final_mass
