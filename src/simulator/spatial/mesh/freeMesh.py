import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from simulator.spatial.field.freeField import FreeField
from simulator.spatial.utils.random_generators import generate_uniform


class FreeMesh:
    """
    Create a mesh of grid cells for 3D space. Performs flux and diffusion calculation, reaction, updates cells during simulation and other mesh and cell related functions.
    """

    def __init__(self, cfg, random_key=42):
        """

        :param self: GridMesh
        :param height: Height of the grid mesh - Y Axis. Must be >0
        :param width: Width of the grid mesh - X Axis. Must be >0
        :param depth: Depth of the grid mesh - Z Axis. Must be >0
        :param D: Diffusion constant
        :param cfg: uv Config
        :param random_key: Random key value for JAX random functions
        """

        self.width = cfg.width
        self.height = cfg.height
        self.depth = cfg.depth
        assert self.width > 0 and self.height > 0 and self.depth > 0, (
            "Spatial dimensions must be greater than 0"
        )
        self.dims = [self.width, self.height, self.depth]
        self.D = cfg.D
        self.n_chemicals = len(cfg["chemical"]["name"])

        self.key, self.sub_key = random.split(random.key(random_key))

        x, y, z = jnp.mgrid[0 : self.width, 0 : self.height, 0 : self.depth]
        self.positions = jnp.vstack([x.ravel(), y.ravel(), z.ravel()]).T

        self.cell_density = cfg.get("cell_density", 10)

        self.cell_vol = self.height * self.width * self.depth
        self.n_cells = int(self.cell_vol * self.cell_density)

        self.key, self.sub_key, self.positions = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 3),
        )
        self.positions = jnp.round(self.positions, decimals=2)

        self.cells = {}
        self.cell_ids = []
        self.reaction_bool = cfg.reaction_bool
        self.diffusion_bool = cfg.diffusion_bool
        curr_id = 0
        self.n_neighbours = 3  # TODO: Use config to set value

        for [i, j, k] in self.positions:
            cell_i = FreeField(
                pos=[i, j, k],
                D=self.D,
                key=curr_id,
                vol=self.cell_vol,
                id=curr_id,
                cfg=cfg,
            )
            cell_i.neighbours = self.get_neighbours(jnp.array([i, j, k]))
            self.cells[curr_id] = cell_i
            self.cell_ids.append(curr_id)
            curr_id += 1
        self.cell_ids = np.array(self.cell_ids)

    def calc_flux(self, cell_id):
        """
        Calculate flux for the cell at position _pos_.
        Also checks if part of the flux is already calculated by another cell before.

        :param pos: Position of cell to calculate the flux
        """
        curr_cell = self.cells[cell_id]
        neighbour_cells = []
        neigh_pos = []
        cell_mass = []
        cell_vol = []
        flux = jnp.zeros((self.n_chemicals, 1))
        for cell_i_id in curr_cell.neighbours:
            cell_i = self.cells[int(cell_i_id)]
            if curr_cell.flux.get(cell_i.id) is not None:
                flux += curr_cell.flux.get(cell_i.id)
                continue
            neighbour_cells.append(cell_i.id)
            cell_mass.append(cell_i.chem.chem_mass)
            cell_vol.append(cell_i.vol)
            neigh_pos.append(jnp.array(cell_i.pos))

        if len(neigh_pos) == 0:
            return flux

        neigh_pos = jnp.vstack(neigh_pos)
        cell_mass = jnp.array(cell_mass)
        cell_vol = jnp.array(cell_vol)
        distances = jnp.linalg.norm(neigh_pos - jnp.array(curr_cell.pos), axis=1)

        def calc_flux_i(D, cell1_mass, cell1_vol, cell2_mass, cell2_vol, dist_i):
            """Helper function for auto vectorization for flux calculation"""
            flux_i = -D * (cell2_mass * cell2_vol - cell1_mass * cell1_vol) / dist_i
            return flux_i

        auto_vec_flux = jax.vmap(calc_flux_i, in_axes=(None, 0, 0, None, None, 0))
        flux_list = auto_vec_flux(
            curr_cell.D,
            cell_mass,
            cell_vol,
            curr_cell.chem.chem_mass,
            curr_cell.vol,
            distances,
        )

        for i in range(len(flux_list)):
            self.cells[neighbour_cells[i]].flux[curr_cell.id] = -1 * flux_list[i]

        flux_list = jnp.append(flux_list, flux.reshape(-1, self.n_chemicals, 1), axis=0)
        total_flux = jnp.sum(flux_list, axis=0)
        return total_flux

    # TODO: Fix diffusion, large negative and large positive values
    def calc_conc_change(self, delta):
        """
        Calculate change in chemical concentration due to diffusion

        :param self: GridMesh
        :param delta: Simulation delta
        """

        # TODO: Assuming area of boundary is 1. Change to dynamic

        def compute_delta_m(cell_id):
            flux = self.calc_flux(cell_id)
            area = 1
            delta_m = flux * area * delta
            return delta_m

        delta_m_l = []
        for i in self.cells.keys():
            delta_m_l.append(compute_delta_m(cell_id=i))

        old_mass_l = []
        new_mass_l = []
        for i in self.cells.keys():
            chem_mass_i = self.cells[i].chem.chem_mass
            old_mass_l.append(chem_mass_i)
            chem_mass_i += delta_m_l[i]
            chem_mass_i.at[chem_mass_i < 0].set(0)
            self.cells[i].chem.chem_mass = chem_mass_i
            new_mass_l.append(chem_mass_i)
            self.cells[i].flux = {}
        return old_mass_l, new_mass_l

    def step(self, step_i, delta, logger):
        """
        Perform simulation step for the mesh and the cells within.

        :param self: GridMesh
        :param step_i: Step index
        :param delta: Simulation delta
        :param logger: MeshLogger object
        """
        # TODO: Vectorize steps

        # Perform diffusion
        if self.diffusion_bool:
            self.calc_conc_change(delta)
            logger is not None and logger.log_chem_state(step=step_i, cells=self.cells)

        def cell_step(cell_id):
            self.cells[cell_id].step(step=step_i, logger=logger)

        # Perform reactions
        if self.reaction_bool:
            for cell_id in self.cell_ids:
                cell_step(cell_id)

    def get_cell_id(self, pos):
        """
        Returns an ID of a cell, useful for an order of cells to compute flux i->j and uniquely identify cells

        :param pos: Position of the cell (i,j,k)
        """
        return pos[0] + self.height * pos[1] + (self.height * self.depth) * pos[2]

    def get_neighbours(self, pos):
        """
        Get neighbours of the cell at pos[idx].

        :param self: GridMesh
        :param idx: index of cell within pos[idx]
        """
        l1_norm = jnp.linalg.norm(pos - self.positions, axis=1, ord=1)
        neigh_idxs = jnp.argsort(l1_norm)[1 : self.n_neighbours + 1]
        return neigh_idxs
