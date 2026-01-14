import jax
from jax import random
import jax.numpy as jnp
from simulator.spatial.field.gridField import GridField
import numpy as np
from simulator.spatial.utils.random_generators import generate_choices


class GridMesh:
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
        if cfg.get("sparse_cells", False):
            self.n_cells = cfg.cell_num
            max_cells = int(self.height * self.width * self.depth)

            self.key, self.sub_key, self.positions = generate_choices(
                key=self.key,
                sub_key=self.sub_key,
                a=self.positions,
                shape=(self.n_cells,),
                replace=False,
            )

            if self.n_cells > max_cells:
                raise ValueError(
                    f"Number of cells {self.n_cells} exceeds maximum possible {max_cells} for given dimensions"
                )
        else:
            self.n_cells = int(self.height * self.width * self.depth)
        self.cell_vol = self.width * self.height * self.depth / self.n_cells

        self.cells = {}
        self.cell_ids = []
        self.reaction_bool = cfg.reaction_bool
        self.diffusion_bool = cfg.diffusion_bool
        curr_id = 0

        for [i, j, k] in self.positions:
            cell_i = GridField(
                pos=[i, j, k],
                D=self.D,
                key=(i + j + k),
                vol=self.cell_vol,
                id=curr_id,
                cfg=cfg,
            )
            cell_i.neighbours = self.get_neighbours(cell_i.pos)
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
            if jnp.any(self.cells[cell_id].chem.chem_mass + delta_m < 0):
                fix_flux(cell_id, self.cells[cell_id].chem.chem_mass + delta_m)
            return delta_m

        def fix_flux(cell_id, diff_mass):
            neighbours = self.cells[cell_id].neighbours
            neg_idx = jnp.where(diff_mass < 0)
            diff_i = (diff_mass[neg_idx] / neighbours.shape[0]).reshape(-1)
            for i in neighbours:
                flux_i = self.cells[i.item()].flux[cell_id][neg_idx] - diff_i
                self.cells[i.item()].flux[cell_id] = (
                    self.cells[i.item()].flux[cell_id].at[neg_idx].set(flux_i)
                )
                self.cells[cell_id].flux[i.item()] = (
                    self.cells[cell_id].flux[i.item()].at[neg_idx].set(-1 * flux_i)
                )

        delta_m_l = []
        ## Initial loop to fix negative masses
        for i in self.cells.keys():
            compute_delta_m(cell_id=i)

        ## Final loop to set fixed masses
        for i in self.cells.keys():
            delta_m_l.append(compute_delta_m(cell_id=i))

        for i in self.cells.keys():
            chem_mass_i = self.cells[i].chem.chem_mass
            chem_mass_i += delta_m_l[i]
            self.cells[i].chem.chem_mass = chem_mass_i
            self.cells[i].flux = {}

    def step(self, step_i, delta, logger):
        """
        Perform simulation step for the mesh and the cells within.

        :param self: GridMesh
        :param step_i: Step index
        :param delta: Simulation delta
        :param logger: mesh_logger object
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

    def get_neighbours(self, idx):
        """
        Get neighbours of the cell at pos[idx].

        :param self: GridMesh
        :param idx: index of cell within pos[idx]
        """

        # 3D neighbours - 26 neighbours
        idx = jnp.array(idx)
        neighbor_idx = jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, 0],
                [-1, -1, 1],
                [-1, 0, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, -1],
                [-1, 1, 0],
                [-1, 1, 1],
                [0, -1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, 0, -1],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],
                [1, -1, -1],
                [1, -1, 0],
                [1, -1, 1],
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, -1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        neighbor_pos = []

        def check_position(arr):
            if not np.any(np.all(self.positions == arr, axis=1)):
                return -1
            else:
                return np.where(np.all(self.positions == arr, axis=1))[0][0]

        for i in neighbor_idx:
            pos_i = i + idx
            neighbour_cell_id = check_position(pos_i)
            if neighbour_cell_id != -1:
                neighbor_pos.append(neighbour_cell_id)
        return neighbor_pos
