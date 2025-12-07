import jax
from jax import random
import jax.numpy as jnp
from simulator.spatial.cell.gridCell import GridCell
import numpy as np


class GridMesh:
    def __init__(self, height, width, depth, D, cfg, random_key=42):
        # Density - Cells(Entities) per unit volume
        # Each pos is positive - x>0, y>0, z>0

        assert width > 0 and height > 0, "Dimensions must be greater than 0"
        self.width = width
        self.height = height
        self.depth = depth
        self.dims = [self.width, self.height, self.depth]
        self.D = D
        self.n_chemicals = len(cfg["chemical"]["name"])

        self.key, self.sub_key = random.split(random.key(random_key))
        self.n_cells = int(self.height * self.width * self.depth)
        self.cell_vol = self.width * self.height * self.depth / self.n_cells
        self.key, self.sub_key = random.split(self.key)
        self.cells = []
        self.pos = []
        self.reaction_bool = cfg.reaction_bool
        self.diffusion_bool = cfg.diffusion_bool

        for i in range(self.width):
            for j in range(self.height):
                for k in range(self.depth):
                    cell_i = GridCell(
                        pos=[i, j, k],
                        D=self.D,
                        key=(i + j + k),
                        vol=self.cell_vol,
                        id=self.get_cell_id([i, j, k]),
                        cfg=cfg,
                    )
                    cell_i.neighbours = self.get_neighbours(cell_i.pos)
                    self.cells.append(cell_i)
                    self.pos.append([i, j, k])
        self.cells = np.array(self.cells).reshape((self.width, self.height, self.depth))
        self.pos = jnp.array(self.pos)

    def calc_flux(self, pos):
        """
        Calculate flux for the cell at position _pos_, to replace with cell _id_
        Checks if part of the flux is already calculated by another cell before

        :param pos: Position of cell to calculate the flux
        """
        curr_cell = self.cells[(*pos,)]
        neighbour_cells = []
        neigh_pos = []
        cell_mass = []
        cell_vol = []
        flux = jnp.zeros((self.n_chemicals, 1))
        for cell_pos_i in curr_cell.neighbours:
            cell_i = self.cells[(*cell_pos_i,)]
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
            self.cells[(*neigh_pos[i],)].flux[curr_cell.id] = -1 * flux_list[i]

        flux_list = jnp.append(flux_list, flux.reshape(-1, self.n_chemicals, 1), axis=0)
        total_flux = jnp.sum(flux_list, axis=0)
        return total_flux

    def calc_conc_change(self, delta):
        # Assuming area of boundary is 1
        def compute_delta_m(pos):
            flux = self.calc_flux(pos)
            area = 1
            delta_m = flux * area * delta
            return delta_m

        delta_m_l = []
        for i in range(len(self.pos)):
            delta_m_l.append(compute_delta_m(self.pos[i]))

        old_mass_l = []
        new_mass_l = []
        for i in range(len(self.pos)):
            chem_mass_i = self.cells[(*self.pos[i],)].chem.chem_mass
            old_mass_l.append(chem_mass_i)
            chem_mass_i += delta_m_l[i]
            chem_mass_i.at[chem_mass_i < 0].set(0)
            self.cells[(*self.pos[i],)].chem.chem_mass = chem_mass_i
            new_mass_l.append(chem_mass_i)
            self.cells[(*self.pos[i],)].flux = {}
        return old_mass_l, new_mass_l

    def get_masses(self):
        mass = []
        for i in range(len(self.pos)):
            mass.append(self.cells[(*self.pos[i],)].chem.chem_mass)
        return mass

    def step(self, delta):
        # Perform diffusion
        if self.diffusion_bool:
            self.calc_conc_change(delta)

        def cell_step(pos_i):
            self.cells[(*pos_i,)].step()

        # Perform reactions

        # cell_step_vec = jax.vmap(cell_step, in_axes=(0))
        # cell_step_vec(self.pos)
        if self.reaction_bool:
            for pos_i in self.pos:
                cell_step(pos_i)

    def get_cell_id(self, pos):
        """
        Returns an ID of a cell, useful for an order of cells to compute flux i->j and uniquely identify cells

        :param pos: Position of the cell to return the ID of
        """
        return 6 * pos[0] + 3 * pos[1] + pos[2]

    def get_neighbours(self, idx):
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
            for i in range(len(arr)):
                if arr[i] < 0 or arr[i] >= self.dims[i]:
                    return False
            return True

        for i in neighbor_idx:
            pos_i = i + idx
            if check_position(pos_i):
                neighbor_pos.append(pos_i)
        return neighbor_pos
