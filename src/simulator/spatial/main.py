import numpy as np
from .mesh.gridMesh import GridMesh
from .cell.cell import Cell
import jax.numpy as jnp


class SpatialSim:
    def __init__(self, height=3, width=3, depth=3, cell_size=1e-2, delta=0.01):
        self.height = height
        self.width = width
        self.depth = depth
        self.cell_size = cell_size
        self.delta = delta

        self.mesh = GridMesh(height=self.height, width=self.width, depth=depth)
        self.cells = []
        for i in range(self.width):
            for j in range(self.height):
                for k in range(self.depth):
                    self.cells.append(
                        Cell(
                            pos=[i, j, k],
                            D=0.5,
                            mesh=self.mesh,
                            key=(i + j + k),
                            vol=self.mesh.cell_vol,
                        )
                    )
        self.cells = np.array(self.cells).reshape((self.width, self.height, self.depth))

    def calc_flux(self, pos):
        neighbour_cells = [self.cells[(*x,)] for x in self.mesh.get_neighbours(pos)]
        curr_cell = self.cells[(*pos,)]
        neigh_pos = jnp.stack([jnp.array(x.pos) for x in neighbour_cells])
        distances = jnp.linalg.norm(neigh_pos - jnp.array(curr_cell.pos), axis=1)
        flux = sum(
            [
                -curr_cell.D
                * (
                    neighbour_cells[x].chemical.mass * neighbour_cells[x].vol
                    - curr_cell.chemical.mass * curr_cell.vol
                )
                / distances[x]
                for x in range(len(neighbour_cells))
            ]
        )
        return flux

    def calc_conc_change(self, pos):
        # Assuming area of boundary is 1
        flux = self.calc_flux(pos)
        area = 1
        delta_m = flux * area * self.delta
        curr_cell = self.cells[(*pos,)]
        print(pos, delta_m)
        curr_cell.chemical.mass += delta_m
