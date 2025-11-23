import numpy as np
from .mesh.gridMesh import GridMesh
from .cell.cell import Cell


class SpatialSim:
    def __init__(self, height=1, width=1, depth=1, density=10, cell_size=1e-2):
        self.height = height
        self.width = width
        self.depth = depth
        self.cell_size = cell_size
        self.density = density

        self.mesh = GridMesh(height=self.height, width=self.width, depth=depth)
        self.cells = []
        for i in range(self.width):
            for j in range(self.height):
                for k in range(self.depth):
                    self.cells.append(
                        Cell(pos=[i, j, k], D=0.5, mesh=self.mesh, key=i + j + k)
                    )
        self.cells = np.array(self.cells).reshape((self.width, self.height, self.depth))

    def calc_flux(self, pos):
        neighbour_cells = [self.cells[(*x,)] for x in self.mesh.get_neighbours(pos)]
        curr_cell = self.cells[(*pos,)]
        flux = sum(
            [
                -curr_cell.D * (x.chemical.conc - curr_cell.chemical.conc)
                for x in neighbour_cells
            ]
        )
        return flux
