from ..fields.chemical import Chemical
from ..mesh.gridMesh import GridMesh


class Cell:
    def __init__(self, pos, D, mesh: GridMesh, key):
        """
        D - Diffusion constant. Shape - (1, ). Possible shape - (n_neighbours, 1)
        conc - Chemical concentration. Current shape - (n_cells, 1) assuming simulation of single chemical.
        """
        self.pos = pos
        self.mesh = mesh
        self.D = D
        self.chemical = Chemical(D, self.mesh, key=key)

    def __repr__(self):
        return f"pos - {self.pos}\nChemical - \t{self.chemical}"
