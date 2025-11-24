from ..fields.chemical import Chemical
from ..mesh.gridMesh import GridMesh


class Cell:
    def __init__(self, pos, D, mesh: GridMesh, key, vol):
        """
        D - Diffusion constant. Shape - (1, ). Possible shape - (n_neighbours, 1)
        conc - Chemical concentration. Current shape - (n_cells, 1) assuming simulation of single chemical.
        """
        self.pos = tuple(pos)
        self.mesh = mesh
        self.D = D
        self.key = key
        self.chemical = Chemical(key=self.key)
        self.vol = vol

    def __repr__(self):
        return f"pos - {self.pos}\nVol - {self.vol}\nChemical - \t{self.chemical}"
