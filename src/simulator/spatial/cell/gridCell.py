from simulator.spatial.fields.chemicalField import ChemicalField


class GridCell:
    def __init__(self, pos, D, key, vol, id, cfg):
        """
        D - Diffusion constant. Shape - (1, ). Possible shape - (n_neighbours, 1)
        conc - Chemical concentration. Current shape - (n_cells, 1) assuming simulation of single chemical.
        """
        self.pos = tuple(pos)
        self.D = D
        self.key = key
        self.chem = ChemicalField(
            chem_names=cfg["chemical"]["name"],
            mol_masses=cfg["chemical"]["mol_mass"],
            key=self.key,
            reaction_config=cfg["reaction"],
            delta=cfg["delta"],
        )
        self.vol = vol
        self.id = id
        self.flux = {}
        self.neighbours = []

    def get_neighbours(self):
        return self.neighbours

    def __str__(self):
        return f"ID-{self.id}\npos - {self.pos}\nVol - {self.vol}"

    def step(self, step, logger):
        self.chem.step(step=step, cell_id=self.id, logger=logger)
