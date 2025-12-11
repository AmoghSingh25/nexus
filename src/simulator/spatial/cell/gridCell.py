from simulator.spatial.fields.chemicalField import ChemicalField


class GridCell:
    """
    GridCell class for GridMesh class. Used to represent the cell inside the GridMesh

    """

    def __init__(self, pos, D, key, vol, id, cfg):
        """
        GridCell class for GridMesh cell.

        :param D: Diffusion constant. Shape - (1, ). Possible shape - (n_neighbours, 1)
        :param conc: Chemical concentration.
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
            use_prob=cfg["reaction_prob"],
        )
        self.vol = vol
        self.id = id
        self.flux = {}
        self.neighbours = []

    def get_neighbours(self):
        """
        Return neighbours of the cell

        :param self: GridCell
        """
        return self.neighbours

    def __str__(self):
        return f"ID-{self.id}\npos - {self.pos}\nVol - {self.vol}"

    def step(self, step, logger):
        """
        Perform simulation step of the cell

        :param self: GridCell
        :param step: Step number
        :param logger: MeshLogger object
        """
        self.chem.step(step=step, cell_id=self.id, logger=logger)
