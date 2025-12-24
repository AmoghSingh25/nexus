from simulator.spatial.layers.chemical import ChemicalLayer


class GridField:
    """
    GridField class for a field in the GridMesh class. Used to represent the field inside the GridMesh

    """

    def __init__(self, pos, D, key, vol, id, cfg):
        """
        GridField class for GridMesh cell.

        :param D: Diffusion constant. Shape - (1, ). Possible shape - (n_neighbours, 1)
        :param conc: Chemical concentration.
        """

        self.pos = tuple(pos)
        self.D = D
        self.key = key
        self.chem = ChemicalLayer(
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

        :param self: GridField
        """
        return self.neighbours

    def __str__(self):
        return f"ID-{self.id}\npos - {self.pos}\nVol - {self.vol}"

    def step(self, step, logger):
        """
        Perform simulation step of the cell

        :param self: GridField
        :param step: Step number
        :param logger: MeshLogger object
        """
        self.chem.step(step=step, cell_id=self.id, logger=logger)
