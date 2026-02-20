class Chemical:
    """
    Class for chemicals.
    """

    def __init__(self, id, mol_mass=1, name="chem1"):
        """
        Initialize Chemical class

        :param self: Chemical
        :param id: Chemical id
        :param mol_mass: Molecular mass of the chemical
        :param name: Name of the chemical
        """
        self.name = name
        self.id = id
        self.mol_mass = mol_mass

    def __repr__(self):
        return f"{self.name} \n ID - {self.id}\n"
