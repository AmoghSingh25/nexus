class Chemical:
    def __init__(self, id, mol_mass=1, name="chem1"):
        self.name = name
        self.id = id
        self.mol_mass = mol_mass

    def __repr__(self):
        return f"{self.name} \n ID - {self.id}\n"
