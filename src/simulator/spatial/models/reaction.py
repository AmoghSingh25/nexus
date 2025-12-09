import jax.numpy as jnp


class Reaction:
    """
    Class for Reactions. Stores reaction-related parameters and data.
    """

    def __init__(self, id, k, reactants, products, chemicals, order, name="Reaction"):
        """
        Initialize Reaction class

        :param self: Reaction
        :param id: Reaction id
        :param k: Reaction rate constant
        :param reactants: List of Reactants
        :param products: List of Products
        :param chemicals: List of chemicals in the simulation
        :param order: Reaction order
        :param name: Reactio name
        """
        self.id = id
        self.k = k
        self.reactants = reactants
        self.products = products
        self.name = name
        self.order = order
        self.chemicals = chemicals
        self.n_chemicals = len(self.chemicals)
        self.reactant_id = jnp.array([self.chemicals.index(x) for x in self.reactants])
        self.prod_id = jnp.array([self.chemicals.index(x) for x in self.products])

    def __repr__(self):
        return f"name={self.name}, ID = {self.id}, reactants = {self.reactants}, products = {self.products}\n"

    def generate_reaction_matrix(self):
        """
        Generate a reaction matrix that outlines the changes in the concentrations of the chemicals.

        :param self: Reaction
        """
        react_matrix = jnp.zeros(shape=(self.n_chemicals, 1))
        for idx in self.reactant_id:
            react_matrix = react_matrix.at[idx].set(-self.k)
        for idx in self.prod_id:
            react_matrix = react_matrix.at[idx].set(self.k)
        return react_matrix
