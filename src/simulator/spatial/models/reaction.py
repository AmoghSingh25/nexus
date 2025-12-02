import jax.numpy as jnp


class Reaction:
    def __init__(self, id, k, reactants, products, chemicals, order, name="Reaction"):
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
        react_matrix = jnp.zeros(shape=(self.n_chemicals, 1))
        for idx in self.reactant_id:
            react_matrix = react_matrix.at[idx].set(self.k)
        for idx in self.prod_id:
            react_matrix = react_matrix.at[idx].set(-self.k)
        return react_matrix
