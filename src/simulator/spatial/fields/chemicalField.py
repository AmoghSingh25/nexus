import jax.numpy as jnp
from jax import random
from simulator.spatial.models.chemical import Chemical
from simulator.spatial.models.reaction import Reaction


class ChemicalField:
    def __init__(self, chem_names, mol_masses, reaction_config, delta, key=42):
        self.chemicals = []
        self.n_chemicals = len(chem_names)
        self.reactions = []
        self.mol_masses = mol_masses
        self.chem_names = chem_names
        self.delta = delta

        for i in range(self.n_chemicals):
            self.chemicals.append(
                Chemical(id=i, mol_mass=mol_masses[i], name=chem_names[i])
            )
        reaction_names = list(reaction_config.keys())
        self.reaction_order_sum = jnp.zeros((3,))
        for i in range(len(reaction_config)):
            reaction_i = reaction_config[reaction_names[i]]
            self.reactions.append(
                Reaction(
                    name=reaction_names[i],
                    id=i,
                    k=reaction_i.rate_coeff,
                    order=reaction_i.order,
                    products=[x[0] for x in reaction_i.products]
                    if reaction_i.products is not None
                    else [],
                    reactants=[x[0] for x in reaction_i.reactants]
                    if reaction_i.reactants is not None
                    else [],
                    chemicals=self.chem_names,
                )
            )
            self.reaction_order_sum = self.reaction_order_sum.at[reaction_i.order].set(
                self.reaction_order_sum[reaction_i.order] + reaction_i.rate_coeff
            )

        self.reaction_prob = jnp.zeros((3,))
        for i in range(len(self.reaction_order_sum)):
            # order_sum_i = (1-jnp.exp(-self.delta * self.reaction_order_sum[i])) / self.reaction_order_sum[i]
            order_sum_i = 1 / self.reaction_order_sum[i]
            self.reaction_prob = self.reaction_prob.at[i].set(order_sum_i)

        self.key, self.sub_key = random.split(random.key(key))
        self.chem_mass = random.uniform(self.sub_key, shape=(self.n_chemicals, 1))
        self.key, self.sub_key = random.split(self.key)

    def __repr__(self):
        return f"No. chemicals - {self.n_chemicals} \nMass - {len(self.reactions)}\n"

    def calc_reaction_change(self):
        react_matrix = jnp.zeros((3, self.n_chemicals, 1))
        for i in range(len(self.reactions)):
            prob_i = self.reactions[i].k * self.reaction_prob[self.reactions[i].order]
            random_prob = random.uniform(self.sub_key)

            self.key, self.sub_key = random.split(self.key)

            if random_prob <= prob_i:
                react_matrix = react_matrix.at[self.reactions[i].order].set(
                    react_matrix[self.reactions[i].order]
                    + self.reactions[i].generate_reaction_matrix()
                )

        react_matrix = react_matrix.at[:].set(react_matrix * self.delta)

        conc_t_1 = self.chem_mass
        conc_order_0 = self.calc_zero_order(conc_t_1, react_matrix=react_matrix[0])
        conc_order_1 = self.calc_first_order(conc_order_0, react_matrix=react_matrix[1])
        conc_order_2 = self.calc_second_order(
            conc_order_1, react_matrix=react_matrix[2]
        )

        self.chem_mass = conc_order_2

    def calc_zero_order(self, curr_conc, react_matrix):
        # Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(curr_conc + react_matrix)
        return curr_conc

    def calc_first_order(self, curr_conc, react_matrix):
        # Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(jnp.exp(jnp.log(curr_conc) + react_matrix))
        return curr_conc

    def calc_second_order(self, curr_conc, react_matrix):
        # Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(1 / (1 / curr_conc - react_matrix))
        return curr_conc

    def step(self):
        self.calc_reaction_change()
