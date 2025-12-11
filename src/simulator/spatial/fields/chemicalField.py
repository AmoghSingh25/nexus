import jax.numpy as jnp
from jax import random
from simulator.spatial.models.chemical import Chemical
from simulator.spatial.models.reaction import Reaction
import numpy as np
from simulator.spatial.utils.random_generators import (
    generate_poisson,
    generate_uniform,
    generate_permutation,
)


class ChemicalField:
    """
    Chemical Field class to store data about the chemicals inside a cell. Handles the reactions within the cells.
    """

    def __init__(self, chem_names, mol_masses, reaction_config, delta, key=42):
        """
        Initialize Chemical Field

        :param self: ChemicalField
        :param chem_names: List of chemical names in the simulation.
        :param mol_masses: List of molecular masses of the chemicals.
        :param reaction_config: Subset of the uv Config related to reactions.
        :param delta: Simulation timestep.
        :param key: Key for the JAX random functions
        """
        self.chemicals = []
        self.n_chemicals = len(chem_names)
        self.reactions = []
        self.mol_masses = mol_masses
        self.chem_names = chem_names
        self.delta = delta
        self.key, self.sub_key = random.split(random.key(key))

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
                    products=[x for x in reaction_i.products]
                    if reaction_i.products is not None
                    else [],
                    products_exp=reaction_i.products_exponent
                    if reaction_i.products is not None
                    else [],
                    reactants=[x for x in reaction_i.reactants]
                    if reaction_i.reactants is not None
                    else [],
                    reactants_exp=reaction_i.reactants_exponent
                    if reaction_i.reactants is not None
                    else [],
                    chemicals=self.chem_names,
                )
            )
            self.reaction_order_sum = self.reaction_order_sum.at[reaction_i.order].set(
                self.reaction_order_sum[reaction_i.order] + reaction_i.rate_coeff
            )

        self.reaction_prob = jnp.zeros((3,))
        for i in range(len(self.reactions)):
            if self.reaction_order_sum[i] > 0.0:
                prob_i = (
                    self.reactions[i].k
                    * (1 - jnp.exp(-self.delta * self.reaction_order_sum[i]))
                ) / self.reaction_order_sum[i]
            else:
                prob_i = 0.0
            self.reactions[i].prob = prob_i * 10.0

        self.key, self.sub_key, self.chem_mass = generate_uniform(
            key=self.key, sub_key=self.sub_key, shape=(self.n_chemicals, 1)
        )
        self.reaction_table = {
            0: self.calc_zero_order,
            1: self.calc_first_order,
            2: self.calc_second_order,
        }

    def __repr__(self):
        return f"No. chemicals - {self.n_chemicals} \nMass - {len(self.reactions)}\n"

    def calc_reaction_change(self, step, cell_id, logger):
        """
        Calculate change in chemical concentration due to reactions. Randomly shuffles the reaction order during each step. The subsequent concentration from the reaction_i is used for reaction_i+1.
        Executes reaction based on probability.

        :param self: ChemicalField
        :param step: Simulation step idx
        :param cell_id: ID of the cell relating to the chemical field
        :param logger: Logger object (Currently setup to use MeshLogger)
        """

        self.key, self.sub_key, reaction_order = generate_permutation(
            key=self.key, sub_key=self.sub_key, x=len(self.reactions)
        )

        conc_t_0 = self.chem_mass
        reaction_ids = []
        chem_concs = []

        for i in reaction_order:
            prob_i = self.reactions[i].prob

            self.key, self.sub_key, random_prob = generate_uniform(
                key=self.key, sub_key=self.sub_key
            )

            if random_prob <= prob_i:
                print("REACTION _ _ ", i)
                self.key, self.sub_key, poisson_i = generate_poisson(
                    key=self.key,
                    sub_key=self.sub_key,
                    lam=self.reactions[i].react_matrix,
                )
                react_matrix_i = poisson_i * self.delta

                conc_t_1 = self.reaction_table[self.reactions[i].order](
                    curr_conc=conc_t_0, react_matrix=react_matrix_i
                )
                reaction_ids.append(i)
                chem_concs.append(np.array(conc_t_1.reshape(-1)))
                conc_t_0 = conc_t_1

        chem_concs = np.array(chem_concs)
        reaction_ids = np.array(reaction_ids, dtype=np.int32)
        logger is not None and logger.log_reaction_state(
            step=step, cell_id=cell_id, reaction_ids=reaction_ids, chem_concs=chem_concs
        )
        self.chem_mass = conc_t_0

    def calc_zero_order(self, curr_conc, react_matrix):
        """
        Perform zero order reaction and return updated concentrations.

        :param self: ChemicalField
        :param curr_conc: Current chemical concentration
        :param react_matrix: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
        """
        # TODO: Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(curr_conc + react_matrix)
        return curr_conc

    def calc_first_order(self, curr_conc, react_matrix):
        """
        Perform first order reaction and return updated concentrations.

        :param self: ChemicalField
        :param curr_conc: Current chemical concentration
        :param react_matrix: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
        """
        # TODO: Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(jnp.exp(jnp.log(curr_conc) + react_matrix))
        return curr_conc

    def calc_second_order(self, curr_conc, react_matrix):
        """
        Perform second order reaction and return updated concentrations.

        :param self: ChemicalField
        :param curr_conc: Current chemical concentration
        :param react_matrix: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
        """
        # TODO: Replace _react_matrix_ with Sample(Poisson(lambda))
        curr_conc = curr_conc.at[:].set(1 / (1 / curr_conc - react_matrix))
        return curr_conc

    def step(self, step, logger, cell_id):
        """
        Execute simulation step for reaction.

        :param self: ChemicalField
        :param step: Index of the current step of the simulation
        :param logger: Logger object (Currently MeshLogger)
        :param cell_id: Index of the cell of the current ChemicalField
        """
        self.calc_reaction_change(step=step, logger=logger, cell_id=cell_id)
