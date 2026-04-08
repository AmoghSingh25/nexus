import jax
import jax.numpy as jnp
from nexus.simulator.spatial.utils.random_generators import (
    generate_poisson,
    generate_uniform,
    generate_permutation,
)


def calc_reaction_change(
    step,
    field_id,
    logger,
    reaction_key,
    n_reactions,
    chem_mass,
    delta,
    reaction_table,
    reaction_prob,
    reaction_matrix,
    reaction_order,
):
    """
    Calculate change in chemical concentration due to reactions. Randomly shuffles the reaction order during each step. The subsequent concentration from the reaction_i is used for reaction_i+1.
    Executes reaction based on probability.

    :param step: Simulation step idx
    :param cell_id: ID of the cell relating to the chemical field
    :param logger: Logger object (Currently setup to use mesh_logger)
    """

    key, sub_key, reaction_order = generate_permutation(
        key=reaction_key[0], sub_key=reaction_key[1], x=n_reactions
    )

    conc_t_0 = chem_mass
    reaction_ids = []

    for i in reaction_order:
        prob_i = reaction_prob[i]

        key, sub_key, random_prob = generate_uniform(key=key, sub_key=sub_key)

        def get_updated_conc(conc_t_0, key, sub_key):
            key, sub_key, poisson_i = generate_poisson(
                key=key,
                sub_key=sub_key,
                lam=reaction_matrix[i],
            )
            react_matrix_i = poisson_i * delta

            conc_t_1 = jax.lax.switch(
                reaction_order[i], reaction_table, (conc_t_0, react_matrix_i)
            )

            ## Skip reaction if resultant concentrations are negative - Assuming sufficient amount of reactant is not available
            def update_reaction():
                reaction_ids.append(i)
                return conc_t_1

            return jax.lax.cond(
                jnp.any(jnp.abs(react_matrix_i) * conc_t_1 < 0),
                lambda _: conc_t_0,
                lambda _: update_reaction(),
                None,
            )

        conc_t_0 = jax.lax.cond(
            random_prob <= prob_i,
            lambda _: get_updated_conc(conc_t_0, key, sub_key),
            lambda _: conc_t_0,
            None,
        )

    ## TODO: Fix logging with JAX traced arrays

    # chem_concs = np.array(chem_concs)
    # reaction_ids = np.array(reaction_ids, dtype=np.int32)

    # logger is not None and logger.log_reaction_order(
    #     step=step, cell_id=field_id, reaction_order=np.asarray(reaction_order)
    # )
    # logger is not None and logger.log_reaction_state(
    #     step=step, cell_id=field_id, reaction_ids=reaction_ids, chem_concs=chem_concs
    # )

    return jnp.array([key, sub_key]), conc_t_0


def calc_zero_order(args):
    """
    Perform zero order reaction and return updated concentrations.

    :param args[0]: Current chemical concentration
    :param args[1]: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
    """
    curr_conc = args[0].at[:].set(args[0] + args[1])
    return curr_conc


def calc_first_order(args):
    """
    Perform first order reaction and return updated concentrations.

    :param args[0]: Current chemical concentration
    :param args[1]: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
    """
    curr_conc = args[0].at[:].set(jnp.exp(jnp.log(args[0]) + args[1]))
    return curr_conc


def calc_second_order(args):
    """
    Perform second order reaction and return updated concentrations.

    :param args[0]: Current chemical concentration
    :param args[1]: Reaction matrix for reaction from Reaction.generate_reaction_matrix()
    """
    curr_conc = args[0].at[:].set(1 / ((1 / args[0]) - args[1]))
    return curr_conc
