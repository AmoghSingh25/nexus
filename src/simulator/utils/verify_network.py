# Code to verify the network that is taken as input in simulator/gpsim.py
# Ensuring node structure, data passed is valid

import jax.numpy as jnp
import networkx as nx
import logging


class MissingRequiredParams(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class IncorrectDimensions(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidEdges(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _missing_keys(req_keys, entry_keys, node_idx):
    message = f"Keys missing in node index {node_idx}, "
    message = message + ",".join(list(set(req_keys) - set(entry_keys)))
    return message


def _check_dims(idx, var_name, var, req_dim, n_cells, cell_dim=0):
    if var.shape != req_dim and var.shape[cell_dim] != 1:
        raise IncorrectDimensions(f"Incorrect dimensions for {var_name} for node {idx}")
    elif n_cells > 1 and var.shape[cell_dim] == 1:
        logging.warn(
            f"\t {var_name} values for node {idx} is given for a single cell, copying for all cells..."
        )


def _verify_network(node_set, edges_set, n_cells):
    """
    Checks performed:
    - Shapes of the arrays are as expected ((n_genes, n_cells), etc.)
    - All required values are provided
    - Edge indices are not out of bounds
    """

    g = nx.DiGraph()
    g.add_edges_from(edges_set)
    flattened_e_set = jnp.array(edges_set).reshape(-1)
    if max(flattened_e_set) >= len(node_set) or min(flattened_e_set) < 0:
        raise InvalidEdges("Invalid edge values.")

    req_keys = {
        "mr": ["basal_rate", "type"],
        "g": {"ki", "prot_half_life", "prot_transcription_rate", "type"},
    }
    for idx in range(len(node_set)):
        node = node_set[idx]
        if "type" not in node.keys():
            raise MissingRequiredParams(
                f"'type' parameter missing for node {idx}. Must be either a Master Regulator(mr) or gene(g)"
            )
        if sorted(req_keys[node["type"]]) != sorted(node.keys()):
            raise MissingRequiredParams(
                _missing_keys(req_keys[node["type"]], node.keys(), idx)
            )

        if node["type"] == "mr":
            try:
                basal_i = jnp.array(node["basal_rate"]).reshape(-1, 1)
                _check_dims(idx, "Basal rates", basal_i, (n_cells, 1), n_cells)
            except ValueError as e:
                raise IncorrectDimensions(
                    "Likely incorrect dimensions. Recheck the dimensions. \n" + str(e)
                )
        else:
            n_regs = len(list(g.predecessors(idx)))
            try:
                ki = jnp.array(node["ki"]).reshape(n_cells, n_regs, -1)
                p_half_life_i = jnp.array(node["prot_half_life"]).reshape(-1, 1)
                prot_trans_rate_i = jnp.array(node["prot_transcription_rate"]).reshape(
                    -1, 1
                )
            except ValueError as e:
                raise IncorrectDimensions(
                    "Likely incorrect dimensions. Recheck the dimensions. \n" + str(e)
                )

            _check_dims(idx, "ki", ki, (n_cells, n_regs, 1), n_cells)
            _check_dims(idx, "Prot half life", p_half_life_i, (n_cells, 1), n_cells)
            _check_dims(
                idx, "Prot transcription rate", prot_trans_rate_i, (n_cells, 1), n_cells
            )

    return True
