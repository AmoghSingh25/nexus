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
        logging.warning(
            f"\t {var_name} values for node {idx} is given for a single cell, copying for all cells..."
        )
        return True
    return False


def _copy_param_vals(var, var_name, n_cells, n_genes, cell_dim=0, gene_dim=1):
    req_dim = [0, 0, 1]
    req_dim[cell_dim] = n_cells
    req_dim[gene_dim] = n_genes
    req_dim = tuple(req_dim)

    if (
        var.ndim > 3
        or (var.ndim == 3 and var.shape != req_dim)
        or (var.ndim == 2 and var.shape != req_dim[1:])
        or (var.ndim == 1 and var.shape != (1,))
    ):
        raise IncorrectDimensions(f"Wrong dimension for {var_name} vector")
    if var.ndim == 2:
        var = var.reshape(1, req_dim[1], 1)
        var = jnp.repeat(var, req_dim[0], axis=0)
    elif var.ndim == 1:
        var = var.reshape(1, 1, 1)
        var = jnp.repeat(var, req_dim[1], axis=1)
        var = jnp.repeat(var, req_dim[0], axis=0)
    return var


def _copy_param_single_gene(var, var_name, n_cells, cell_dim=0):
    req_dim = [0, 1]
    req_dim[cell_dim] = n_cells
    req_dim = tuple(req_dim)

    if var.ndim == 2:
        var = jnp.repeat(var, req_dim[0], axis=0)
    elif var.ndim == 1:
        var = var.reshape(1, 1)
        var = jnp.repeat(var, req_dim[0], axis=0)
    return var


def _verify_network(node_set, edges_set, n_cells, protein_sim, copy_data=False):
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
        "mr": set(["basal_rate", "type"]),
        "g": set(["ki", "type"]),
        "g_opt": set(["prot_half_life", "prot_transcription_rate"]),
    }
    copy_cells = False
    for idx in range(len(node_set)):
        node = node_set[idx]
        if "type" not in node.keys():
            raise MissingRequiredParams(
                f"'type' parameter missing for node {idx}. Must be either a Master Regulator(mr) or gene(g)"
            )
        if not req_keys[node["type"]] <= set(list(node.keys())):
            raise MissingRequiredParams(
                _missing_keys(req_keys[node["type"]], node.keys(), idx)
            )
        if (
            protein_sim
            and node["type"] == "g"
            and not req_keys[node["type"] + "_opt"] <= set(list(node.keys()))
        ):
            raise MissingRequiredParams(
                _missing_keys(req_keys[node["type"] + "_opt"], node.keys(), idx)
            )
        if node["type"] == "mr":
            try:
                basal_i = jnp.array(node["basal_rate"]).reshape(-1, 1)
                if _check_dims(idx, "Basal rates", basal_i, (n_cells, 1), n_cells):
                    basal_i = _copy_param_single_gene(
                        basal_i, "Basal rate", n_cells=n_cells
                    )
                node_set[idx]["basal_rate"] = basal_i
            except ValueError as e:
                raise IncorrectDimensions(
                    "Likely incorrect dimensions. Recheck the dimensions. \n" + str(e)
                )
        elif g.has_node(idx) and len(node["ki"]) > 0:
            n_regs = len(list(g.predecessors(idx)))
            try:
                ki = jnp.array(node["ki"]).reshape(-1, n_regs, 1)
                if _check_dims(idx, "ki", ki, (n_cells, n_regs, 1), n_cells):
                    ki = _copy_param_single_gene(ki, "ki", n_cells=n_cells)

                node_set[idx]["ki"] = ki

                if protein_sim:
                    p_half_life_i = jnp.array(node["prot_half_life"]).reshape(-1, 1)
                    prot_trans_rate_i = jnp.array(
                        node["prot_transcription_rate"]
                    ).reshape(-1, 1)
                    if _check_dims(
                        idx, "Prot half life", p_half_life_i, (n_cells, 1), n_cells
                    ):
                        p_half_life_i = _copy_param_single_gene(
                            p_half_life_i, "prot_half_life", n_cells=n_cells
                        )

                    if _check_dims(
                        idx,
                        "Prot transcription rate",
                        prot_trans_rate_i,
                        (n_cells, 1),
                        n_cells,
                    ):
                        prot_trans_rate_i = _copy_param_single_gene(
                            prot_trans_rate_i,
                            "prot_transcription_rate",
                            n_cells=n_cells,
                        )

                    node_set[idx]["prot_half_life"] = p_half_life_i
                    node_set[idx]["prot_transcription_rate"] = prot_trans_rate_i

            except ValueError as e:
                raise IncorrectDimensions(
                    "Likely incorrect dimensions. Recheck the dimensions. \n" + str(e)
                )
    return node_set, edges_set, copy_cells
