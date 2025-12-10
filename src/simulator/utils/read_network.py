import os
import jax.numpy as jnp
import networkx as nx
import polars as pl
import yaml
import logging
from simulator.utils.verify_network import _verify_network


def _create_bins(bin_vals, n_bins, n_cells):
    frmt_bins = []
    cells_per_bin = n_cells // n_bins
    for i in range(n_bins):
        frmt_bins.extend([bin_vals[i]] * cells_per_bin)
    if n_cells % n_bins != 0:
        frmt_bins.extend([bin_vals[-1]] * (n_cells - len(frmt_bins)))
    return jnp.array(frmt_bins)


def _read_txt(mr_file, gene_file, n_cells):
    mr_data = pl.read_csv(mr_file, separator=",", has_header=False).to_jax()
    mr_nodes = []
    n_bins = len(mr_data[0]) - 1
    for i in mr_data:
        _create_bins(i[1:], n_bins=9, n_cells=n_cells)
        mr_nodes.append(
            {
                "id": i[0].astype(int).item(),
                "type": "mr",
                "basal_rate": _create_bins(i[1:], n_bins, n_cells),
            }
        )

    gene_data = pl.read_csv(gene_file, separator=",", has_header=False).to_jax()
    gene_nodes = []
    edges = []
    for i in gene_data:
        num_reg = i[1].astype(int).item()
        gene_nodes.append(
            {
                "id": i[0].astype(int).item(),
                "regs": i[2 : 2 + num_reg],
                "ki": jnp.array(i[2 + num_reg : 2 + 2 * num_reg]),
                "coop": i[2 + 2 * num_reg : 2 + 3 * num_reg],
                "type": "g",
            }
        )
        for reg_i in gene_nodes[-1]["regs"].astype(int):
            edges.append((reg_i.item(), gene_nodes[-1]["id"]))

    g = nx.DiGraph()
    g.add_edges_from(edges)

    new_node_ids = {}
    ct = 0
    for i in jnp.array(list(nx.topological_sort(nx.line_graph(g)))):
        if i[0].item() in new_node_ids:
            continue
        new_node_ids[i[0].item()] = ct
        ct += 1
    for i in range(len(g.nodes())):
        if i in new_node_ids:
            continue
        new_node_ids[i] = ct
        ct += 1

    new_mr_nodes = []
    new_gene_nodes = []
    for i in mr_nodes:
        mr_node_i = i
        mr_node_i["id"] = new_node_ids[mr_node_i["id"]]
        new_mr_nodes.append(mr_node_i)

    for i in gene_nodes:
        g_node_i = i
        g_node_i["id"] = new_node_ids[g_node_i["id"]]
        for reg_i in range(len(i["regs"])):
            g_node_i["regs"] = (
                g_node_i["regs"].at[reg_i].set(new_node_ids[i["regs"][reg_i].item()])
            )
        g_node_i["regs"], g_node_i["ki"], g_node_i["coop"] = zip(
            *sorted(zip(g_node_i["regs"], g_node_i["ki"], g_node_i["coop"]))
        )

        g_node_i["regs"] = jnp.array(g_node_i["regs"])
        g_node_i["ki"] = jnp.array(g_node_i["ki"])
        g_node_i["coop"] = jnp.array(g_node_i["coop"])
        new_gene_nodes.append(g_node_i)

    complete_node_data = new_gene_nodes + new_mr_nodes
    complete_node_data.sort(key=lambda x: x["id"])
    renamed_edges = []
    for i in edges:
        renamed_edges.append((new_node_ids[i[0]], new_node_ids[i[1]]))
    return complete_node_data, renamed_edges


def _read_config(file_path):
    with open(file_path, "r") as file:
        node_data, edge_data = yaml.safe_load(file)
    return node_data, edge_data


def _read_data(gene_data, mr_data, config_file, n_cells, protein_sim=False):
    if config_file != "" and (gene_data is not None or mr_data is not None):
        raise Exception(
            "Only config file or gene data and mr data can be used in a single simulation"
        )

    if config_file != "" and os.path.exists(config_file):
        logging.info(f"Using the configuration file - {config_file}")
        node_data, edge_data = _read_config(config_file)
    else:
        if not (os.path.exists(gene_data) and os.path.exists(mr_data)):
            raise Exception("One of the data paths does not exist")
        logging.info("Using gene and mr data files")
        node_data, edge_data = _read_txt(
            mr_file=mr_data, gene_file=gene_data, n_cells=n_cells
        )

    copy_cells = _verify_network(node_data, edge_data, n_cells, protein_sim)
    logging.info("Network checks passed")
    return node_data, edge_data, copy_cells
