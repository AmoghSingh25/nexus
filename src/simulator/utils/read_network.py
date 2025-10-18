import os
import jax.numpy as jnp
import networkx as nx
from jax import random
import polars as pl


def _read_txt(mr_file, gene_file, n_cells):
    key, sub_key = random.split(random.key(42))
    mr_data = pl.read_csv(mr_file, separator=",", has_header=False).to_jax()
    mr_nodes = []
    for i in mr_data:
        mr_nodes.append(
            {
                "id": i[0].astype(int).item(),
                "type": "mr",
                "b_low": jnp.min(i[1:]).item(),
                "b_high": jnp.max(i[1:]).item(),
                "basal_rate": random.uniform(
                    sub_key,
                    (n_cells, 1),
                    minval=jnp.min(i[1:]).item(),
                    maxval=jnp.max(i[1:]).item(),
                ),
            }
        )
        key, sub_key = random.split(key)

    gene_data = pl.read_csv(gene_file, separator=",", has_header=False).to_jax()
    gene_nodes = []
    edges = []
    for i in gene_data:
        num_reg = i[1].astype(int).item()
        gene_nodes.append(
            {
                "id": i[0].astype(int).item(),
                "regs": i[2 : 2 + num_reg],
                "ki": i[2 + num_reg : 2 + 2 * num_reg],
                "coop": i[2 + 2 * num_reg : 2 + 3 * num_reg],
                "type": "g",
            }
        )
        for reg_i in gene_nodes[-1]["regs"].astype(int):
            edges.append((reg_i.item(), gene_nodes[-1]["id"]))
    # edges = jnp.array(edges)
    g = nx.DiGraph()
    g.add_edges_from(edges)

    new_node_ids = {}
    ct = 0
    for i in jnp.array(list(nx.topological_sort(nx.line_graph(g)))).flatten():
        if i.item() in new_node_ids:
            continue
        new_node_ids[i.item()] = ct
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
    renamed_edges = []
    for i in edges:
        renamed_edges.append((new_node_ids[i[0]], new_node_ids[i[1]]))
    return complete_node_data, renamed_edges


def _read_config(file_path):
    print("config = ", file_path)


def _read_data(gene_data, mr_data, config_file, n_cells):
    if config_file != "" and (gene_data is not None or mr_data is not None):
        raise Exception(
            "Only config file or gene data and mr data can be used in a single simulation"
        )

    if config_file != "" and os.path.exists(config_file):
        print(f"Using the configuration file - {config_file}")
    else:
        if not (os.path.exists(gene_data) and os.path.exists(mr_data)):
            raise Exception("One of the data paths does not exist")
        print("Using gene and mr data files")
        return _read_txt(mr_file=mr_data, gene_file=gene_data, n_cells=n_cells)
