import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from jax import random
    import polars as pl
    import networkx as nx
    import matplotlib
    from tqdm import tqdm
    from simulator import gpsim

    matplotlib.style.use("default")
    return gpsim, mo, nx, pl, random, tqdm


@app.cell
def _(pl):
    grn_network = pl.read_csv(
        "notebooks/protein_rna_test_data/interaction_data.tab",
        separator="\t",
        has_header=False,
    )

    rna_prot_conc = pl.read_excel(
        "notebooks/protein_rna_test_data/occupancy.xls",
        sheet_name="Protein synthesis estimates (2)",
    )
    return grn_network, rna_prot_conc


@app.cell
def _(mo):
    mo.md(r"""## Creating the GRN network""")
    return


@app.cell
def _(grn_network):
    grn_network.head()
    return


@app.cell
def _(grn_network):
    # Regulator - Gene
    grn_edges = grn_network[:, [0, 2]]
    grn_edges = grn_edges.drop_nans()
    grn_edges = grn_edges.drop_nulls()
    grn_edges = grn_edges.to_numpy()
    return (grn_edges,)


@app.cell
def _(grn_edges, nx):
    g = nx.DiGraph()
    g.add_edges_from(grn_edges)
    return (g,)


@app.cell
def _(g):
    len(g.edges())
    return


@app.cell
def _(mo):
    mo.md(r"""## Creating the concentration data for RNA and proteins""")
    return


@app.cell
def _(rna_prot_conc):
    rna_prot_conc.head()
    return


@app.cell
def _(rna_prot_conc):
    ids = rna_prot_conc["YORF"]
    rna_conc = rna_prot_conc["mRNA"]
    prot_conc = rna_prot_conc["PROT"]
    prot_tran_rate = rna_prot_conc["Relative translation rate"]
    return ids, prot_conc, prot_tran_rate, rna_conc


@app.cell
def _(ids, prot_conc, prot_tran_rate, rna_conc):
    conc_dict = {
        ids[x]: {
            "p": prot_conc[x],
            "r": rna_conc[x],
            "idx": x,
            "prot_translation_rate": prot_tran_rate[x],
        }
        for x in range(len(ids))
    }
    return (conc_dict,)


@app.cell
def _():
    _s = """YJL157c
    YDR309c
    YER133w
    YDR098c
    YER174c
    YPL059w
    YML075c
    YLR450w
    YIL046w
    YDL130w
    YOL039w
    YDR382w
    YKR002w
    YER095w
    YML032c
    YIL148w
    YKR094c
    YOL020w"""
    shortlisted_prots = _s.split("\n")
    shortlisted_prots = [x.upper() for x in shortlisted_prots]
    prot_half_lives = [
        25,
        30,
        180,
        -1,
        -1,
        240,
        240,
        60,
        20,
        15,
        -1,
        300,
        840,
        120,
        15,
        120,
        120,
        90,
    ]
    return prot_half_lives, shortlisted_prots


@app.cell
def _(conc_dict, prot_half_lives, shortlisted_prots):
    for i in conc_dict:
        conc_dict[i]["prot_half_life"] = 0
        if i.upper() in shortlisted_prots:
            idx = shortlisted_prots.index(i)
            if prot_half_lives[idx] != -1:
                conc_dict[i]["prot_half_life"] = prot_half_lives[idx]
    return


@app.cell
def _(conc_dict, g, random, tqdm):
    simulator_config = []
    n_cells = 1000
    key, sub_key = random.split(random.key(42))
    nodes_names = []
    for _i in tqdm(conc_dict):
        if _i in g.nodes() and len(list(g.predecessors(_i))) == 0:
            _gene_entry = {}
            n_regs = len(list(g.successors(_i)))

            _gene_entry["basal_rate"] = (
                random.uniform(minval=0, maxval=1, shape=(n_cells, 1), key=sub_key),
            )
            key, sub_key = random.split(key)
            _gene_entry["ki"] = random.uniform(
                minval=-2, maxval=2, shape=(n_cells, n_regs), key=sub_key
            )
            key, sub_key = random.split(key)
            _gene_entry["type"] = "mr"

            simulator_config.append(_gene_entry)
            nodes_names.append(_i)

        elif _i in g.nodes():
            # print("Gene")
            _gene_entry = {}
            n_regs = len(list(g.successors(_i)))

            _gene_entry["basal_rate"] = random.uniform(
                minval=0, maxval=1, shape=(n_cells, 1), key=sub_key
            )
            key, sub_key = random.split(key)
            _gene_entry["ki"] = random.uniform(
                minval=-2, maxval=2, shape=(n_cells, n_regs), key=sub_key
            )
            key, sub_key = random.split(key)
            _gene_entry["type"] = "g"
            _gene_entry["name"] = _i

            if conc_dict[_i].get("prot_half_life") is not None:
                _gene_entry["prot_half_life"] = [conc_dict[_i]["prot_half_life"]]
                _gene_entry["prot_transcription_rate"] = [
                    conc_dict[_i]["prot_translation_rate"]
                ]

            simulator_config.append(_gene_entry)
            nodes_names.append(_i)
    return n_cells, nodes_names, simulator_config, sub_key


@app.cell
def _(g, nodes_names, tqdm):
    edges_list = []
    for _i, _j in tqdm(g.edges):
        if _i in nodes_names and _j in nodes_names:
            edges_list.append((nodes_names.index(_i), nodes_names.index(_j)))
    return (edges_list,)


@app.cell
def _(edges_list):
    len(edges_list)
    return


@app.cell
def _(simulator_config):
    len(simulator_config)
    return


@app.cell
def _(edges_list, gpsim, n_cells, random, simulator_config, sub_key):
    gpsim.simulator(
        node_set=simulator_config,
        edges_set=edges_list[:100],
        n_cells=n_cells,
        decay=random.uniform(
            minval=0,
            maxval=0.99,
            key=sub_key,
            shape=(n_cells, len(simulator_config), 1),
        ),
    )
    # key, sub_key = random.split(key)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
