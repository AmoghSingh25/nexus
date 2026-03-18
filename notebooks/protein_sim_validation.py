import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from jax import random, nn
    import polars as pl
    import networkx as nx
    import matplotlib
    from tqdm import tqdm
    from pympler.classtracker import asizeof
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    from hydra import initialize_config_dir, compose

    matplotlib.style.use("default")
    return (
        compose,
        initialize_config_dir,
        jnp,
        mo,
        nn,
        np,
        nx,
        os,
        pl,
        plt,
        random,
        time,
        tqdm,
    )


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg
    return (get_config,)


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
    mo.md(r"""
    ## Creating the GRN network
    """)
    return


@app.cell
def _(rna_prot_conc):
    rna_prot_conc.head()
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
    mo.md(r"""
    ## Creating the concentration data for RNA and proteins
    """)
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
    n_cells = 10
    key, sub_key = random.split(random.key(42))
    nodes_names = []

    basal_rate_range = (0, 1e-3)
    ki_range = (-1e-3, 1e-3)

    for _i in tqdm(conc_dict):
        if _i in g.nodes() and len(list(g.predecessors(_i))) == 0:
            _gene_entry = {}
            n_regs = len(list(g.successors(_i)))

            _gene_entry["basal_rate"] = (
                random.uniform(
                    minval=basal_rate_range[0],
                    maxval=basal_rate_range[1],
                    shape=(n_cells, 1),
                    key=sub_key,
                ),
            )
            key, sub_key = random.split(key)
            _gene_entry["ki"] = random.uniform(
                minval=ki_range[0],
                maxval=ki_range[1],
                shape=(n_cells, n_regs),
                key=sub_key,
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
                minval=basal_rate_range[0],
                maxval=basal_rate_range[1],
                shape=(n_cells, 1),
                key=sub_key,
            )
            key, sub_key = random.split(key)
            _gene_entry["ki"] = random.uniform(
                minval=ki_range[0],
                maxval=ki_range[1],
                shape=(n_cells, n_regs),
                key=sub_key,
            )
            key, sub_key = random.split(key)
            _gene_entry["type"] = "g"
            _gene_entry["name"] = _i

            _gene_entry["prot_half_life"] = [conc_dict[_i]["prot_half_life"]]
            _gene_entry["prot_transcription_rate"] = [
                conc_dict[_i]["prot_translation_rate"]
            ]

            simulator_config.append(_gene_entry)
            nodes_names.append(_i)
    return ki_range, n_cells, nodes_names, simulator_config


@app.cell
def _(g, nodes_names, nx, tqdm):
    edges_list = []
    _edges = []
    for _i, _j in tqdm(g.edges):
        if _i in nodes_names and _j in nodes_names:
            edges_list.append((nodes_names.index(_i), nodes_names.index(_j)))
            _edges.append((_i, _j))
    g_updated = nx.DiGraph()
    g_updated.add_edges_from(_edges)
    return (edges_list,)


@app.cell
def _(
    edges_list,
    ki_range,
    n_cells,
    nodes_names,
    nx,
    random,
    simulator_config,
):
    _g = nx.DiGraph()
    _temp = [(nodes_names[i[0]], nodes_names[i[1]]) for i in edges_list]
    _g.add_edges_from(_temp)

    node_set = []
    edges_set = []
    node_names_refined = []
    _ct = 0
    for _i in range(len(simulator_config)):
        if simulator_config[_i].get("name") is not None and _g.has_node(
            simulator_config[_i]["name"]
        ):
            node_set.append(simulator_config[_i])
            node_names_refined.append(simulator_config[_i]["name"])

    for _i in _temp:
        if _i[0] in node_names_refined and _i[1] in node_names_refined:
            edges_set.append(
                (node_names_refined.index(_i[0]), node_names_refined.index(_i[1]))
            )

    del _g
    _g1 = nx.DiGraph()
    _g1.add_edges_from(
        [(node_names_refined[_i[0]], node_names_refined[_i[1]]) for _i in edges_set]
    )

    _key, _sub_key = random.split(random.key(42))

    for _i in range(len(node_set)):
        _n_regs = len(list(_g1.predecessors(node_set[_i]["name"])))
        node_set[_i]["ki"] = random.uniform(
            minval=ki_range[0],
            maxval=ki_range[1],
            shape=(n_cells, _n_regs),
            key=_sub_key,
        )
        _key, _sub_key = random.split(_key)
    return edges_set, node_names_refined, node_set


@app.cell
def _(edges_set, node_set):
    print("Number of edges - ", len(edges_set))
    print("Number of nodes - ", len(node_set))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning parameters with backprop
    """)
    return


@app.cell
def _(
    edges_set,
    get_config,
    ids,
    jnp,
    n_cells,
    nn,
    node_set,
    np,
    prot_conc,
    random,
    rna_conc,
    time,
):
    _prot_conc_target = []
    _output_rna_names = np.array([_i["name"] for _i in node_set])
    _target_rna_ids = [ids.to_list().index(_i) for _i in _output_rna_names]

    for _i in range(len(node_set)):
        _prot_conc_target.append(prot_conc[list(ids).index(node_set[_i]["name"])])

    _prot_conc_target = nn.standardize(jnp.array(_prot_conc_target))

    _rna_target = nn.standardize(rna_conc.to_numpy()[_target_rna_ids])

    from simulator.grn.grnSim import GRNSim

    _key, _sub_key = random.split(random.key(42))
    _mu_decay = 5.6e-1
    _std_decay = 1e-1
    _decay = _mu_decay + _std_decay * random.normal(
        key=_sub_key, shape=(n_cells, 1913, 1)
    )

    cfg = get_config(config_name="config")
    cfg.grn.n_cells = n_cells
    cfg.grn.protein_sim = True
    cfg.grn.non_mr_basal = True
    cfg.grn.decay = _decay.tolist()
    cfg.grn.logging = True
    cfg.grn.learn_params = True  # Toggle if disabling backprop
    cfg.grn.epochs = 1000

    _t1 = time.time()
    sim = GRNSim(
        node_set=node_set,
        edges_set=edges_set,
        cfg=cfg.grn,
        target_gene_conc=_rna_target,
        target_prot_conc=_prot_conc_target,
    )
    prev_gene_conc, prev_prot_conc = sim.gene_conc, sim.prot_conc
    prev_params = [
        sim.basal_rates,
        sim.decay,
        sim.ki_matrix,
        sim.hill_coeffs,
        sim.prot_tran_rates,
        sim.prot_decay,
    ]
    if cfg.grn.learn_params:
        sim.basal_rates = sim.learnt_gene_params['basal_rates']
        sim.decay = sim.learnt_gene_params['decay']
        sim.ki_matrix = sim.learnt_gene_params['ki_matrix']
        sim.hill_coeffs = sim.learnt_gene_params['hill_coeffs']
        sim.prot_tran_rates = sim.learnt_prot_params['prot_tran_rates']
        sim.prot_decay = sim.learnt_prot_params['prot_decay']
        sim.steady_states, sim.prot_steady_state, _, _ = sim.calc_steady_states(
            learn_params=False
        )
    # sim.run_sim()
    return (sim,)


@app.cell
def _(plt, sim):
    # plt.plot(prev_gene_conc[:, 0], label="Without backprop")
    plt.plot(sim.steady_states[:, 0], label="Pred")
    plt.plot(sim.target_gene_conc, label="Target")
    # plt.yscale("log")
    plt.legend()
    return


@app.cell
def _(plt, sim):
    # plt.plot(prev_gene_conc[:, 0], label="Without backprop")
    plt.plot(sim.prot_steady_state[:, 0], label="Pred")
    plt.plot(sim.target_prot_conc, label="Target")
    # plt.yscale("log")
    plt.legend()
    return


@app.cell
def _(jnp, nn):
    def z_score_norm(inp, axis=0):
        return nn.standardize(inp)
        mean = jnp.mean(inp, axis=axis)
        std = jnp.std(inp, axis=axis)
        inp = (inp - mean) / std
        return inp
    return (z_score_norm,)


@app.cell
def _(
    ids,
    jnp,
    n_cells,
    node_names_refined,
    node_set,
    np,
    plt,
    prot_conc,
    rna_conc,
    shortlisted_prots,
    sim,
    z_score_norm,
):
    _prot_conc_target = []
    _output_prots = np.array(node_names_refined)[
        jnp.unique(
            jnp.where(
                (sim.prot_steady_state != jnp.inf)
                & (sim.prot_steady_state != -jnp.inf)
            )[0]
        )
    ]

    _pred_prod_ids = jnp.unique(
        jnp.where(
            (sim.prot_steady_state != jnp.inf) & (sim.prot_steady_state != -jnp.inf)
        )[0]
    )
    _output_prod_names = np.array(node_names_refined)[_pred_prod_ids]
    _target_prots_ids = [ids.to_list().index(_i) for _i in _output_prod_names]

    _output_rna_names = np.array([_i["name"] for _i in node_set])
    _target_rna_ids = [ids.to_list().index(_i) for _i in _output_rna_names]

    _rna_pred = sim.steady_states.reshape(-1, n_cells)
    _prot_pred =sim.prot_steady_state[_pred_prod_ids].reshape(-1, n_cells)

    for _i in range(len(prot_conc)):
        if ids[_i] in _output_prod_names:
            _prot_conc_target.append(prot_conc[_i])


    _prot_conc_shortlisted = []
    _prot_conc_pred_shortlisted = []
    for _i in range(len(prot_conc)):
        if ids[_i] in shortlisted_prots:
            _prot_conc_shortlisted.append(prot_conc[_i])
            _prot_conc_pred_shortlisted.append(_prot_pred[_i])

    _prot_conc_target = jnp.array(_prot_conc_target)

    _prot_conc_shortlisted = jnp.array(_prot_conc_shortlisted)
    _prot_conc_pred_shortlisted = jnp.array(_prot_conc_pred_shortlisted)

    _prot_target = z_score_norm(_prot_conc_target)
    _rna_target = z_score_norm(rna_conc.to_numpy()[_target_rna_ids])

    _min_gene_loss = jnp.inf


    def _calc_mse(_target, _pred):
        _min_loss = jnp.inf
        idx = 0
        for _i in range(n_cells):
            _norm_pred = z_score_norm(_pred[:, _i])
            _norm_target = z_score_norm(_target)
            _mse_loss = jnp.mean((_norm_target - _norm_pred) ** 2)
            if _mse_loss < _min_loss:
                _min_loss = min(_min_loss, _mse_loss)
                idx = _i
        return _min_loss, idx


    _min_rna_l, _min_rna_idx = _calc_mse(_rna_target, _rna_pred)
    _min_prot_sh_l, _min_prot_sh_l_idx = _calc_mse(
        _prot_conc_shortlisted, _prot_conc_pred_shortlisted
    )
    _min_prot_l, _min_prot_idx = _calc_mse(
        _prot_target, _prot_pred[:, [_min_prot_sh_l_idx]]
    )

    print("Comparing steady states")
    print("Min RNA MSE loss = ", _min_rna_l)
    print("Min Protein MSE loss = ", _min_prot_l)
    print("Min Shortlisted Protein MSE loss = ", _min_prot_sh_l)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(20, 6))
    _ax1.plot(_rna_target, label="Target")
    _ax1.plot(_rna_pred[:, _min_rna_idx], label="Pred")
    _ax1.set_xlabel("RNA ID", fontsize=14)
    _ax1.set_ylabel("Standardized concentration", fontsize=14)
    _ax1.set_title("RNA concentration comparison", fontsize=18)
    _ax1.set_aspect("auto")
    _ax1.legend()

    _ax2.plot(_prot_target, label="Target")
    _ax2.plot(_prot_pred[:, _min_prot_sh_l_idx], label="Pred")
    _ax2.set_xlabel("Prot ID", fontsize=14)
    _ax2.set_ylabel("Standardized concentration", fontsize=14)
    _ax2.set_title("Protein concentration comparison", fontsize=18)
    _ax2.set_aspect("auto")
    _ax2.legend()

    # _ax2[0].plot(_prot_conc_shortlisted, label="Target")
    # _ax2[0].plot(_prot_conc_pred_shortlisted[:, _min_prot_sh_l_idx], label="Pred")
    # _ax2[0].set_xlabel("Prot ID", fontsize=14)
    # _ax2[0].set_ylabel("Standardized concentration", fontsize=14)
    # _ax2[0].set_title(
    #     "Concentration comparison of proteins with known half lives ", fontsize=18
    # )
    # _ax2[0].set_aspect("auto")
    # _ax2[0].legend()

    plt.savefig(
        "outputs/images/steady_state_comparison_range.pdf", bbox_inches="tight"
    )

    fig, _ax1 = plt.subplots(1, 1, figsize=(10, 6))
    _ax1.plot(_prot_conc_shortlisted, label="Target")
    _ax1.plot(_prot_conc_pred_shortlisted[:, _min_prot_sh_l_idx], label="Pred")
    _ax1.set_xlabel("Prot ID", fontsize=14)
    _ax1.set_ylabel("Standardized concentration", fontsize=14)
    _ax1.set_title(
        "Concentration comparison of proteins with known half lives ", fontsize=18
    )
    _ax1.set_aspect("auto")
    _ax1.legend()
    plt.savefig(
        "outputs/images/steady_state_comparison_range_2.pdf", bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(
    ids,
    jnp,
    n_cells,
    nn,
    node_names_refined,
    node_set,
    np,
    plt,
    prot_conc,
    rna_conc,
    sim,
):
    _prot_conc_target = []
    _output_prots = np.array(node_names_refined)[
        jnp.unique(jnp.where(sim.prot_steady_state != jnp.inf)[0])
    ]
    _pred_prod_ids = jnp.unique(jnp.where(sim.prot_steady_state != jnp.inf)[0])
    _output_prod_names = np.array(node_names_refined)[_pred_prod_ids]
    _target_prots_ids = [ids.to_list().index(_i) for _i in _output_prod_names]

    _output_rna_names = np.array([_i["name"] for _i in node_set])
    _target_rna_ids = [ids.to_list().index(_i) for _i in _output_rna_names]

    _rna_pred = nn.standardize(sim.steady_states.reshape(-1, n_cells), axis=1)
    _prot_pred = nn.standardize(
        sim.prot_steady_state[_pred_prod_ids].reshape(-1, n_cells), axis=1
    )

    for _i in range(len(prot_conc)):
        if ids[_i] in _output_prod_names:
            _prot_conc_target.append(prot_conc[_i])

    _prot_conc_target = np.array(_prot_conc_target)

    _prot_target = nn.standardize(_prot_conc_target)
    _rna_target = nn.standardize(rna_conc.to_numpy()[_target_rna_ids])

    _min_gene_loss = jnp.inf


    def _calc_mse(_target, _pred):
        _min_loss = jnp.inf
        idx = 0
        for _i in range(n_cells):
            _norm_pred = nn.standardize(_pred[:, _i])
            _mse_loss = jnp.mean((_target - _norm_pred) ** 2)
            if _mse_loss < _min_loss:
                _min_loss = min(_min_loss, _mse_loss)
                idx = _i
        return _min_loss, idx


    _min_rna_l, _min_rna_idx = _calc_mse(_rna_target, _rna_pred)
    _min_prot_l, _min_prot_idx = _calc_mse(_prot_target, _prot_pred)

    print("Comparing steady states")
    print("Min RNA MSE loss = ", _min_rna_l)
    print("Min Protein MSE loss = ", _min_prot_l)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(20, 6))
    _ax1.plot(_rna_target, label="Target")
    _ax1.plot(_rna_pred[:, _min_rna_idx], label="Pred")
    _ax1.set_xlabel("RNA ID", fontsize=14)
    _ax1.set_ylabel("Standardized concentration", fontsize=14)
    _ax1.set_title("RNA concentration comparison", fontsize=18)
    _ax1.set_aspect("auto")
    _ax1.legend()

    _ax2.plot(_prot_target, label="Target")
    _ax2.plot(_prot_pred[:, _min_prot_idx], label="Pred")
    _ax2.set_xlabel("Prot ID", fontsize=14)
    _ax2.set_ylabel("Standardized concentration", fontsize=14)
    _ax2.set_title("Protein concentration comparison", fontsize=18)
    _ax2.set_aspect("auto")
    _ax2.legend()

    plt.savefig("outputs/images/steady_state_comparison.pdf", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(sim):
    sim.run_sim()
    return


@app.cell
def _(
    gene_conc_1,
    ids,
    jnp,
    n_cells,
    nn,
    node_names_refined,
    node_set,
    np,
    plt,
    prot_conc,
    prot_conc_1,
    rna_conc,
):
    _prot_conc_target = []
    _output_prots = np.array(node_names_refined)[
        jnp.unique(jnp.where(prot_conc_1 != jnp.inf)[0])
    ]
    _pred_prod_ids = jnp.unique(jnp.where(prot_conc_1 > 0)[0])
    _output_prod_names = np.array(node_names_refined)[_pred_prod_ids]
    _target_prots_ids = [ids.to_list().index(_i) for _i in _output_prod_names]

    _output_rna_names = np.array([_i["name"] for _i in node_set])
    _target_rna_ids = [ids.to_list().index(_i) for _i in _output_rna_names]

    _rna_pred = nn.standardize(gene_conc_1.reshape(-1, n_cells), axis=1)
    _prot_pred = nn.standardize(
        prot_conc_1[_pred_prod_ids].reshape(-1, n_cells), axis=1
    )

    for _i in range(len(prot_conc)):
        if ids[_i] in _output_prod_names:
            _prot_conc_target.append(prot_conc[_i])

    _prot_conc_target = jnp.array(_prot_conc_target)

    _prot_target = nn.standardize(_prot_conc_target)
    _rna_target = nn.standardize(rna_conc.to_numpy()[_target_rna_ids])

    _min_gene_loss = jnp.inf


    def _calc_mse(_target, _pred):
        _min_loss = jnp.inf
        idx = 0
        for _i in range(n_cells):
            _norm_pred = nn.standardize(_pred[:, _i])
            _mse_loss = jnp.mean((_target - _norm_pred) ** 2)
            if _mse_loss < _min_loss:
                _min_loss = min(_min_loss, _mse_loss)
                idx = _i
        return _min_loss, idx


    _min_rna_l, _min_rna_idx = _calc_mse(_rna_target, _rna_pred)
    _min_prot_l, _min_prot_idx = _calc_mse(_prot_target, _prot_pred)

    print("Comparing concentrations after 10 iterations")
    print("Min RNA MSE loss = ", _min_rna_l)
    print("Min Protein MSE loss = ", _min_prot_l)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _ax1.plot(_rna_target, label="Target")
    _ax1.plot(_rna_pred[:, _min_rna_idx], label="Pred")
    _ax1.set_xlabel("RNA ID")
    _ax1.set_ylabel("Standardized concentration")
    _ax1.set_title("RNA conc plot")
    _ax1.legend()

    _ax2.plot(_prot_target, label="Target")
    _ax2.plot(_prot_pred[:, _min_prot_idx], label="Pred")
    _ax2.set_title("Prot conc plot")
    _ax2.set_xlabel("Prot ID")
    _ax2.set_ylabel("Standardized concentration")
    _ax2.legend()

    plt.savefig("outputs/images/prot_10_run_comparison.pdf")
    plt.show()
    return


@app.cell
def _(
    ids,
    jnp,
    n_cells,
    nn,
    node_names_refined,
    node_set,
    np,
    plt,
    prot_conc,
    rna_conc,
    sim,
):
    _prot_conc_target = []
    _output_prots = np.array(node_names_refined)[
        jnp.unique(jnp.where(sim.prot_conc != jnp.inf)[0])
    ]
    _pred_prod_ids = jnp.unique(jnp.where(sim.prot_conc > 0)[0])
    _output_prod_names = np.array(node_names_refined)[_pred_prod_ids]
    _target_prots_ids = [ids.to_list().index(_i) for _i in _output_prod_names]

    _output_rna_names = np.array([_i["name"] for _i in node_set])
    _target_rna_ids = [ids.to_list().index(_i) for _i in _output_rna_names]

    _rna_pred = nn.standardize(sim.gene_conc.reshape(-1, n_cells), axis=1)
    _prot_pred = nn.standardize(
        sim.prot_conc[_pred_prod_ids].reshape(-1, n_cells), axis=1
    )

    for _i in range(len(prot_conc)):
        if ids[_i] in _output_prod_names:
            _prot_conc_target.append(prot_conc[_i])

    _prot_conc_target = jnp.array(_prot_conc_target)

    _prot_target = nn.standardize(_prot_conc_target)
    _rna_target = nn.standardize(rna_conc.to_numpy()[_target_rna_ids])

    _min_gene_loss = jnp.inf


    def _calc_mse(_target, _pred):
        _min_loss = jnp.inf
        idx = 0
        for _i in range(n_cells):
            _norm_pred = nn.standardize(_pred[:, _i])
            _mse_loss = jnp.mean((_target - _norm_pred) ** 2)
            if _mse_loss < _min_loss:
                _min_loss = min(_min_loss, _mse_loss)
                idx = _i
        return _min_loss, idx


    _min_rna_l, _min_rna_idx = _calc_mse(_rna_target, _rna_pred)
    _min_prot_l, _min_prot_idx = _calc_mse(_prot_target, _prot_pred)

    print("Comparing concentrations after 100 iterations")
    print("Min RNA MSE loss = ", _min_rna_l)
    print("Min Protein MSE loss = ", _min_prot_l)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _ax1.plot(_rna_target, label="Target")
    _ax1.plot(_rna_pred[:, _min_rna_idx], label="Pred")
    _ax1.set_xlabel("RNA ID")
    _ax1.set_ylabel("Standardized concentration")
    _ax1.set_title("RNA conc plot")
    _ax1.legend()

    _ax2.plot(_prot_target, label="Target")
    _ax2.plot(_prot_pred[:, _min_prot_idx], label="Pred")
    _ax2.set_title("Prot conc plot")
    _ax2.set_xlabel("Prot ID")
    _ax2.set_ylabel("Standardized concentration")
    _ax2.legend()

    plt.savefig("outputs/images/prot_100_run_comparison.pdf")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
