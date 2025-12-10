import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time
    import jax.numpy as jnp
    from hydra import initialize_config_dir, compose
    import os
    from simulator import grnSim

    matplotlib.style.use("default")
    import logging

    logger = logging.getLogger("sample_logger")
    logger.setLevel(logging.ERROR)
    return compose, grnSim, initialize_config_dir, jnp, np, os, plt, time


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(get_config):
    cfg = get_config(config_name="test_config")
    return (cfg,)


@app.cell
def _(cfg, grnSim):
    _s = grnSim.GRNSim(cfg=cfg.grn)
    _s.run_sim()
    return


@app.cell
def _(np, plt):
    def plot_conc(vals, cell_no):
        vals = np.array(vals)
        for _i in range(vals[:, :, cell_no].shape[1]):
            plt.plot(vals[:, _i, cell_no], label=f"Node {[_i]}")
        plt.title("Expression values of a single cell")
        plt.xlabel("Steps")
        plt.ylabel("Expression value")
        plt.legend()
        plt.grid(True)
        plt.show()

    return


@app.cell
def _(gpsim, time):
    n_cells_1 = 2700
    n_genes_1 = 100
    _start = time.time()
    sim1 = gpsim.simulator(
        gene_data="configs/sample_data/Interaction_cID_4.txt",
        mr_data="configs/sample_data/Regs_cID_4.txt",
        n_cells=n_cells_1,
        protein_sim=False,
        decay=[0.8],
        noise=False,
    )
    _start2 = time.time()
    # a, b = sim1.run_sim(1)
    _end = time.time()
    print("Time taken = ", _end - _start)
    print("Steady state calculation time = ", _start2 - _start)
    print("Simulation time = ", _end - _start2)
    return n_cells_1, n_genes_1, sim1


@app.cell
def _():
    node_mapping = {
        67: 0,
        62: 1,
        17: 2,
        56: 3,
        44: 4,
        84: 5,
        93: 6,
        74: 7,
        1: 8,
        14: 9,
        0: 10,
        2: 11,
        3: 12,
        4: 13,
        5: 14,
        6: 15,
        7: 16,
        8: 17,
        9: 18,
        10: 19,
        11: 20,
        12: 21,
        13: 22,
        15: 23,
        16: 24,
        18: 25,
        19: 26,
        20: 27,
        21: 28,
        22: 29,
        23: 30,
        24: 31,
        25: 32,
        26: 33,
        27: 34,
        28: 35,
        29: 36,
        30: 37,
        31: 38,
        32: 39,
        33: 40,
        34: 41,
        35: 42,
        36: 43,
        37: 44,
        38: 45,
        39: 46,
        40: 47,
        41: 48,
        42: 49,
        43: 50,
        45: 51,
        46: 52,
        47: 53,
        48: 54,
        49: 55,
        50: 56,
        51: 57,
        52: 58,
        53: 59,
        54: 60,
        55: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 66,
        63: 67,
        64: 68,
        65: 69,
        66: 70,
        68: 71,
        69: 72,
        70: 73,
        71: 74,
        72: 75,
        73: 76,
        75: 77,
        76: 78,
        77: 79,
        78: 80,
        79: 81,
        80: 82,
        81: 83,
        82: 84,
        83: 85,
        85: 86,
        86: 87,
        87: 88,
        88: 89,
        89: 90,
        90: 91,
        91: 92,
        92: 93,
        94: 94,
        95: 95,
        96: 96,
        97: 97,
        98: 98,
        99: 99,
    }
    return (node_mapping,)


@app.cell
def _():
    import pickle

    with open("src/tests/saved_outputs/saved_output.pkl", "rb") as f:
        sergio_output_1 = pickle.load(f)
    with open("src/tests/saved_outputs/saved_output_9cells.pkl", "rb") as f:
        sergio_output_2 = pickle.load(f)
    return sergio_output_1, sergio_output_2


@app.cell
def _(jnp, n_cells_1, n_genes_1, node_mapping, sim1):
    sim_output_1 = sim1.gene_conc.reshape((n_genes_1, n_cells_1))
    reordered_output_1 = jnp.zeros_like(sim_output_1)
    for i in node_mapping:
        reordered_output_1 = reordered_output_1.at[i].set(sim_output_1[node_mapping[i]])
    return (reordered_output_1,)


@app.cell
def _(jnp, reordered_output_1, sergio_output_1):
    print(jnp.allclose(reordered_output_1, sergio_output_1, rtol=1e-6, atol=1e-32))
    return


@app.cell
def _(gpsim, time):
    n_cells_2 = 9
    n_genes_2 = 100
    _start = time.time()
    sim3 = gpsim.simulator(
        gene_data="configs/sample_data/Interaction_cID_4.txt",
        mr_data="configs/sample_data/Regs_cID_4.txt",
        n_cells=n_cells_2,
        protein_sim=False,
    )
    _start2 = time.time()
    _a, _b = sim3.run_sim(1)
    _end = time.time()
    print("Total time taken = ", _end - _start)
    print("Steady state calculation time = ", _start2 - _start)
    print("Simulation time = ", _end - _start2)
    return n_cells_2, n_genes_2, sim3


@app.cell
def _(jnp, n_cells_2, n_genes_2, node_mapping, sim3):
    sim_output_2 = sim3.steady_states.reshape((n_genes_2, n_cells_2))
    reordered_output_2 = jnp.zeros_like(sim_output_2)
    for _i in node_mapping:
        reordered_output_2 = reordered_output_2.at[_i].set(
            sim_output_2[node_mapping[_i]]
        )
    return (reordered_output_2,)


@app.cell
def _(jnp, reordered_output_2, sergio_output_2):
    print(jnp.allclose(reordered_output_2, sergio_output_2, rtol=1e-6, atol=1e-32))
    return


if __name__ == "__main__":
    app.run()
