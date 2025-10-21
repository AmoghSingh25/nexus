import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    matplotlib.style.use("default")
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.WARN)
    logging.debug("test")
    return np, plt, time


@app.cell
def _():
    from simulator import gpsim

    return (gpsim,)


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
    ## Without JIT
    _start = time.time()
    for i in range(10):
        _sim = gpsim.simulator(
            config_file="configs/sample_data/sample_network_2cell.yaml",
            n_cells=2,
            protein_sim=False,
        )
        _a, _b = _sim.run_sim(10)
    _end = time.time()

    print("Running time for simulator = ", (_end - _start) / 10)

    # plot_conc(_a, 0)
    # plot_conc(_b, 0)
    return


@app.cell
def _(gpsim, time):
    ## With JIT
    _start = time.time()
    for _ in range(10):
        _sim = gpsim.simulator(
            config_file="configs/sample_data/sample_network_1cell.yaml",
            n_cells=1,
            protein_sim=True,
        )
        _a, _b = _sim.run_sim(10)
    _end = time.time()

    print("Running time for simulator = ", (_end - _start) / 10)

    # plot_conc(_a, 0)
    # plot_conc(_b, 0)
    return


@app.cell
def _(a):
    a
    return


@app.cell
def _(gpsim):
    sim = gpsim.simulator(
        "configs/sample_data/Interaction_cID_4.txt",
        "configs/sample_data/Regs_cID_4.txt",
    )
    # a2, b2 = sim.run_sim(10)
    return (sim,)


@app.cell
def _(node_set_old, sim):
    node_set_old
    sim.run_sim(10)
    return


@app.cell
def _(edges_set, gpsim, node_set):
    sim2 = gpsim.simulator(node_set, edges_set, n_cells=2)
    a, b = sim2.run_sim(10)
    return (a,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
