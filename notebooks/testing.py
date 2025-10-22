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

    return (plot_conc,)


@app.cell
def _(gpsim, time):
    ## With JIT
    _start = time.time()
    for _ in range(10):
        _sim = gpsim.simulator(
            config_file="configs/sample_data/sample_network_1cell.yaml",
            n_cells=1,
            protein_sim=False,
        )
        _a, _b = _sim.run_sim(10)
    _end = time.time()

    print("Running time for simulator = ", (_end - _start) / 10)

    # plot_conc(_a, 0)
    # plot_conc(_b, 0)
    return


@app.cell
def _(gpsim, plot_conc):
    sim = gpsim.simulator(
        config_file="configs/sample_data/sample_network_1cell.yaml",
        n_cells=1,
        protein_sim=False,
    )
    _a, _b = sim.run_sim(10)

    plot_conc(_a, 0)
    plot_conc(_b, 0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
