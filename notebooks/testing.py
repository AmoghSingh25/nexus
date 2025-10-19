import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.style.use("default")
    return np, plt


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
def _():
    node_set_old = [
        {"basal_rate": [0.20165], "type": "mr"},
        {"basal_rate": [0.25316], "type": "mr"},
        {
            "ki": [1.73104, 2.64137],
            "prot_half_life": [3],
            "prot_transcription_rate": [10],
            "type": "g",
        },
        {
            "ki": [2.34309],
            "prot_half_life": [2],
            "prot_transcription_rate": [12],
            "type": "g",
        },
    ]
    return (node_set_old,)


@app.cell
def _():
    node_set = [
        {"basal_rate": [0.20165, 0.19165], "type": "mr"},
        {"basal_rate": [0.25316, 0.21132], "type": "mr"},
        {
            "ki": [
                [1, 2],  # Cell 1
                [3, 4],
            ],  # Cell 2
            "prot_half_life": [3, 4],
            "prot_transcription_rate": [10, 11],
            "type": "g",
        },
        {
            "ki": [[2.34309], [1.64324]],
            "prot_half_life": [2, 1],
            "prot_transcription_rate": [12, 13],
            "type": "g",
        },
    ]

    edges_set = [[0, 2], [1, 2], [2, 3]]
    return edges_set, node_set


@app.cell
def _(gpsim, plot_conc):
    _sim = gpsim.simulator(
        config_file="configs/sample_data/sample_network_2cell.yaml", n_cells=2
    )
    _a, _b = _sim.run_sim(10)

    plot_conc(_a, 0)
    plot_conc(_b, 0)
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
