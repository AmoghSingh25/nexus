import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    from simulator import gpsim

    return (gpsim,)


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
def _(edges_set, gpsim, node_set_old):
    sim = gpsim.simulator(node_set_old, edges_set, n_cells=1)
    a2, b2 = sim.run_sim(10)
    return (a2,)


@app.cell
def _(edges_set, gpsim, node_set):
    sim2 = gpsim.simulator(node_set, edges_set, n_cells=2)
    a, b = sim2.run_sim(10)
    return a, b


@app.cell
def _(b, np, plt):
    _a = np.array(b)
    _cell_no = 0
    for _i in range(_a[:, :, _cell_no].shape[1]):
        plt.plot(_a[:, _i, _cell_no], label=f"Node {[_i]}")
    plt.title("Concentrations of proteins of a single cell")
    plt.xlabel("Steps")
    plt.ylabel("Expression value")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(a2, np, plt):
    _a = np.array(a2)
    _cell_no = 0
    for _i in range(_a[:, :, _cell_no].shape[1]):
        plt.plot(_a[:, _i, _cell_no], label=f"Node {[_i]}")
    plt.title("Expression values of a single cell")
    plt.xlabel("Steps")
    plt.ylabel("Expression value")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(a, np, plt):
    _a = np.array(a)
    _cell_no = 0
    for _i in range(_a[:, :, _cell_no].shape[1]):
        plt.plot(_a[:, _i, _cell_no], label=f"Node {[_i]}")
    plt.title("Expression values of a single cell")
    plt.xlabel("Steps")
    plt.ylabel("Expression value")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
