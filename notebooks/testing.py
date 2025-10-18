import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


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
            "transcription_rate": [10],
            "type": "g",
        },
        {
            "ki": [2.34309],
            "prot_half_life": [2],
            "transcription_rate": [10],
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
            "prot_half_life": [3],
            "transcription_rate": [10],
            "type": "g",
        },
        {
            "ki": [[2.34309], [1.64324]],
            "prot_half_life": [2],
            "transcription_rate": [10],
            "type": "g",
        },
    ]

    edges_set = [[0, 2], [1, 2], [2, 3]]
    return edges_set, node_set


@app.cell
def _(edges_set, gpsim, node_set_old):
    sim = gpsim.simulator(node_set_old, edges_set, n_cells=1)
    a, b = sim.run_sim(10)
    return


@app.cell
def _(edges_set, gpsim, node_set):
    sim = gpsim.simulator(node_set, edges_set, n_cells=2)
    a, b = sim.run_sim(10)
    return


@app.cell
def _(p_vals_, plt):
    for _i in range(p_vals_.shape[1]):
        plt.plot(p_vals_[:, _i], label=f"Node {[_i]}")
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
