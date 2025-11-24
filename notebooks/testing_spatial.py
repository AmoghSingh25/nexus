import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from simulator.spatial.main import SpatialSim
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("ggplot")
    return SpatialSim, plt


@app.cell
def _(SpatialSim):
    s = SpatialSim(delta=0.1)
    return (s,)


@app.cell
def _(s):
    masses_b = []
    for _i in s.cells.reshape(-1):
        masses_b.append(_i.chemical.mass)
    return (masses_b,)


@app.cell
def _(s):
    for _i in s.cells.reshape(-1):
        s.calc_conc_change(_i.pos)
    return


@app.cell
def _(s):
    masses_a = []
    for _i in s.cells.reshape(-1):
        masses_a.append(_i.chemical.mass)
    return (masses_a,)


@app.cell
def _(masses_a, masses_b, plt):
    plt.plot(masses_b, label="before")
    plt.plot(masses_a, label="after")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
