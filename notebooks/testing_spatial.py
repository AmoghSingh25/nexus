import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from simulator.spatial.spatialSim import SpatialSim
    import matplotlib.pyplot as plt
    import matplotlib
    from hydra import (
        initialize_config_dir,
        compose,
    )
    import jax.numpy as jnp

    matplotlib.style.use("ggplot")
    return SpatialSim, compose, initialize_config_dir, jnp, plt


@app.cell
def _(compose, initialize_config_dir):
    import os

    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="config")
    return (cfg,)


@app.cell
def _(cfg):
    cfg["spatial_sim"]["reaction"]
    return


@app.cell
def _(SpatialSim, cfg):
    s = SpatialSim(cfg)
    return (s,)


@app.cell
def _(s):
    s.mesh.cells[0, 0, 0].chem.calc_reaction_change()
    return


app._unparsable_cell(
    r"""
    |ret1, ret2 = s.run_sim(10)
    """,
    name="_",
)


@app.cell
def _(jnp, plt, ret1, ret2):
    _a = jnp.array(ret1)
    _b = jnp.array(ret2)

    plt.plot(_a[:, 0].reshape(-1), label="Before")
    plt.plot(_b[:, 0].reshape(-1), label="After")
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
