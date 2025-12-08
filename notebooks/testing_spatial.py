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
    import marimo as mo

    matplotlib.style.use("default")
    return SpatialSim, compose, initialize_config_dir, jnp, mo, plt


@app.cell
def _(compose, initialize_config_dir):
    import os

    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="config")
    return (cfg,)


@app.cell
def _(cfg):
    len(cfg["spatial_sim"]["chemical"].name)
    return


@app.cell
def _(SpatialSim, cfg):
    s = SpatialSim(cfg)
    s.run_sim()
    return (s,)


@app.cell
def _(s):
    s.logger.retrieve_chem_data(0)
    return


@app.cell
def _(s):
    s.logger.retrieve_chem_data(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chemical 0
    """)
    return


@app.cell
def _(SpatialSim, cfg, jnp, plt):
    _s = SpatialSim(cfg)
    _ret1, _ret2 = _s.run_sim(10)

    _a = jnp.array(_ret1)
    _b = jnp.array(_ret2)

    plt.plot(_a[:, 0].reshape(-1), label="Before")
    plt.plot(_b[:, 0].reshape(-1), label="After")
    plt.title("Without reaction, only diffusion")
    plt.legend()
    plt.show()
    return


@app.cell
def _(SpatialSim, cfg, jnp, plt):
    cfg["spatial_sim"]["reaction_bool"] = False
    _s = SpatialSim(cfg)
    _ret1, _ret2 = _s.run_sim(10)

    _a = jnp.array(_ret1)
    _b = jnp.array(_ret2)

    plt.plot(_a[:, 0].reshape(-1), label="Before")
    plt.plot(_b[:, 0].reshape(-1), label="After")
    plt.title("Without reaction, only diffusion")
    plt.legend()
    plt.show()
    return


@app.cell
def _(SpatialSim, cfg, jnp, plt):
    cfg["spatial_sim"]["reaction_bool"] = True
    cfg["spatial_sim"]["diffusion_bool"] = False
    _s = SpatialSim(cfg)
    _ret1, _ret2 = _s.run_sim(10)

    _a = jnp.array(_ret1)
    _b = jnp.array(_ret2)

    plt.plot(_a[:, 0].reshape(-1), label="Before")
    plt.plot(_b[:, 0].reshape(-1), label="After")
    plt.title("Without diffusion, only reaction")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
