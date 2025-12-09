import marimo

__generated_with = "0.18.4"
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
    import os

    matplotlib.style.use("default")
    return SpatialSim, compose, initialize_config_dir, os, plt


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(plt, s1):
    def plot_chemical_data(sim, chem_id):
        _s = s1
        _chem_data_before = _s.logger.retrieve_chem_data(step=0, chem_id=chem_id)[
            "conc"
        ]
        _chem_data_after = _s.logger.retrieve_chem_data(step=-1, chem_id=chem_id)[
            "conc"
        ]
        plt.plot(_chem_data_before, label="Before")
        plt.plot(_chem_data_after, label="After")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Chemical Mass")

    return (plot_chemical_data,)


@app.cell
def _():
    chem_id = 1
    return (chem_id,)


@app.cell
def _(SpatialSim, get_config):
    _config = get_config()
    s1 = SpatialSim(_config)
    s1.run_sim()
    return (s1,)


@app.cell
def _(chem_id, plot_chemical_data, plt, s1):
    plot_chemical_data(s1, chem_id=chem_id)
    plt.title(f"Chemical - {str(chem_id)}")
    plt.show()
    return


@app.cell
def _(SpatialSim, get_config):
    _config = get_config()
    _config["spatial_sim"]["reaction_bool"] = False
    s2 = SpatialSim(_config)
    s2.run_sim()
    return (s2,)


@app.cell
def _(chem_id, plot_chemical_data, plt, s2):
    plot_chemical_data(s2, 2)
    plt.title(f"Chemical - {str(chem_id)}. No Reaction")
    plt.show()
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_chemical_data, plt):
    _config = get_config()
    _config["spatial_sim"]["diffusion_bool"] = False
    s3 = SpatialSim(_config)
    s3.run_sim()

    plot_chemical_data(s3, chem_id=chem_id)
    plt.title(f"Chemical - {str(chem_id)}. No diffusion")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
