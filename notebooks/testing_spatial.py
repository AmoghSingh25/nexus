import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from simulator.spatial.spatialSim import SpatialSim
    from simulator.spatial_vec.spatialSim import SpatialSimVec
    import matplotlib.pyplot as plt
    import matplotlib
    from hydra import (
        initialize_config_dir,
        compose,
    )
    import os
    import marimo as mo

    matplotlib.style.use("default")
    return (
        SpatialSim,
        SpatialSimVec,
        compose,
        initialize_config_dir,
        mo,
        os,
        plt,
    )


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="test_config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(plt):
    def plot_cell_chemicals(sim, chem_id):
        _s = sim
        _chem_data_before = _s.logger.retrieve_chem_data(step=0, chem_id=chem_id)[
            "conc"
        ]
        _chem_data_after = _s.logger.retrieve_chem_data(step=-1, chem_id=chem_id)[
            "conc"
        ]
        plt.plot(_chem_data_before, label="Before")
        plt.plot(_chem_data_after, label="After")
        plt.xticks(
            list(range(len(_chem_data_before))),
            labels=list(range(len(_chem_data_before))),
        )
        plt.legend()
        plt.xlabel("Cell ID")
        plt.ylabel("Chemical Mass")

    def plot_chemical_progress(sim, chem_id, cell_id=0):
        _s = sim
        _chem_datas = []
        for _chem_i in range(_s.mesh.n_chemicals):
            _chem_datas.append(
                _s.logger.retrieve_chem_data(chem_id=_chem_i, cell_id=cell_id)["conc"]
            )
            plt.plot(_chem_datas[-1], label=f"Chem {str(_chem_i)}")

        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Chemical Mass")

    return plot_cell_chemicals, plot_chemical_progress


@app.cell
def _():
    chem_id = 0
    return (chem_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grid Mesh
    """)
    return


@app.cell
def _(SpatialSim, get_config):
    _config = get_config()
    _config.spatial_sim.diffusion_bool = True
    _config.spatial_sim.reaction_bool = True
    _config.spatial_sim.logging = True
    s1 = SpatialSim(_config.spatial_sim)
    s1.run_sim()
    return (s1,)


@app.cell
def _(chem_id, plot_chemical_data, plt, s1):
    plot_chemical_data(s1, chem_id=chem_id)
    s1.cleanup()
    plt.title(f"chem{str(chem_id + 1)}")
    plt.show()
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_chemical_data, plt):
    _config = get_config()
    _config["spatial_sim"]["diffusion_bool"] = False
    _s = SpatialSim(_config.spatial_sim)
    _s.run_sim()

    plot_chemical_data(_s, chem_id=chem_id)
    _s.cleanup()
    plt.title(f"chem{str(chem_id + 1)}. Only reaction")
    plt.show()
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_chemical_data, plt):
    _config = get_config()
    _config["spatial_sim"]["reaction_bool"] = False
    _config["spatial_sim"]["diffusion_bool"] = True
    s2 = SpatialSim(_config.spatial_sim)
    s2.run_sim()
    plot_chemical_data(s2, chem_id=chem_id)
    s2.cleanup()
    plt.title(f"chem{str(chem_id + 1)}. Only diffusion")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Free Mesh testing
    """)
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_chemical_progress, plt):
    _cell_id = 1
    _config = get_config("freemesh_config")
    _config["spatial_sim"]["n_steps"] = 10
    _config["spatial_sim"]["reaction_bool"] = False
    s3 = SpatialSim(_config.spatial_sim)
    s3.run_sim()
    plot_chemical_progress(s3, chem_id=chem_id, cell_id=_cell_id)
    # s3.cleanup()
    plt.title(f"chem{str(chem_id + 1)} in cell {str(_cell_id)}")
    plt.show()
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_cell_chemicals, plt):
    _config = get_config("freemesh_config")
    _config["spatial_sim"]["diffusion_bool"] = False
    _s = SpatialSim(_config.spatial_sim)
    _s.run_sim()
    plot_cell_chemicals(_s, chem_id=chem_id)
    _s.cleanup()
    plt.title(f"chem{str(chem_id + 1)}. Only reaction")
    plt.show()
    return


@app.cell
def _(SpatialSim, chem_id, get_config, plot_cell_chemicals, plt):
    _config = get_config("freemesh_config")
    _config["spatial_sim"]["reaction_bool"] = False
    _config["spatial_sim"]["diffusion_bool"] = True
    _s = SpatialSim(_config.spatial_sim)
    _s.run_sim()
    plot_cell_chemicals(_s, chem_id=chem_id)
    _s.cleanup()
    plt.title(f"chem{str(chem_id + 1)}. Only diffusion")
    plt.show()
    return


@app.cell
def _(SpatialSim, get_config):
    base_config = get_config("freemesh_config")
    base_config.spatial_sim.reaction_bool = False
    base_config.spatial_sim.diffusion_bool = True

    _s = SpatialSim(base_config.spatial_sim)
    _s.run_sim()
    chem_before = _s.logger.retrieve_chem_data(step=0)["conc"]
    chem_after = _s.logger.retrieve_chem_data(step=-1)["conc"]
    _s.cleanup()
    return chem_after, chem_before


@app.cell
def _(chem_before):
    chem_before
    return


@app.cell
def _(chem_after):
    chem_after
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vectorized spatial testing
    """)
    return


@app.cell
def _(SpatialSimVec, get_config):
    _config = get_config("freemesh_config")
    _config.spatial_sim.n_steps = 10
    _config.spatial_sim.diffusion_bool = True
    _config.spatial_sim.reaction_bool = False
    _config.spatial_sim.movement_bool = True
    _config.spatial_sim.cycle_bool = True
    _config.spatial_sim.logging = True
    _config.spatial_sim.debug_plot = False
    _s1 = SpatialSimVec(_config.spatial_sim)
    _s1.run_sim()
    print(_s1.mesh.field_positions)
    _s1.cleanup()
    return


@app.cell
def _(SpatialSimVec, get_config, plot_cell_chemicals, plt):
    _config = get_config("freemesh_config")
    _config.spatial_sim.diffusion_bool = True
    _config.spatial_sim.reaction_bool = False
    _config.spatial_sim.reaction_prob = False
    # _config.spatial_sim.logging = False
    _s1 = SpatialSimVec(_config.spatial_sim)
    _s1.run_sim()

    _chemid = 2
    plot_cell_chemicals(_s1, chem_id=_chemid)
    _s1.cleanup()
    plt.title(f"chem{str(_chemid + 1)}")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
