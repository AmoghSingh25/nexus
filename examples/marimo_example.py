import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from nexus.simulator.run_linked_sim import run_sim
    from nexus.simulator.grn.grnSim import GRNSim
    from nexus.simulator.spatial.spatialSim import SpatialSim
    from nexus.simulator.utils.file_manager import _read_file
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    import pprint
    import yaml
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    return (
        GRNSim,
        OmegaConf,
        SpatialSim,
        compose,
        initialize_config_dir,
        jnp,
        mo,
        os,
        plt,
        pprint,
        run_sim,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading config
    """)
    return


@app.cell
def _(compose, initialize_config_dir, os):
    def get_config(config_name="test_config"):
        conf_path = os.path.join(os.getcwd(), "configs")
        with initialize_config_dir(version_base=None, config_dir=conf_path):
            cfg = compose(config_name=config_name)
        return cfg

    return (get_config,)


@app.cell
def _(get_config):
    cfg = get_config("grn_config")
    return (cfg,)


@app.cell
def _(OmegaConf, cfg):
    print(OmegaConf.to_yaml(cfg=cfg.grn))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Running GRN Sim
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## RNA Sim using SERGIO input files
    """)
    return


@app.cell
def _(GRNSim, cfg):
    grn_sim = GRNSim(cfg=cfg.grn)
    return (grn_sim,)


@app.cell
def _(grn_sim, jnp):
    rna_conc, prot_conc = grn_sim.run_sim()
    rna_conc = jnp.array(rna_conc)
    prot_conc = jnp.array(prot_conc)
    return prot_conc, rna_conc


@app.cell
def _(prot_conc, rna_conc):
    print(rna_conc.shape)  ## RNA concentration after running simulator (No. of steps, No. of RNAs, No. of Cells, 1)
    print(prot_conc.shape) ## Protein concentration after running simulator (No. of steps, No. of Proteins, No. of Cells, 1)
    return


@app.cell
def _(plt, rna_conc):
    ## Plot concentration of RNAs in cell 0
    fig = plt.figure(figsize=(12, 6))
    _ticks = list(range(rna_conc.shape[0]))
    plt.plot(rna_conc[-1, :, 0])
    plt.ylabel("Conc.")
    plt.xlabel("RNA Idx")
    plt.title("Concentration of RNAs in cell 0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running RNA and protein sim

    Uses a custom config for defining the RNA and Protein parameters
    """)
    return


@app.cell
def _(pprint, yaml):
    with open("sample_data/sample_network_1cell.yaml", "r") as file:
        node_set, edge_set = yaml.safe_load(file)
    print("RNA and protein configuration")
    pprint.pprint(node_set)

    print("Network configuration")
    pprint.pprint(edge_set)
    return


@app.cell
def _(get_config):
    cfg2 = get_config("grn_prot_config")
    return (cfg2,)


@app.cell
def _(OmegaConf, cfg2):
    print(OmegaConf.to_yaml(cfg=cfg2.grn))
    return


@app.cell
def _(GRNSim, cfg2):
    grn_sim2 = GRNSim(cfg=cfg2.grn)
    return (grn_sim2,)


@app.cell
def _(grn_sim2, jnp):
    rna_conc2, prot_conc2 = grn_sim2.run_sim()
    rna_conc2 = jnp.array(rna_conc2)
    prot_conc2 = jnp.array(prot_conc2)
    return prot_conc2, rna_conc2


@app.cell
def _(plt, prot_conc2, rna_conc2):
    ## Plot concentration of RNAs in cell 0
    _fig = plt.figure(figsize=(12, 6))
    _subplots = _fig.subfigures(1, 2)

    ax1 = _subplots[0].subplots()
    _ticks = list(range(rna_conc2.shape[1]))
    ax1.plot(rna_conc2[-1, :, 0])
    ax1.set_ylabel("Conc.")
    ax1.set_xlabel("RNA Idx")
    ax1.set_title("Concentration of RNAs in cell 0")

    ax2 = _subplots[1].subplots()
    ax2.plot(prot_conc2[-1, :, 0])
    ax2.set_ylabel("Conc.")
    ax2.set_xlabel("Prot Idx")
    ax2.set_title("Concentration of Protein in cell 0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Spatial Sim
    """)
    return


@app.cell
def _(OmegaConf, get_config):
    cfg3 = get_config("spatial_config")
    print(OmegaConf.to_yaml(cfg3.spatial_sim))
    return (cfg3,)


@app.cell
def _(SpatialSim, cfg3):
    spatial_sim = SpatialSim(cfg3.spatial_sim)
    return (spatial_sim,)


@app.cell
def _(spatial_sim):
    spatial_sim.run_sim()
    return


@app.cell
def _(plt, spatial_sim):
    plt.figure(figsize=(12, 6))
    _data = spatial_sim.logger.retrieve_chem_data(field_id=1)
    _chem_data = []
    for _i in range(len(set(_data["chem"]))):
        _idx = _data["chem"] == _i
        _chem_data.append(_data["conc"][_idx])
        plt.plot(_data["conc"][_idx], label=f"Chem - {str(_i)}")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Conc.")
    plt.title("Concentration of chemicals in Field 1 across timesteps")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Linked Sim - Running spatial and GRN sim together
    """)
    return


@app.cell
def _(get_config):
    cfg4 = get_config("cell_field_config")
    return (cfg4,)


@app.cell
def _(cfg4, run_sim):
    run_sim(cfg4)
    return


@app.cell
def _():
    field_conc = _read_file("outputs/field_concs.pkl")
    return (field_conc,)


@app.cell
def _(field_conc, plt):
    _field = 2
    _chem_conc = [[], [], [], []]
    for i in field_conc:
        for j in range(field_conc[0].shape[1]):
            _chem_conc[j].append(i[_field][j])
    for _i in range(len(_chem_conc)):
        plt.plot(_chem_conc[_i], label=f"Chem - {_i}")
    _ticks = list(range(0, len(field_conc), 2))
    plt.legend()
    plt.title(f"Conc. plot of chemicals in field {str(_field)}")
    plt.xlabel("Timestep")
    plt.ylabel("Conc")
    plt.xticks(_ticks)
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
