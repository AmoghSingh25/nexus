import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax.numpy as jnp
    from nexus.simulator.grn.grnSim import GRNSim
    import os
    from hydra import initialize_config_dir, compose

    return GRNSim, compose, initialize_config_dir, jnp, mo, os


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
    cfg = get_config()
    cfg.grn.config_file = "configs/sample_data/sample_network_self_loop.yaml"
    return (cfg,)


@app.cell
def _(GRNSim, cfg, jnp):
    sim = GRNSim(cfg.grn)

    ss = sim.calc_steady_states()
    assert not jnp.isnan(ss[0]).any(), "Gene values contain NaNs"
    assert not jnp.isnan(ss[1]).any(), "Protein values contain NaNs"

    sim.run_sim()
    _diff_gene = jnp.abs(sim.gene_conc - ss[0])
    _diff_prot = jnp.abs(sim.prot_conc - ss[1])
    assert jnp.all(_diff_gene < 1e-3), "Gene conc. diverge after steady state calculation in self-loop"
    assert jnp.all(_diff_prot < 1e-3), "Prot conc. diverge after steady state calculation in self-loop"
    return (ss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Perturb genes and re-run sim
    """)
    return


@app.cell
def _(cfg):
    cfg
    return


@app.cell
def _(GRNSim, cfg, ss):
    cfg2 = cfg
    cfg2.grn.n_steps = 1000
    cfg2.grn.delta = 0.01

    sim2 = GRNSim(cfg2.grn)
    sim2.gene_conc = ss[0]*0.1
    sim2.prot_conc = ss[1]*0.1
    _ret = sim2.run_sim()
    return (sim2,)


@app.cell
def _(jnp, sim2, ss):
    _gene_diff = sim2.gene_conc - ss[0]
    _prot_diff = sim2.prot_conc - ss[1]
    _rel_gene_diff = jnp.max((jnp.abs(sim2.gene_conc - ss[0]) / jnp.max(ss[0])) * 100)
    _rel_prot_diff = jnp.max((jnp.abs(sim2.prot_conc - ss[1]) / jnp.max(ss[1]))*100)

    assert jnp.all(jnp.abs(sim2.gene_conc - ss[0])<1e-3), "Absolute diff of gene concs do not re-stabilize"
    assert jnp.all(_rel_gene_diff < 0.1), "Relative diff of gene concs do not re-stabilize"
    assert jnp.all(jnp.abs(sim2.prot_conc - ss[1])<1e-3), "Absolute diff of protein concs do not re-stabilize"
    assert jnp.all(_rel_prot_diff<0.1), "Relative diff of protein concs do not re-stabilize"
    return


@app.cell
def _(jnp, sim2, ss):
    jnp.abs(sim2.gene_conc - ss[0])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
