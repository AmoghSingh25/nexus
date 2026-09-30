import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from hydra import initialize_config_dir, compose
    from nexus.simulator.grn.grnSim import GRNSim
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    import matplotlib
    import jax

    matplotlib.style.use("default")
    return GRNSim, OmegaConf, compose, initialize_config_dir, jnp, mo, os, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Repressilator without learning
    """)
    return


@app.cell
def _(compose, initialize_config_dir, os):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="repressilator_config")

    print("Configuration loaded. Steps:", cfg.grn.n_steps)
    return (cfg,)


@app.cell
def _(OmegaConf, cfg):
    print(OmegaConf.to_yaml(cfg))
    return


@app.cell
def _(GRNSim, cfg, jnp):
    init_conc = jnp.array(
        [
            [[3.0], [0.1], [0.1]],  # G0
            [[0.1], [3.0], [0.1]],  # G1
            [[0.1], [0.1], [3.0]],  # G2
        ]
    )


    ## high k_i and h = 5.0
    sim_sustained = GRNSim(cfg.grn)
    sim_sustained.gene_conc = init_conc
    sim_sustained.steady_states = init_conc

    ## h = 3.0
    sim_damped = GRNSim(cfg.grn)
    sim_damped.gene_conc = init_conc
    sim_damped.steady_states = init_conc
    sim_damped.hill_coeffs = sim_damped.hill_coeffs.at[:].set(3.0)
    return sim_damped, sim_sustained


@app.cell
def _(sim_damped, sim_sustained):
    out_sustained = sim_sustained.run_sim()
    traj_sustained = out_sustained.gene_traj.squeeze(-1)

    out_damped = sim_damped.run_sim()
    traj_damped = out_damped.gene_traj.squeeze(-1)

    print("Simulations completed.")
    return traj_damped, traj_sustained


@app.cell
def _(plt, traj_damped, traj_sustained):
    fig_timeseries, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    steps = range(traj_sustained.shape[0])

    # Plot Sustained (Cell 0)
    axes[0].plot(steps, traj_sustained[:, 0, 0], color="#FF5733", lw=2, label="G0")
    axes[0].plot(steps, traj_sustained[:, 1, 0], color="#33FF57", lw=2, label="G1")
    axes[0].plot(steps, traj_sustained[:, 2, 0], color="#3357FF", lw=2, label="G2")
    axes[0].set_title(
        "Continuous Oscillations (High Hill coeff. h = 5.0)", fontsize=12
    )
    axes[0].set_ylabel("Concentration")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Plot Damped (Cell 0)
    axes[1].plot(steps, traj_damped[:, 0, 0], color="#FF5733", lw=2, label="G0")
    axes[1].plot(steps, traj_damped[:, 1, 0], color="#33FF57", lw=2, label="G1")
    axes[1].plot(steps, traj_damped[:, 2, 0], color="#3357FF", lw=2, label="G2")
    axes[1].set_title("Damped Oscillations (Low Hill coeff. h = 2.0)", fontsize=12)
    axes[1].set_ylabel("Concentration")
    axes[1].set_xlabel("Time step")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig_timeseries
    return


@app.cell
def _(plt, traj_damped, traj_sustained):
    fig_phase = plt.figure(figsize=(14, 6))

    # Sustained Limit Cycle
    ax1 = fig_phase.add_subplot(121, projection="3d")
    ax1.plot(
        traj_sustained[:, 0, 0],
        traj_sustained[:, 1, 0],
        traj_sustained[:, 2, 0],
        color="purple",
        lw=2,
    )
    ax1.set_xlabel("G0")
    ax1.set_ylabel("G1")
    ax1.set_zlabel("G2")
    ax1.set_title("Sustained: Stable Limit Cycle")
    ax1.grid(True)

    # Damped Spiral
    ax2 = fig_phase.add_subplot(122, projection="3d")
    ax2.plot(
        traj_damped[:, 0, 0],
        traj_damped[:, 1, 0],
        traj_damped[:, 2, 0],
        color="blue",
        lw=2,
    )
    ax2.scatter(
        traj_damped[0, 0, 0],
        traj_damped[0, 1, 0],
        traj_damped[0, 2, 0],
        color="red",
        s=50,
        label="Start",
    )
    ax2.scatter(
        traj_damped[-1, 0, 0],
        traj_damped[-1, 1, 0],
        traj_damped[-1, 2, 0],
        color="green",
        s=50,
        label="End (Steady State)",
    )
    ax2.set_xlabel("G0")
    ax2.set_ylabel("G1")
    ax2.set_zlabel("G2")
    ax2.set_title("Damped: Stable Point Attractor (Spiral)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig_phase
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
