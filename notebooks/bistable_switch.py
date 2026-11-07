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
    import matplotlib

    matplotlib.style.use("default")
    return GRNSim, compose, initialize_config_dir, jnp, mo, os, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Bistable switch
    Have the same parameter set, but each gene reaches different final concentrations based on the initial concentration values, while mutually repressed.
    """)
    return


@app.cell
def _(compose, initialize_config_dir, os):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="bistable_config")
    return (cfg,)


@app.cell
def _(GRNSim, cfg):
    sim = GRNSim(cfg.grn)
    sim.ki_matrix = sim.ki_matrix.at[sim.ki_matrix != 0].set(-5.0)
    print(
        "Cell 0 Decay:",
        sim.decay[0, :, 0].tolist(),
        "; Cell 1 Decay:",
        sim.decay[1, :, 0].tolist(),
    )
    print(
        "Cell 0 Hill Coeffs:",
        sim.hill_coeffs[0, :, 0].tolist(),
        "; Cell 1 Hill Coeffs:",
        sim.hill_coeffs[1, :, 0].tolist(),
    )
    print("Cell 0 ki_matrix:\n", sim.ki_matrix[0].tolist())
    print("Cell 1 ki_matrix:\n", sim.ki_matrix[1].tolist())
    return (sim,)


@app.cell
def _(mo):
    mo.md("""
    ## Run Simulation
    """)
    return


@app.cell
def _(jnp, sim):
    # Cell 0: G0 high, G1 low
    # Cell 1: G0 low, G1 high
    init_conc = jnp.array(
        [
            [[3.0], [1.0]],  # G0
            [[1.0], [3.0]],  # G1
        ]
    )

    sim.gene_conc = init_conc
    sim.steady_states = init_conc

    output = sim.run_sim()
    gene_traj = output.gene_traj.squeeze(-1)
    return (gene_traj,)


@app.cell
def _(mo):
    mo.md("""
    ## Plots
    """)
    return


@app.cell
def _(gene_traj, plt):
    fig_traj, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps_range = range(gene_traj.shape[0])

    # Cell 0
    axes[0].plot(steps_range, gene_traj[:, 0, 0], label="G0", color="red", lw=2)
    axes[0].plot(steps_range, gene_traj[:, 1, 0], label="G1", color="blue", lw=2)
    axes[0].set_title("Cell 0 Trajectory (Converges to G0 High)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Concentration")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Cell 1
    axes[1].plot(steps_range, gene_traj[:, 0, 1], label="G0", color="red", lw=2)
    axes[1].plot(steps_range, gene_traj[:, 1, 1], label="G1", color="blue", lw=2)
    axes[1].set_title("Cell 1 Trajectory (Converges to G1 High)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Concentration")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig_traj.suptitle(
        "Bistable switch state change (Without learning)", fontsize=14
    )
    plt.tight_layout()
    fig_traj
    return


@app.cell
def _(mo):
    mo.md("""
 
    """)
    return


@app.cell
def _(gene_traj, plt):
    fig_phase = plt.figure(figsize=(6, 6))

    # Cell 0 trajectory
    plt.plot(
        gene_traj[:, 0, 0],
        gene_traj[:, 1, 0],
        label="Cell 0 Trajectory",
        color="darkred",
        lw=2,
    )
    plt.scatter(
        gene_traj[0, 0, 0],
        gene_traj[0, 1, 0],
        color="red",
        marker="o",
        s=80,
        label="Cell 0 Start",
    )
    plt.scatter(
        gene_traj[-1, 0, 0],
        gene_traj[-1, 1, 0],
        color="red",
        marker="x",
        s=100,
        label="Cell 0 Attractor",
    )

    # Cell 1 trajectory
    plt.plot(
        gene_traj[:, 0, 1],
        gene_traj[:, 1, 1],
        label="Cell 1 Trajectory",
        color="darkblue",
        lw=2,
    )
    plt.scatter(
        gene_traj[0, 0, 1],
        gene_traj[0, 1, 1],
        color="blue",
        marker="o",
        s=80,
        label="Cell 1 Start",
    )
    plt.scatter(
        gene_traj[-1, 0, 1],
        gene_traj[-1, 1, 1],
        color="blue",
        marker="x",
        s=100,
        label="Cell 1 Attractor",
    )

    plt.xlabel("G0 concentration")
    plt.ylabel("G1 concentration")
    plt.title("Bistable Switch Phase Portrait (Shared Parameters)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_phase
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
