import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from hydra import initialize_config_dir, compose
    from nexus.simulator.grn.grnSim import GRNSim
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, lax, nn
    import optax
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("default")
    return (
        GRNSim,
        compose,
        initialize_config_dir,
        jax,
        jit,
        jnp,
        lax,
        mo,
        optax,
        os,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Learning one parameter set for all cells for bistable switch.
    """)
    return


@app.cell
def _(compose, initialize_config_dir, os):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="bistable_config")
    return (cfg,)


@app.cell
def _(GRNSim, cfg, jnp, lax):
    sim = GRNSim(cfg.grn)


    def run_rollout(params, sim_instance, init_gene_conc, steps):
        n_cells = sim_instance.n_cells

        basal_rates_cell = jnp.repeat(
            jnp.exp(params["basal_rates"]), n_cells, axis=1
        )
        decay_cell = jnp.repeat(jnp.exp(params["decay"]), n_cells, axis=0)
        ki_matrix_cell = jnp.repeat(params["ki_matrix"], n_cells, axis=0)
        hill_coeffs_cell = jnp.repeat(
            jnp.clip(params["hill_coeffs"], 1.0, 4.0), n_cells, axis=0
        )
        prot_tran_rates_cell = jnp.repeat(
            jnp.exp(params["prot_tran_rates"]), n_cells, axis=0
        )
        prot_decay_cell = jnp.repeat(jnp.exp(params["prot_decay"]), n_cells, axis=0)

        def step_fn(carry, step_idx):
            gene_conc, prot_conc = carry
            next_gene, next_prot = sim_instance.calc_x_t(
                gene_conc,
                prot_conc,
                sim_instance.delta,
                decay_cell,
                basal_rates_cell,
                ki_matrix_cell,
                sim_instance.is_mr,
                sim_instance.noise_amp,
                hill_coeffs_cell,
                prot_tran_rates_cell,
                prot_decay_cell,
                sim_instance.noise_a[step_idx],
                sim_instance.noise_b[step_idx],
            )
            next_gene = next_gene.reshape(*next_gene.shape, 1)
            next_prot = next_prot.reshape(*next_prot.shape, 1)
            return (next_gene, next_prot), next_gene

        init_prot = jnp.zeros_like(init_gene_conc)
        _, gene_traj = lax.scan(
            step_fn, (init_gene_conc, init_prot), jnp.arange(steps)
        )
        return gene_traj.squeeze(-1)

    return run_rollout, sim


@app.cell
def _(mo):
    mo.md("""
    ## Learning parameters
    """)
    return


@app.cell
def _(jnp, run_rollout, sim):
    init_conc = jnp.array(
        [
            [[3.0], [1.0]],  # G0
            [[1.0], [3.0]],  # G1
        ]
    )


    def loss_fn(params, w):
        gene_traj = run_rollout(params, sim, init_conc, sim.n_steps)

        final_state_0 = gene_traj[-1, :, 0]
        final_state_1 = gene_traj[-1, :, 1]
        sep_loss = -jnp.sum(
            (final_state_0 - final_state_1) ** 2
        )  ## Increase the seperation between the final states in the two cells for promoting bistable attractors

        diff_last_steps = gene_traj[-1] - gene_traj[-2]
        stable_loss = (
            jnp.sum(diff_last_steps**2) * 50.0
        )  ## Decrease seperation between t-1 and t-2 states in the cells to ensure convergence to an equilibrium

        bound_loss = jnp.sum(
            jnp.maximum(gene_traj - 8.0, 0.0) ** 2
        )  ## Cap values of gene_traj to under 8
        pos_loss = (
            jnp.sum(jnp.maximum(-gene_traj, 0.0) ** 2) * 100.0
        )  ## Ensure gene_traj is >0

        return (
            w[0] * sep_loss
            + w[1] * stable_loss
            + w[2] * bound_loss
            + w[3] * pos_loss
        )

    return init_conc, loss_fn


@app.cell
def _(jax, jit, jnp, loss_fn, optax, plt):
    optimizer = None
    params = None


    @jit
    def train_step(params, opt_state, w):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, w)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val


    def perform_learning(epochs, w):
        global optimizer, params
        params = {
            "basal_rates": jnp.log(jnp.full((2, 1), 0.5)),  # (n_genes, 1)
            "decay": jnp.log(jnp.full((1, 2, 1), 1.0)),  # (1, n_genes, 1)
            "ki_matrix": jnp.array(
                [[[0.0, -1.0], [-1.0, 0.0]]]
            ),  # (1, n_genes, n_genes)
            "hill_coeffs": jnp.full((1, 2, 1), 2.0),
            "prot_tran_rates": jnp.zeros((1, 2, 1)),
            "prot_decay": jnp.zeros((1, 2, 1)),
        }

        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(params)

        # epochs = 80
        opt_state_val = opt_state
        params_val = params
        losses = []
        for epoch in range(epochs):
            params, opt_state, loss_val = train_step(params, opt_state, w)
            if len(losses) > 0 and loss_val < min(losses):
                params_val = params
                opt_state_val = opt_state

            losses.append(loss_val.item())
            if epoch % 20 == 0:
                print(f"Epoch {epoch:02d} - Loss: {loss_val:.4f}")
        # Plot training loss
        opt_state = opt_state_val
        params = params_val
        fig_loss = plt.figure(figsize=(8, 4))
        plt.plot(losses, color="blue", lw=2)
        plt.title(f"Loss curve: {epochs} Weight: {w}")
        plt.xlabel(f"Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)

    return params, perform_learning


@app.cell
def _(plt):
    def plot_figures(learnt_traj, epochs=None, weight=None):
        # Plot trajectories for Cell 0 and Cell 1
        fig_traj, axes = plt.subplots(1, 2, figsize=(12, 5))
        steps_range = range(learnt_traj.shape[0])

        # Cell 0
        axes[0].plot(
            steps_range, learnt_traj[:, 0, 0], label="G0", color="red", lw=2
        )
        axes[0].plot(
            steps_range, learnt_traj[:, 1, 0], label="G1", color="blue", lw=2
        )
        axes[0].set_title("Cell 0 (G0 Dominates)")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Concentration")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        # Cell 1
        axes[1].plot(
            steps_range, learnt_traj[:, 0, 1], label="G0", color="red", lw=2
        )
        axes[1].plot(
            steps_range, learnt_traj[:, 1, 1], label="G1", color="blue", lw=2
        )
        axes[1].set_title("Cell 1 (G1 Dominates)")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Concentration")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        fig_traj.suptitle(
            f"Bistable switch cell trajectory: Epochs {epochs} Weight {weight}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()

    return (plot_figures,)


@app.cell
def _(
    init_conc,
    jnp,
    params,
    perform_learning,
    plot_figures,
    run_rollout,
    sim,
):
    WEIGHT = jnp.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0],
            [10.0, 1.0, 1.0, 1.0],
            [1.0, 10.0, 1.0, 1.0],
            [1.0, 1.0, 10.0, 1.0],
            [1.0, 1.0, 1.0, 10.0],
        ]
    )

    # for _i in [5, 10, 30, 50, 70, 80]:
    for _w in WEIGHT:
        _i = 200  # Set to 80 epochs
        print(f"Epoch - {_i}")
        perform_learning(_i, _w)
        gene_traj = run_rollout(params, sim, init_conc, sim.n_steps)
        plot_figures(gene_traj, _i, _w)
    return


@app.cell
def _(mo, params):
    mo.md(f"""
    ### Learned Shared Parameters

    Learned parameters for both cells:

    * **Basal rates (raw parameters)**: `{params["basal_rates"].tolist()}`
    * **Decay rates (raw parameters)**: `{params["decay"].tolist()}`
    * **Repression coefficients (ki_matrix)**:
    ```python
    {params["ki_matrix"].tolist()}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Simulate with learnt parameters
    """)
    return


@app.cell
def _(init_conc, params, run_rollout, sim):
    # Simulate trajectory using learned parameters
    learned_traj = run_rollout(params, sim, init_conc, sim.n_steps)
    return (learned_traj,)


@app.cell
def _(learned_traj, plt):
    # Plot trajectories for Cell 0 and Cell 1
    fig_traj, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps_range = range(learned_traj.shape[0])

    # Cell 0
    axes[0].plot(steps_range, learned_traj[:, 0, 0], label="G0", color="red", lw=2)
    axes[0].plot(steps_range, learned_traj[:, 1, 0], label="G1", color="blue", lw=2)
    axes[0].set_title("Cell 0 Trajectory (G0 Dominates)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Concentration")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Cell 1
    axes[1].plot(steps_range, learned_traj[:, 0, 1], label="G0", color="red", lw=2)
    axes[1].plot(steps_range, learned_traj[:, 1, 1], label="G1", color="blue", lw=2)
    axes[1].set_title("Cell 1 Trajectory (G1 Dominates)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Concentration")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig_traj.suptitle("Bistable switch Trajectories (with learning)", fontsize=14)
    plt.tight_layout()
    fig_traj
    return


@app.cell
def _(learned_traj, plt):
    # Phase Portrait
    fig_phase = plt.figure(figsize=(6, 6))
    plt.plot(
        learned_traj[:, 0, 0],
        learned_traj[:, 1, 0],
        label="Cell 0 Trajectory",
        color="darkred",
        lw=2,
    )
    plt.scatter(
        learned_traj[0, 0, 0],
        learned_traj[0, 1, 0],
        color="red",
        marker="o",
        s=80,
        label="Cell 0 Start",
    )
    plt.scatter(
        learned_traj[-1, 0, 0],
        learned_traj[-1, 1, 0],
        color="red",
        marker="x",
        s=100,
        label="Cell 0 Attractor",
    )

    plt.plot(
        learned_traj[:, 0, 1],
        learned_traj[:, 1, 1],
        label="Cell 1 Trajectory",
        color="darkblue",
        lw=2,
    )
    plt.scatter(
        learned_traj[0, 0, 1],
        learned_traj[0, 1, 1],
        color="blue",
        marker="o",
        s=80,
        label="Cell 1 Start",
    )
    plt.scatter(
        learned_traj[-1, 0, 1],
        learned_traj[-1, 1, 1],
        color="blue",
        marker="x",
        s=100,
        label="Cell 1 Attractor",
    )

    plt.xlabel("G0 concentration")
    plt.ylabel("G1 concentration")
    plt.title("Bistable switch: Phase diagram")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_phase
    return


@app.cell
def _(mo):
    mo.md(r"""
    Robustness of parameter using perturbations
    """)
    return


@app.cell
def _(init_conc, jax, jnp, params, run_rollout, sim):
    def run_perturb(noise_level, noise_idx, seed=42):
        key = jax.random.PRNGKey(seed)
        perturbed_params = {}
        for k, v in params.items():
            if k in ["basal_rates", "decay", "ki_matrix", "hill_coeffs"]:
                noise = jax.random.normal(key, v.shape) * noise_level
                perturbed_params[k] = v * (1.0 + noise)
            else:
                perturbed_params[k] = v
        traj = run_rollout(perturbed_params, sim, init_conc, sim.n_steps)
        final_dist = jnp.sqrt(jnp.sum((traj[-1, :, 0] - traj[-1, :, 1]) ** 2))
        return final_dist.item()

    return (run_perturb,)


@app.cell
def _(plt, run_perturb):
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    distances = [run_perturb(nl, idx) for idx, nl in enumerate(noise_levels)]

    fig_rob = plt.figure(figsize=(8, 4))
    plt.plot(noise_levels, distances, marker="o", color="red", lw=2)
    plt.title("Robustness to noise")
    plt.xlabel("Perturbation Level (Std Dev)")
    plt.ylabel("Attractor Distance")
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_rob
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
