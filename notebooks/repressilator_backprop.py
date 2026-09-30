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
    from functools import partial
    import matplotlib

    matplotlib.style.use("default")
    return (
        GRNSim,
        compose,
        initialize_config_dir,
        jax,
        jnp,
        lax,
        mo,
        optax,
        os,
        partial,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Repressilator Parameter Learning
    """)
    return


@app.cell
def _(compose, initialize_config_dir, os):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name="repressilator_config")

    print("Configuration loaded. Cells:", cfg.grn.n_cells)
    return (cfg,)


@app.cell
def _(GRNSim, cfg, jnp, lax):
    sim = GRNSim(cfg.grn)


    def run_rollout(params, sim_instance, init_gene_conc, steps):
        n_cells = sim_instance.n_cells

        basal_rates_cell = jnp.repeat(
            jnp.exp(params["basal_rates"]), n_cells, axis=1
        )  # (3, 3)
        decay_cell = jnp.repeat(
            jnp.exp(params["decay"]), n_cells, axis=0
        )  # (3, 3, 1)
        hill_coeffs_cell = jnp.repeat(
            jnp.clip(params["hill_coeffs"], 1.0, 6.0), n_cells, axis=0
        )  # (3, 3, 1)

        ki_val = -jnp.exp(params["ki_coeffs"])  # (1, 3)
        ki_matrix = jnp.zeros((1, 3, 3))
        ki_matrix = ki_matrix.at[0, 0, 2].set(ki_val[0, 0])
        ki_matrix = ki_matrix.at[0, 1, 0].set(ki_val[0, 1])
        ki_matrix = ki_matrix.at[0, 2, 1].set(ki_val[0, 2])
        ki_matrix_cell = jnp.repeat(ki_matrix, n_cells, axis=0)  # (3, 3, 3)

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
        return gene_traj.squeeze(-1)  # (steps, n_genes, n_cells)

    return run_rollout, sim


@app.cell
def _(jnp, run_rollout, sim):
    # Phase-shifted initial conditions for rollout
    init_conc = jnp.array(
        [
            [[3.0], [0.1], [0.1]],  # G0
            [[0.1], [3.0], [0.1]],  # G1
            [[0.1], [0.1], [3.0]],  # G2
        ]
    )


    def loss_fn(params, steps_count, w):
        gene_traj = run_rollout(params, sim, init_conc, steps_count)

        # Maximize temporal variance of the trajectory (negative sign to minimize)
        var_loss = -jnp.mean(jnp.var(gene_traj, axis=0))

        # Boundary constraints: penalize values going above 15.0 or below 0.1
        bound_loss = jnp.mean(jnp.maximum(gene_traj - 15.0, 0.0) ** 2) + jnp.mean(
            jnp.maximum(0.1 - gene_traj, 0.0) ** 2
        )

        # return var_loss * 5.0 + bound_loss * 10.0
        return var_loss * w[0] + bound_loss * w[1]

    return (init_conc,)


@app.cell
def _(mo):
    mo.md("""
    ## Parameter Learning
    """)
    return


@app.cell
def _(jax, optax, partial):
    def train_step_def(optimizer, loss_fn, w):
        @partial(jax.jit, static_argnums=(2,))
        def train_step(params, opt_state, steps_count, w):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, steps_count, w)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        return train_step

    return


@app.cell
def _(plt):
    def create_loss_plot(_losses, weights, epochs=None):
        fig_loss = plt.figure(figsize=(8, 4))
        plt.plot(_losses, color="purple", lw=2)
        plt.title(
            f"Loss curve: variation loss: {weights[0]} bounding loss: {weights[1]}"
            + " Epochs "
            + str(epochs)
            if epochs is not None
            else ""
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    return


app._unparsable_cell(
    r"""
    # w = jnp.array([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0], [3.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0], [1.0, 5.0], [1.0, 10.0]])
    w = jnp.array([[1.0, 3.0]])
    EPOCHS = [5, 10, 30, 50, 100, 150, 200, 300]

    # for _w in w:
    for epochs in EPOCHS:
        _w = w[0]
       params = {
            "basal_rates": jnp.log(jnp.full((3, 1), 0.1)),
            "decay": jnp.log(jnp.full((1, 3, 1), 0.5)),
            "ki_coeffs": jnp.log(
                jnp.full((1, 3), 2.0)
            ),  # Ki = -2.0 (weak repression)
            "hill_coeffs": jnp.full((1, 3, 1), 2.5),
            "prot_tran_rates": jnp.zeros((1, 3, 1)),
            "prot_decay": jnp.zeros((1, 3, 1)),
        }

        # Setup optimizer
        optimizer = optax.adam(learning_rate=0.05)
        opt_state = optimizer.init(params)

        print(f"Current weight = {_w}")
        train_step = train_step_def(optimizer, loss_fn, _w)
        # Run optimization
        # epochs = 300
        losses = []
        params_val, opt_state_val = params, opt_state
        for epoch in range(epochs):
            params, opt_state, loss_val = train_step(params, opt_state, 150, _w)
            if len(losses) > 0 and loss_val < min(losses):
                params_val = params
                opt_state_val = opt_state

            losses.append(loss_val.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} - Min Loss: {min(losses):.4f}")
        create_loss_plot(losses, weights=_w, epochs=epochs)

        learned_traj = run_rollout(params_val, sim, init_conc, 1000)

        # fig_traj, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        steps_range = range(learned_traj.shape[0])

        colors = ["#FF5733", "#33FF57", "#3357FF"]
        plt.figure(figsize=(12, 6))
        for i in range(3):
            plt.plot(
                steps_range,
                learned_traj[:, i, 0],
                color=colors[i],
                lw=2.5,
                label=f"G{i}",
            )

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        plt.xlabel("Time step")
        plt.suptitle(
            f"Repressilator Sustained Oscillations: Epochs {epochs}", fontsize=14
        )
        plt.tight_layout()
        plt.show()
    """,
    name="_"
)


@app.cell
def _():
    return


@app.cell
def _(learned_traj, plt):
    # 3D Phase Portrait
    fig_3d = plt.figure(figsize=(8, 8))
    ax = fig_3d.add_subplot(111, projection="3d")
    ax.plot(
        learned_traj[:, 0, 0],
        learned_traj[:, 1, 0],
        learned_traj[:, 2, 0],
        color="purple",
        lw=2.5,
    )
    ax.set_xlabel("G0 Conc.")
    ax.set_ylabel("G1 Conc.")
    ax.set_zlabel("G2 Conc.")
    ax.set_title("3D Phase Portrait: Recovered Limit Cycle")
    fig_3d
    return


@app.cell
def _(init_conc, jax, jnp, params, run_rollout, sim):
    def add_perturb(noise_level, seed=42):
        key = jax.random.PRNGKey(seed)
        perturbed_params = {}
        for k, v in params.items():
            if k in ["basal_rates", "decay", "ki_coeffs", "hill_coeffs"]:
                noise = jax.random.normal(key, v.shape) * noise_level
                perturbed_params[k] = v * (1.0 + noise)
            else:
                perturbed_params[k] = v

        traj = run_rollout(perturbed_params, sim, init_conc, sim.n_steps)
        return jnp.mean(jnp.var(traj, axis=0)).item()

    return (add_perturb,)


@app.cell
def _(add_perturb, plt):
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    variances = [add_perturb(nl) for nl in noise_levels]

    fig_rob = plt.figure(figsize=(8, 4))
    plt.plot(noise_levels, variances, marker="o", color="blue", lw=2)
    plt.title("Robustness to noise")
    plt.xlabel("Perturbation Level (Std Dev)")
    plt.ylabel("Change in oscillation")
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_rob
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
