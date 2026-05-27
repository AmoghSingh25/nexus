import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib
    from jax import nn

    matplotlib.style.use("default")
    return nn, plt


@app.cell
def _():
    import os
    import sys
    import polars as pl
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    import os
    import sys

    sys.path.append(os.getcwd())

    from nexus.simulator.spatial.spatialSim import SpatialSim
    from nexus.simulator.grn.grnSim import GRNSim


    def load_st_data(
        cells_path, transcripts_path, n_cells_target=500, n_genes_target=50
    ):
        cells_df = pl.read_parquet(cells_path)
        selected_cells = cells_df.head(n_cells_target)
        cell_ids = selected_cells["cell_id"].to_list()
        transcripts_df = pl.read_parquet(transcripts_path)
        filtered_transcripts = transcripts_df.filter(
            pl.col("cell_id").is_in(cell_ids)
        )

        filtered_transcripts = filtered_transcripts.filter(
            ~pl.col("feature_name").str.to_lowercase().str.contains("unassigned")
        )

        gene_counts = (
            filtered_transcripts.group_by("feature_name")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        top_genes = gene_counts.head(n_genes_target)["feature_name"].to_list()
        final_transcripts = filtered_transcripts.filter(
            pl.col("feature_name").is_in(top_genes)
        )

        print(f"Selected {len(cell_ids)} cells and {len(top_genes)} genes.")

        counts_df = final_transcripts.group_by(["feature_name", "cell_id"]).agg(
            pl.len().alias("count")
        )

        pivot_df = counts_df.pivot(
            values="count",
            index="feature_name",
            columns="cell_id",
            aggregate_function="first",
        ).fill_null(0)

        for cid in cell_ids:
            if cid not in pivot_df.columns:
                pivot_df = pivot_df.with_columns(pl.lit(0).alias(cid))

        order_df = pl.DataFrame({"feature_name": top_genes})
        pivot_df = order_df.join(pivot_df, on="feature_name", how="left").fill_null(
            0
        )

        gene_matrix = pivot_df.select(cell_ids).to_numpy()

        x_coords = selected_cells["x_centroid"].to_numpy()
        y_coords = selected_cells["y_centroid"].to_numpy()
        z_coords = jnp.zeros_like(x_coords)
        positions = jnp.column_stack((x_coords, y_coords, z_coords))

        return positions, gene_matrix, top_genes, cell_ids


    def generate_config(n_cells, n_genes, width, height):
        config_dict = {
            "grn": {
                "n_cells": n_cells,
                "delta": 0.01,
                "n_steps": 10,
                "epochs": 500,
                "random_key": 42,
                "lr": 1,
                "logging": False,
                "log_dir": "logs",
                "learn_params": True,
                "non_mr_basal": True,
                "noise": False,
                "noise_amplitude": 0.1,
                "decay": [0.8],
                "hill_coeffs": [1.0],
                "protein_sim": False,
            },
            "spatial_sim": {
                "height": float(height),
                "width": float(width),
                "depth": 10.0,
                "mesh_type": "lattice-free",
                "field_resolution": 3,
                "cell_concentration": float(n_cells / (width * height * 10.0)),
                "n_cell_type": 1,
                "D": 0.5,
                "diffusion_bool": True,
                "n_neighbours": 3,
                "delta": 0.01,
                "n_steps": 10,
                "random_key": 42,
                "logging": False,
                "log_dir": "logs",
                "movement_bool": True,
                "movement": {
                    "cell1": {
                        "qty_ratio": 1.0,
                        "attraction_coeff": 1.0,
                        "repulsion_coeff": 0.1,
                        "drift_vel_coeff": 0.01,
                        "random_vel_coeff": 0.01,
                    }
                },
                "cycle_bool": False,
                "cycle": {
                    "cell1": {
                        "cycle_len": 10,
                        "interphase_len": 0.9,
                        "necrosis_death_prob": 0.0,
                        "apoptosis_death_prob": 0.0,
                        "cell_target_vol": 1.0,
                        "cell_density": 1.0,
                        "cell_vol_growth_rate": 1.0,
                        "cell_decay_rate": 0.5,
                    }
                },
                "chemical": {"name": ["chem1"], "mol_mass": [1]},
                "reaction_bool": False,
                "reaction_prob": False,
                "reaction": {},
            },
        }
        return OmegaConf.create(config_dict)


    def build_grn_graph(n_cells, n_genes, gene_names):

        node_set = []
        for i, name in enumerate(gene_names):
            node = {"type": "mr", "basal_rate": [0.1] * n_cells, "ki": []}
            node_set.append(node)

        edges_set = []
        return node_set, edges_set

    return (
        GRNSim,
        SpatialSim,
        build_grn_graph,
        generate_config,
        jnp,
        load_st_data,
    )


@app.cell
def _(SpatialSim, generate_config, jnp, load_st_data):
    cells_path = "ST_Data/cells.parquet"
    transcripts_path = "ST_Data/transcripts.parquet"

    n_cells_target = 500
    n_genes_target = 100

    positions, gene_matrix, top_genes, cell_ids = load_st_data(
        cells_path, transcripts_path, n_cells_target, n_genes_target
    )

    min_x, max_x = jnp.min(positions[:, 0]), jnp.max(positions[:, 0])
    min_y, max_y = jnp.min(positions[:, 1]), jnp.max(positions[:, 1])
    width = max_x - min_x + 10.0
    height = max_y - min_y + 10.0

    centered_positions = positions - jnp.array(
        [(max_x + min_x) / 2, (max_y + min_y) / 2, 0]
    )

    cfg = generate_config(n_cells_target, n_genes_target, width, height)

    print("Initializing SpatialSim...")
    spatial_sim = SpatialSim(cfg.spatial_sim)

    spatial_sim.mesh.n_cells = n_cells_target
    spatial_sim.mesh.cell_positions = centered_positions


    for attr in [
        "cell_vel",
        "cell_states",
        "cell_time",
        "cell_radius",
        "cell_vol",
        "cell_mass",
        "cell_density",
        "cell_attraction_coeff",
        "cell_repulsion_coeff",
        "cell_drift_vel_coeff",
        "cell_random_vel_coeff",
        "cell_death_decay_coeff",
        "cell_death_prob",
        "cell_prg_death_prob",
        "interphase_chkpt",
        "mitosis_chkpt",
        "cell_target_vol",
        "cell_vol_growth_rate",
        "cell_type_mask",
    ]:
        if hasattr(spatial_sim.mesh, attr):
            arr = getattr(spatial_sim.mesh, attr)
            if isinstance(arr, jnp.ndarray) and arr.shape[0] > 0:
                if arr.shape[0] >= n_cells_target:
                    setattr(spatial_sim.mesh, attr, arr[:n_cells_target])
                else:
                    pad_size = n_cells_target - arr.shape[0]
                    pad_vals = jnp.repeat(arr[-1:], pad_size, axis=0)
                    setattr(
                        spatial_sim.mesh,
                        attr,
                        jnp.concatenate([arr, pad_vals], axis=0),
                    )
    return (
        cfg,
        gene_matrix,
        n_cells_target,
        n_genes_target,
        spatial_sim,
        top_genes,
    )


@app.cell
def _(
    GRNSim,
    build_grn_graph,
    cfg,
    gene_matrix,
    jnp,
    n_cells_target,
    n_genes_target,
    spatial_sim,
    top_genes,
):
    spatial_sim.mesh.live_cells_mask = spatial_sim.mesh.cell_states != -1
    spatial_sim.mesh.prg_cells_mask = spatial_sim.mesh.cell_states == -2

    target_matrix = jnp.array(gene_matrix).reshape(
        n_genes_target, n_cells_target, 1
    )
    mean_gene = target_matrix.mean(axis=1)

    print("Initializing GRNSim...")
    node_set, edges_set = build_grn_graph(n_cells_target, n_genes_target, top_genes)
    grn_sim = GRNSim(
        cfg.grn,
        node_set=node_set,
        edges_set=edges_set,
        target_gene_conc=target_matrix,
        target_prot_conc=None,
    )
    return (grn_sim,)


@app.cell
def _(grn_sim, nn):
    grn_sim.basal_rates = nn.softplus(grn_sim.learnt_gene_params["basal_rates"])
    grn_sim.decay = nn.softplus(grn_sim.learnt_gene_params["decay"])
    grn_sim.ki_matrix = grn_sim.learnt_gene_params["ki_matrix"]
    grn_sim.hill_coeffs = grn_sim.learnt_gene_params["hill_coeffs"]

    grn_sim.steady_states, grn_sim.prot_steady_state, _, _ = (
        grn_sim.calc_steady_states(learn_params=False)
    )
    learnt_gene_params, learnt_prot_params = grn_sim.learn_params_fn()

    # 2. Update the simulator parameters with the optimized values
    grn_sim.basal_rates = learnt_gene_params["basal_rates"]
    grn_sim.decay = learnt_gene_params["decay"]
    grn_sim.ki_matrix = learnt_gene_params["ki_matrix"]
    grn_sim.hill_coeffs = learnt_gene_params["hill_coeffs"]

    if grn_sim.protein_sim:
        grn_sim.prot_tran_rates = nn.softplus(learnt_prot_params["prot_tran_rates"])
        grn_sim.prot_decay = nn.softplus(learnt_prot_params["prot_decay"])

    grn_sim.gene_conc = grn_sim.steady_states
    if grn_sim.protein_sim:
        grn_sim.prot_conc = grn_sim.prot_steady_state

    ret_2 = grn_sim.run_sim()
    return (ret_2,)


@app.cell
def _(grn_sim, plt):
    plt.plot(grn_sim.steady_states[:, 0])
    plt.plot(grn_sim.target_gene_conc[:, 0])
    return


@app.cell
def _(grn_sim):
    ret_3 = grn_sim.run_sim()
    return (ret_3,)


@app.cell
def _(grn_sim, plt, ret_2, ret_3):
    plt.plot(ret_2.gene_traj[-1][:, 0], label="Pred")
    plt.plot(ret_3.gene_traj[-1][:, 0], label="Pred - 2")
    plt.plot(grn_sim.target_gene_conc[:, 0], label="Orig")
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
