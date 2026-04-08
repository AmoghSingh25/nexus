import jax.numpy as jnp
from jax import random
from nexus.simulator.spatial.utils.random_generators import generate_choices

## TODO: Use Clustering in field-cell computation - Assign fields after 2 steps to allow stabilization
## TODO: K Medoids
## TODO: Delauney triangulation


def k_mean_clustering(num_clusters, positions, num_steps=20):
    ## Initial cluster assignment
    cluster_assgn = jnp.zeros(len(positions))
    clust_size = len(positions) // num_clusters

    key, sub_key = random.split(random.key(42))
    key, sub_key, ret = generate_choices(
        key, sub_key, a=len(positions), shape=(num_clusters, clust_size), replace=False
    )
    for i in range(len(ret)):
        cluster_assgn = cluster_assgn.at[ret[i]].set(i + 1)
    cluster_assgn = cluster_assgn.at[cluster_assgn == 0].set(num_clusters).astype(int)

    cluster_means = jnp.zeros((num_clusters, 3))

    for i in range(num_steps):
        for i in range(num_clusters):
            cluster_means = cluster_means.at[i].set(
                jnp.mean(positions[cluster_assgn == i + 1], axis=0)
            )

        rep_pos = jnp.repeat(
            positions.reshape(positions.shape[0], 1, positions.shape[1]),
            axis=1,
            repeats=3,
        )
        new_cluster_assgns = (
            jnp.argmin(jnp.linalg.norm(rep_pos - cluster_means, axis=2, ord=1), axis=1)
            + 1
        )
        cluster_assgn = new_cluster_assgns

    for i in range(num_clusters):
        cluster_means = cluster_means.at[i].set(
            jnp.mean(positions[cluster_assgn == i + 1], axis=0)
        )

    return cluster_assgn - 1, cluster_means
