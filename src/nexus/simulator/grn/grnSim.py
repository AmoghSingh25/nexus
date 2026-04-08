# Assumptions
# - Simulation starts from steady state calculation
# - Basal rates of non-MR is 0 (from SERGIO) - Can also be configured for non zero basal rates

import os
import time
import networkx as nx
import jax.numpy as jnp
from jax import vmap, random, lax, jit, clear_caches, grad, config, nn
from nexus.simulator.utils.read_network import _read_data
from tqdm import tqdm
import logging
from nexus.simulator.noise_models.wiener_noise import WienerNoise
import numpy as np
from nexus.simulator.utils.verify_network import _copy_param_vals
from omegaconf import DictConfig
from nexus.simulator.grn.logger.grnLogger import GRNLogger
import optax


class GRNSim:
    def __init__(
        self,
        cfg: DictConfig,
        node_set=None,
        edges_set=None,
        target_gene_conc=None,
        target_prot_conc=None,
    ):
        """
        Shapes of variables :

        gene_conc -     (n_genes, n_cells, 1)
        steady_states - (n_genes, n_cells, 1)
        basal_rates -   (n_genes, n_cells, 1)
        ki_matrix -     (n_cells, n_genes, n_genes)
        is_mr -         (n_genes,)
        prot_conc -     (n_genes, n_cells, 1)
        prot_kt -       (n_cells, n_genes)
        prot_kd -       (n_cells, n_genes)
        decay -         (n_cells, n_genes, 1)
        """
        config.update("jax_debug_nans", True)

        self.delta = cfg.delta
        self.n_cells = cfg.n_cells
        self.protein_sim = cfg.protein_sim
        self.noise = cfg.noise
        if self.noise:
            self.noise_amp = jnp.array([cfg.noise_amplitude])
            self.delta_sq = jnp.sqrt(self.delta)
        else:
            self.noise_amp = jnp.array([0.0])
        self.copy_cells = False
        self.non_mr_basal = cfg.non_mr_basal
        self.decay = jnp.array(cfg.decay)
        self.hill_coeffs = jnp.array(cfg.hill_coeffs)
        self.n_steps = cfg.n_steps
        self.learn_params = cfg.get("learn_params", False)
        self.epochs = cfg.get("epochs", 0)
        self.lr = cfg.get("lr", 0.01)
        if node_set is None and edges_set is None:
            node_set, edges_set, self.copy_cells = _read_data(
                gene_data=cfg.get("gene_data", None),
                mr_data=cfg.get("mr_data", None),
                config_file=cfg.get("config_file", ""),
                n_cells=self.n_cells,
                protein_sim=self.protein_sim,
            )
        self.n_genes = len(node_set)

        ## Setting up logger
        self.is_logging = cfg.get("logging", False)
        if self.is_logging:
            self.timestamp = cfg.get("log_file_name", str(int(time.time())))
            self.logger = GRNLogger(
                log_dir=os.path.join("logs"),
                file_name=self.timestamp,
                n_steps=self.n_steps,
                n_cells=self.n_cells,
                n_genes=self.n_genes,
            )

        if target_gene_conc is not None:
            self.target_gene_conc = target_gene_conc.reshape(self.n_genes, 1)
            self.target_prot_conc = target_prot_conc.reshape(self.n_genes, 1)

        self.decay = _copy_param_vals(
            var=self.decay, var_name="Decay", n_cells=self.n_cells, n_genes=self.n_genes
        )

        self.hill_coeffs = _copy_param_vals(
            var=self.hill_coeffs,
            var_name="hill coefficients",
            n_cells=self.n_cells,
            n_genes=self.n_genes,
        )
        self.noise_amp = _copy_param_vals(
            var=self.noise_amp,
            var_name="noise amplitude",
            n_cells=self.n_cells,
            n_genes=self.n_genes,
        )

        self.key, self.sub_key = random.split(random.key(cfg.get("random_key", 42)))

        self.noise_a = WienerNoise(delta=self.delta)
        self.noise_b = WienerNoise(delta=self.delta, random_key=43)

        self.basal_rates = []
        self.ki_values = []
        self.is_mr = []
        self.gene_conc = jnp.zeros((self.n_genes, self.n_cells, 1))
        self.steady_states = jnp.zeros_like(self.gene_conc)

        self.prot_conc = jnp.zeros_like(self.gene_conc)

        self.prot_steady_state = jnp.zeros_like(self.gene_conc)
        self.prot_tran_rates = jnp.zeros_like(self.gene_conc)
        self.prot_decay = jnp.zeros_like(self.gene_conc)
        self.prot_half_lives = jnp.zeros_like(self.gene_conc)
        self.ki_matrix = np.zeros((self.n_cells, self.n_genes, self.n_genes))

        self.g = nx.DiGraph()
        self.g.add_edges_from(edges_set)

        for i in range(len(node_set)):
            node = node_set[i]
            self.is_mr.append(True if node["type"] == "mr" else False)
            if node["type"] == "mr":
                if self.copy_cells:
                    self.basal_rates.append(
                        _copy_param_vals(
                            jnp.array(node["basal_rate"]),
                            "Basal rate",
                            n_cells=self.n_cells,
                            n_genes=self.n_genes,
                        )
                    )
                else:
                    self.basal_rates.append(
                        jnp.array(node["basal_rate"]).reshape((self.n_cells, 1))
                    )
                self.ki_values.append(jnp.array([0]))
            else:
                regulators = list(sorted(self.g.predecessors(i)))
                basal_rate_i = jnp.zeros((self.n_cells, 1))  # 0 basal rate for non-MRs
                self.g.add_node(i)
                ki_vals = jnp.array(node["ki"])
                if self.non_mr_basal:
                    self.basal_rates.append(
                        jnp.array(node["basal_rate"]).reshape((self.n_cells, 1))
                    )
                else:
                    self.basal_rates.append(basal_rate_i)
                if ki_vals.ndim == 2:
                    self.ki_matrix[:, i, regulators] = ki_vals
                else:
                    self.ki_matrix[:, i, regulators] = ki_vals[0][:, 0].reshape(1, -1)

                if self.protein_sim:
                    self.prot_half_lives = self.prot_half_lives.at[i].set(
                        jnp.array(node["prot_half_life"]).reshape(-1, 1)
                    )
                    self.prot_tran_rates = self.prot_tran_rates.at[i].set(
                        jnp.array(node["prot_transcription_rate"]).reshape(-1, 1)
                    )

        if self.protein_sim:
            self.prot_half_lives = self.prot_half_lives.reshape(
                self.n_genes, self.n_cells
            ).T.reshape(self.n_cells, self.n_genes, 1)
            self.prot_tran_rates = self.prot_tran_rates.reshape(
                self.n_genes, self.n_cells
            ).T.reshape(self.n_cells, self.n_genes, 1)
            self.prot_decay = jnp.nan_to_num(
                jnp.log(2) / self.prot_half_lives, nan=0.0, neginf=0.0, posinf=0.0
            )
        else:
            self.prot_half_lives = jnp.zeros((self.n_cells, self.n_genes, 1))
            self.prot_tran_rates = jnp.zeros((self.n_cells, self.n_genes, 1))
            self.prot_decay = jnp.zeros_like(self.prot_half_lives)

        logging.info("Setting KI Matrix")
        self.ki_matrix = jnp.array(self.ki_matrix)
        del self.ki_values, self.g, node_set, edges_set
        clear_caches()

        self.basal_rates = jnp.array(self.basal_rates).reshape(
            self.n_genes, self.n_cells
        )
        self.is_mr = jnp.array(self.is_mr)

        ## Create JIT functions
        self.jit_pij = jit(self.calc_pij)
        self.jit_x_t = jit(self.calc_x_t)

        logging.info("Calculating steady states...")
        (
            self.gene_conc,
            self.prot_conc,
            self.learnt_gene_params,
            self.learnt_prot_params,
        ) = self.calc_steady_states()
        self.steady_states = self.gene_conc
        self.prot_steady_state = self.prot_conc

        logging.info("Steady state concentrations calculated.")

    def calc_steady_state_mr(self, b, decay):
        """Steady state calculation for MRs"""
        return (b / decay).reshape(-1, 1), jnp.zeros_like(b).reshape(-1, 1)

    def calc_steady_state_g(
        self,
        is_mr,
        idx,
        basal_rates,
        decay,
        gene_conc,
        gene_cell_mean,
        k_i,
        hill_coeff,
        p_kt,
        p_kd,
    ):
        """Steady state calculation for genes and proteins"""
        eps = 1e-8
        e_x = self.jit_pij(
            is_mr=is_mr,
            idx=idx,
            basal_rates=basal_rates,
            gene_conc=gene_conc,
            gene_cell_mean=gene_cell_mean,
            k_i=k_i,
            _hill=hill_coeff,
        ) / (decay + eps)
        p_kd = jnp.maximum(p_kd, 1e-4)
        p_c = lax.cond(
            self.protein_sim,
            lambda _: (p_kt * e_x) / p_kd,
            lambda _: jnp.zeros_like(e_x, dtype=jnp.float32),
            operand=None,
        )

        return e_x, p_c

    def calc_steady_states(self, learn_params=True):
        """Calculates the steady state concentrations for the MR and Gene nodes.
        The steady state concentrations are calculated using the method mentioned in Equation 8 and Equation 10 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        A similar method is used for estimating the steady state concentration of the proteins and is mentioned in `docs/simulator.md`
        """

        def _single_gene_steady_state(
            n_cells_range,
            idx,
            is_mr,
            basal_rate,
            decay,
            ki_matrix,
            gene_conc,
            all_cell_conc,
            hill_coeff,
            prot_trans,
            prot_decay,
        ):
            def _single_cell_steady_state(
                gene_idx,
                cell_idx,
                is_mr,
                basal_rate,
                decay,
                ki_matrix,
                gene_conc,
                all_cell_conc,
                hill_coeff,
                prot_trans,
                prot_decay,
            ):
                jit_calc_steady_state = self.calc_steady_state_g
                return jit_calc_steady_state(
                    is_mr=is_mr,
                    idx=gene_idx,
                    basal_rates=basal_rate,
                    decay=decay,
                    gene_conc=gene_conc[:, cell_idx],
                    gene_cell_mean=all_cell_conc,
                    k_i=ki_matrix,
                    hill_coeff=hill_coeff,
                    p_kt=prot_trans,
                    p_kd=prot_decay,
                )

            vmap_single_cell = jit(
                vmap(
                    _single_cell_steady_state,
                    in_axes=(None, 0, None, 0, 0, 0, None, None, 0, 0, 0),
                )
            )
            steady_vals = lax.cond(
                is_mr,
                lambda _: jit(self.calc_steady_state_mr)(basal_rate, decay[:, idx]),
                lambda _: vmap_single_cell(
                    idx,
                    n_cells_range,
                    is_mr,
                    basal_rate,
                    decay,
                    ki_matrix,
                    gene_conc,
                    all_cell_conc,
                    hill_coeff,
                    prot_trans,
                    prot_decay,
                ),
                operand=None,
            )

            return steady_vals[0], steady_vals[1]

        gene_conc, prot_conc = self.gene_conc, self.prot_conc
        ## TODO: VMAP over genes does not work as genes depend on the concentration of the regulator genes

        # vmap_gene_conc, vmap_prot_conc = _all_genes_steady_state(
        #     self.n_cells,
        #     self.is_mr,
        #     self.basal_rates,
        #     self.decay,
        #     self.ki_matrix,
        #     gene_conc,
        #     self.hill_coeffs,
        #     self.prot_tran_rates,
        #     self.prot_decay,
        #     self.prot_conc,
        #     self.prot_steady_state
        # )
        calc_steady_state_jit = jit(_single_gene_steady_state)

        for i in tqdm(range(self.n_genes)):
            g_conc, p_conc = calc_steady_state_jit(
                n_cells_range=jnp.arange(self.n_cells),
                idx=i,
                is_mr=self.is_mr[i],
                basal_rate=self.basal_rates[i],
                decay=self.decay[:, i],
                ki_matrix=self.ki_matrix[:, i, :],
                gene_conc=gene_conc,
                all_cell_conc=jnp.mean(
                    gene_conc, axis=1
                ),  # To calculate half response as mean conc across all cells
                hill_coeff=self.hill_coeffs[:, i],
                prot_trans=self.prot_tran_rates[:, i],
                prot_decay=self.prot_decay[:, i],
            )
            gene_conc = gene_conc.at[i].set(g_conc)
            prot_conc = prot_conc.at[i].set(p_conc)

        ## Copy params to shift values

        if self.learn_params and learn_params:
            logging.info("Running backpropagation to learn parameters...")

            def loss_target(
                params,
                params_prot,
                n_cells_range,
                n_genes_range,
                is_mr,
                gene_conc,
                target_conc,
                prot_tran_rates,
                target_gene=True,
            ):
                def _all_genes_steady_state(
                    n_cells_range,
                    n_genes_range,
                    is_mr,
                    basal_rates,
                    decay,
                    ki_matrix,
                    gene_conc,
                    hill_coeffs,
                    prot_tran_rates,
                    prot_decay,
                ):
                    vmap_all_genes = vmap(
                        _single_gene_steady_state,
                        in_axes=(None, 0, 0, 0, 1, 1, None, None, 1, 1, 1),
                    )

                    ret_ = vmap_all_genes(
                        n_cells_range,
                        n_genes_range,
                        is_mr,
                        basal_rates,
                        decay,
                        ki_matrix,
                        gene_conc,
                        jnp.mean(gene_conc, axis=1),
                        hill_coeffs,
                        prot_tran_rates,
                        prot_decay,
                    )
                    return ret_

                params["hill_coeffs"] = jnp.clip(params["hill_coeffs"], 1.0, 4.0)
                g_conc, p_conc = _all_genes_steady_state(
                    n_cells_range,
                    n_genes_range,
                    is_mr,
                    nn.softplus(params["basal_rates"]),
                    nn.softplus(params["decay"]),
                    params["ki_matrix"],
                    gene_conc,
                    params["hill_coeffs"],
                    nn.softplus(params_prot["prot_tran_rates"]),
                    nn.softplus(params_prot["prot_decay"]),
                )
                n_cells = len(n_cells_range)
                p_conc = jnp.clip(p_conc, min=jnp.min(target_conc))

                l2_gene_coeff = 1e-4
                l2_prot_coeff = 1e-4
                loss = lax.cond(
                    target_gene,
                    lambda _: (
                        jnp.sqrt(
                            jnp.sum(
                                (
                                    jnp.log1p(
                                        target_conc.repeat(axis=1, repeats=n_cells)
                                    )
                                    - jnp.log1p(g_conc.squeeze(2))
                                )
                                ** 2,
                                dtype=jnp.float32,
                            )
                        ).astype(jnp.float32)
                        + l2_gene_coeff * jnp.sum(params["ki_matrix"] ** 2)
                    ),
                    lambda _: (
                        jnp.sqrt(
                            jnp.sum(
                                (
                                    jnp.log1p(
                                        target_conc.repeat(axis=1, repeats=n_cells)
                                    )
                                    - jnp.log1p(p_conc.squeeze(2))
                                )
                                ** 2,
                                dtype=jnp.float32,
                            )
                        ).astype(jnp.float32)
                        + l2_prot_coeff * jnp.sum(params_prot["prot_tran_rates"] ** 2)
                    ),
                    operand=None,
                )
                return loss

            grad_loss_gene = jit(grad(loss_target, argnums=0))
            grad_loss_prot = jit(grad(loss_target, argnums=1))
            calc_loss = jit(loss_target)
            basal_rates = self.basal_rates
            decay = self.decay
            ki_matrix = self.ki_matrix
            hill_coeffs = self.hill_coeffs
            prot_tran_rates = self.prot_tran_rates
            prot_decay = self.prot_decay

            params_gene = {
                "basal_rates": basal_rates,
                "decay": decay,
                "ki_matrix": ki_matrix,
                "hill_coeffs": hill_coeffs,
            }
            params_prot = {
                "prot_tran_rates": prot_tran_rates,
                "prot_decay": prot_decay,
            }
            scheduler = optax.cosine_decay_schedule(
                init_value=self.lr, decay_steps=self.epochs
            )
            scheduler_prot = optax.cosine_decay_schedule(
                init_value=self.lr, decay_steps=self.epochs
            )

            optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
            optimizer_prot = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=scheduler_prot),
            )

            opt_state = optimizer.init(params_gene)
            opt_state_prot = optimizer_prot.init(params_prot)
            losses = []

            for e_i in range(self.epochs):
                logging.info(f"\tEpoch - {e_i}")
                grad_gene = grad_loss_gene(
                    params_gene,
                    params_prot,
                    jnp.arange(self.n_cells),
                    jnp.arange(self.n_genes),
                    self.is_mr,
                    self.gene_conc,
                    target_conc=self.target_gene_conc,
                    prot_tran_rates=prot_tran_rates,
                )
                updates, opt_state = optimizer.update(grad_gene, opt_state)
                params_gene = optax.apply_updates(params_gene, updates)
                if e_i % 50 == 0:
                    loss_i = calc_loss(
                        params_gene,
                        params_prot,
                        jnp.arange(self.n_cells),
                        jnp.arange(self.n_genes),
                        self.is_mr,
                        self.gene_conc,
                        target_conc=self.target_gene_conc,
                        prot_tran_rates=prot_tran_rates,
                    )
                    loss_i_prot = calc_loss(
                        params_gene,
                        params_prot,
                        jnp.arange(self.n_cells),
                        jnp.arange(self.n_genes),
                        self.is_mr,
                        self.gene_conc,
                        target_conc=self.target_prot_conc,
                        prot_tran_rates=prot_tran_rates,
                        target_gene=False,
                    )
                    losses.append(loss_i)
                    print(
                        f"Epoch - {e_i} Loss = {loss_i} Protein loss = {loss_i_prot} Learing rate = {opt_state.hyperparams['learning_rate']}"
                    )

                grad_prot = grad_loss_prot(
                    params_gene,
                    params_prot,
                    jnp.arange(self.n_cells),
                    jnp.arange(self.n_genes),
                    self.is_mr,
                    self.gene_conc,
                    target_conc=self.target_prot_conc,
                    prot_tran_rates=prot_tran_rates,
                    target_gene=False,
                )
                updates, opt_state_prot = optimizer_prot.update(
                    grad_prot, opt_state_prot
                )
                params_prot = optax.apply_updates(params_prot, updates)
            return gene_conc, prot_conc, params_gene, params_prot
        else:
            return gene_conc, prot_conc, {}, {}

    def calc_pij(
        self, is_mr, idx, basal_rates, gene_conc, gene_cell_mean, k_i, _hill=1
    ):
        """Calculates the production rate of each gene as a function of its regulator genes as given in Equation 5, Equation 6 and Equation 7 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        Current assumption -
            Half life is same as E[x] for genes

        """

        def _calc_pij_g(gene_conc, gene_cell_mean, k_i, _hill=1):
            eps = 1e-8
            num = jnp.pow(gene_conc, _hill)
            frac = num / (jnp.pow(gene_cell_mean, _hill) + num + eps)

            def _calc_hill(f_i, k_ij):
                return lax.cond(
                    k_ij < 0, lambda _: 1 - f_i, lambda _: f_i, operand=None
                )

            vmap_hill_func = vmap(_calc_hill, in_axes=(0, 0))

            frac = vmap_hill_func(frac, k_i)
            frac = frac * jnp.abs(k_i).reshape(-1, 1)
            frac = jnp.nan_to_num(frac, nan=0.0, neginf=0.0, posinf=0.0).reshape(-1)
            return frac.sum(axis=0)

        return lax.cond(
            is_mr,
            lambda x: jnp.zeros_like(basal_rates),
            lambda x: (
                _calc_pij_g(
                    gene_conc=gene_conc,
                    gene_cell_mean=gene_cell_mean,
                    k_i=k_i,
                    _hill=_hill,
                )
                + basal_rates
            ),
            operand=None,
        )

    def calc_x_t(
        self,
        gene_conc,
        prot_conc,
        delta,
        decay,
        basal_rates,
        k_i,
        is_mr,
        noise_amp,
        hill_coeff,
        prot_kt,
        prot_kd,
        noise_a,
        noise_b,
    ):
        """Estimate the concentration of each gene and protein at the next time step.

        For gene concentration, the simulator uses Equation 3 given in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        For protein concentration, the simulator uses Equation 1 given in
        Kuchta et al. Predicting proteome dynamics using gene expression data. Scientific Reports, 8(1), 13866

        Protein concentration at time $t$ is computed using the gene concentration at time $t-1$.
        """

        def _single_gene_x_t(
            idx,
            gene_conc,
            delta,
            decay,
            basal_rates,
            k_i,
            is_mr,
            gene_cell_mean,
            noise_amp,
            hill_coeff,
            prot_conc,
            prot_kt,
            prot_kd,
            noise_a_i,
            noise_b_i,
        ):
            # gene_conc, delta, decay, steady_state, is_mr - Entire arrays passed for all genes in a cell
            p_i = self.jit_pij(
                is_mr=is_mr,
                idx=idx,
                basal_rates=basal_rates,
                gene_conc=gene_conc,
                gene_cell_mean=gene_cell_mean,
                k_i=k_i,
                _hill=hill_coeff,
            )
            x_t_gene = gene_conc[idx] + (p_i - decay * gene_conc[idx]) * delta

            if self.noise:
                noise_add = noise_amp * (
                    jnp.sqrt(p_i) * noise_a_i
                    + jnp.sqrt(decay * gene_conc[idx]) * noise_b_i
                )
                x_t_gene += noise_add

            x_t_prot = (
                prot_conc + (prot_kt * gene_conc[idx] - prot_kd * prot_conc) * delta
            )
            return x_t_gene, x_t_prot

        # Auto vectorization over all the genes
        auto_vec_genes = vmap(
            _single_gene_x_t,
            in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

        # Auto vectorization over auto_vec_genes for all cells
        auto_vec_cells = vmap(
            auto_vec_genes,
            in_axes=(None, 1, None, 0, 1, 0, None, None, 0, 0, 1, 0, 0, 0, 0),
        )

        x_t, p_t = auto_vec_cells(
            jnp.arange(self.n_genes),
            gene_conc,
            delta,
            decay,
            basal_rates,
            k_i,
            is_mr,
            jnp.mean(gene_conc, axis=1),
            noise_amp,
            hill_coeff,
            prot_conc,
            prot_kt,
            prot_kd,
            noise_a,
            noise_b,
        )

        x_t = x_t.reshape((self.n_cells, self.n_genes)).T
        p_t = p_t.reshape((self.n_cells, self.n_genes)).T
        return x_t, p_t

    def run_sim(self, step=None):
        """
        Run the simulation for n_steps

        :param self: simulator
        :param n_steps: Number of steps to run the simulation.
        """
        if step is None:
            logging.info("Running simulator...")
            gene_conc_history = []
            prot_conc_history = []
            for t_i in tqdm(range(self.n_steps)):
                wiener_noise_a = self.noise_a.generate_noise(
                    shape=(self.n_cells, self.n_genes)
                )
                wiener_noise_b = self.noise_b.generate_noise(
                    shape=(self.n_cells, self.n_genes)
                )

                self.key, self.sub_key = random.split(self.key)
                _gene_conc, _prot_conc = self.jit_x_t(
                    self.gene_conc,
                    self.prot_conc,
                    self.delta,
                    self.decay,
                    self.basal_rates,
                    self.ki_matrix,
                    self.is_mr,
                    self.noise_amp,
                    self.hill_coeffs,
                    self.prot_tran_rates,
                    self.prot_decay,
                    wiener_noise_a,
                    wiener_noise_b,
                )
                _gene_conc = _gene_conc.reshape(*_gene_conc.shape, 1)
                _prot_conc = _prot_conc.reshape(*_prot_conc.shape, 1)
                self.gene_conc = self.gene_conc.at[:].set(_gene_conc)
                self.prot_conc = self.prot_conc.at[:].set(_prot_conc)
                if self.is_logging:
                    self.logger.log_conc(
                        step=t_i if step is None else step,
                        cell=None,
                        gene_conc=self.gene_conc,
                        prot_conc=self.prot_conc,
                    )
                clear_caches()
                gene_conc_history.append(self.gene_conc)
                prot_conc_history.append(self.prot_conc)
            logging.info("Simulation ended...")
            return gene_conc_history, prot_conc_history
        else:
            wiener_noise_a = self.noise_a.generate_noise(
                shape=(self.n_cells, self.n_genes)
            )
            wiener_noise_b = self.noise_b.generate_noise(
                shape=(self.n_cells, self.n_genes)
            )

            self.key, self.sub_key = random.split(self.key)
            _gene_conc, _prot_conc = self.jit_x_t(
                self.gene_conc,
                self.prot_conc,
                self.delta,
                self.decay,
                self.basal_rates,
                self.ki_matrix,
                self.is_mr,
                self.noise_amp,
                self.hill_coeffs,
                self.prot_tran_rates,
                self.prot_decay,
                wiener_noise_a,
                wiener_noise_b,
            )
            _gene_conc = _gene_conc.reshape(*_gene_conc.shape, 1)
            _prot_conc = _prot_conc.reshape(*_prot_conc.shape, 1)
            self.gene_conc = self.gene_conc.at[:].set(_gene_conc)
            self.prot_conc = self.prot_conc.at[:].set(_prot_conc)
            if self.is_logging:
                self.logger.log_conc(
                    step=t_i if step is None else step,
                    cell=None,
                    gene_conc=self.gene_conc,
                    prot_conc=self.prot_conc,
                )
            clear_caches()
            return self.gene_conc, self.prot_conc
