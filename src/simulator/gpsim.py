# Assumptions
# - Simulation starts from steady state calculation
# - Basal rates of non-MR is 0 (from SERGIO)

import networkx as nx
import jax.numpy as jnp
from jax import vmap, random, lax, jit, clear_caches
from .utils.read_network import _read_data
from tqdm import tqdm
import logging
from .noise_models.wiener_noise import WienerNoise
import numpy as np
from .utils.verify_network import _copy_param_vals


class simulator:
    def __init__(
        self,
        gene_data=None,
        mr_data=None,
        node_set=None,
        edges_set=None,
        config_file="",
        n_cells=1,
        protein_sim=True,
        non_mr_basal=False,
        delta=0.01,
        noise=True,
        noise_amplitude=[0.1],
        decay=[0.8],
        hill_coeffs=[1.0],
    ):
        """
        Shapes of variables :

        gene_conc -     (n_genes, n_cells, 1)
        steady_states - (n_genes, n_cells, 1)
        basal_rates -   (n_genes, n_cells, 1)
        ki_matrix -     (n_cells, n_genes, n_genes)
        is_mr -         (n_genes,)
        decay -         (1,1)    # For now
        prot_conc -     (n_genes, n_cells, 1)
        prot_kt -       (n_cells, n_genes)
        prot_kd -       (n_cells, n_genes)
        decay -         (n_cells, n_genes, 1)
        """

        self.delta = delta
        self.n_cells = n_cells
        self.protein_sim = protein_sim
        self.noise = noise
        self.noise_amp = jnp.array(noise_amplitude)
        self.copy_cells = False
        self.non_mr_basal = non_mr_basal
        if node_set is None and edges_set is None:
            node_set, edges_set, self.copy_cells = _read_data(
                gene_data=gene_data,
                mr_data=mr_data,
                config_file=config_file,
                n_cells=n_cells,
                protein_sim=protein_sim,
            )
        self.n_genes = len(node_set)

        self.decay = jnp.array(decay)
        self.hill_coeffs = jnp.array(hill_coeffs)

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

        self.key, self.sub_key = random.split(random.key(42))

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
                self.basal_rates.append(
                    jnp.array(node["basal_rate"]).reshape((n_cells, 1))
                )
                self.ki_values.append(jnp.array([0]))
            else:
                regulators = list(sorted(self.g.predecessors(i)))
                basal_rate_i = jnp.zeros((n_cells, 1))  # 0 basal rate for non-MRs
                self.key, self.sub_key = random.split(self.key)
                self.g.add_node(i)
                ki_vals = jnp.array(node["ki"])
                if self.non_mr_basal:
                    self.basal_rates.append(
                        jnp.array(node["basal_rate"]).reshape((n_cells, 1))
                    )
                else:
                    self.basal_rates.append(basal_rate_i)

                if ki_vals.ndim == 2:
                    # ki_vals = ki_vals.reshape(self.n_cells, len(regulators), 1)
                    print(self.n_cells)
                    print(ki_vals.shape)
                    print(self.ki_matrix[:, i, regulators].shape)
                    # ki_vals = jnp.repeat(ki_vals, repeats=self.n_cells, axis=0)
                    self.ki_matrix[:, i, regulators] = ki_vals

                # self.ki_matrix = self.ki_matrix.at[:, i, regulators].set(ki_vals)

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
        self.gene_conc, self.prot_conc = self.calc_steady_states()
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
        e_x = (
            self.jit_pij(
                is_mr=is_mr,
                idx=idx,
                basal_rates=basal_rates,
                gene_conc=gene_conc,
                gene_cell_mean=gene_cell_mean,
                k_i=k_i,
                _hill=hill_coeff,
            )
            / decay
        )
        if self.protein_sim:
            p_c = (p_kt * e_x) / p_kd
        else:
            p_c = jnp.zeros_like(e_x)

        return e_x, p_c

    def calc_steady_states(self):
        """Calculates the steady state concentrations for the MR and Gene nodes.
        The steady state concentrations are calculated using the method mentioned in Equation 8 and Equation 10 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        A similar method is used for estimating the steady state concentration of the proteins and is mentioned in `docs/simulator.md`
        """
        logging.info("Calculating steady states...")

        def _single_gene_steady_state(
            n_cells,
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
            prot_conc,
            prot_ss,
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
                prot_conc,
                prot_ss,
            ):
                return self.calc_steady_state_g(
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

            vmap_single_cell = vmap(
                _single_cell_steady_state,
                in_axes=(None, 0, None, 0, 0, 0, None, None, 0, 0, 0, None, None),
            )
            steady_vals = lax.cond(
                is_mr,
                lambda _: self.calc_steady_state_mr(basal_rate, decay[:, idx]),
                lambda _: vmap_single_cell(
                    idx,
                    jnp.arange(n_cells),
                    is_mr,
                    basal_rate,
                    decay,
                    ki_matrix,
                    gene_conc,
                    all_cell_conc,
                    hill_coeff,
                    prot_trans,
                    prot_decay,
                    prot_conc,
                    prot_ss,
                ),
                operand=None,
            )

            return steady_vals[0], steady_vals[1]

        gene_conc, prot_conc = self.gene_conc, self.prot_conc
        for i in range(self.n_genes):
            g_conc, p_conc = _single_gene_steady_state(
                n_cells=self.n_cells,
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
                prot_conc=self.prot_conc,
                prot_ss=self.prot_steady_state,
            )
            gene_conc = gene_conc.at[i].set(g_conc)
            prot_conc = prot_conc.at[i].set(p_conc)
        return gene_conc, prot_conc

    def calc_pij(
        self, is_mr, idx, basal_rates, gene_conc, gene_cell_mean, k_i, _hill=1
    ):
        """Calculates the production rate of each gene as a function of its regulator genes as given in Equation 5, Equation 6 and Equation 7 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        Current assumption -
            Half life is same as E[x] for genes

        """

        def _calc_pij_g(gene_conc, gene_cell_mean, k_i, _hill=1):
            frac = jnp.pow(gene_conc, _hill) / (
                jnp.pow(gene_cell_mean, _hill) + jnp.pow(gene_conc, _hill)
            )

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
            lambda x: _calc_pij_g(
                gene_conc=gene_conc, gene_cell_mean=gene_cell_mean, k_i=k_i, _hill=_hill
            )
            + basal_rates,
            operand=None,
        )

    def calc_x_t(self):
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
        ):
            # gene_conc, delta, decay, steady_state, is_mr - Entire arrays passed for all genes in a cell
            p_i = (
                self.jit_pij(
                    is_mr=is_mr,
                    idx=idx,
                    basal_rates=basal_rates,
                    gene_conc=gene_conc,
                    gene_cell_mean=gene_cell_mean,
                    k_i=k_i,
                    _hill=hill_coeff,
                )
                + basal_rates
            )
            x_t_gene = gene_conc[idx] + (p_i - decay * gene_conc[idx]) * delta

            if self.noise:
                noise_add = noise_amp * (
                    jnp.sqrt(p_i) * self.noise_a.generate_noise()
                    + jnp.sqrt(decay * gene_conc[idx])
                )
                x_t_gene += noise_add

            x_t_prot = (
                prot_conc + (prot_kt * gene_conc[idx] - prot_kd * prot_conc) * delta
            )
            return x_t_gene, x_t_prot

        # Auto vectorization over all the genes
        auto_vec_genes = vmap(
            _single_gene_x_t, in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )

        # Auto vectorization over auto_vec_genes for all cells
        auto_vec_cells = vmap(
            auto_vec_genes,
            in_axes=(None, 1, None, 0, 1, 0, None, None, 0, 0, 1, 0, 0),
        )

        x_t, p_t = auto_vec_cells(
            jnp.arange(self.n_genes),
            self.gene_conc,
            self.delta,
            self.decay,
            self.basal_rates,
            self.ki_matrix,
            self.is_mr,
            jnp.mean(self.gene_conc, axis=1),
            self.noise_amp,
            self.hill_coeffs,
            self.prot_conc,
            self.prot_tran_rates,
            self.prot_decay,
        )
        x_t = x_t.reshape((self.n_cells, self.n_genes)).T
        p_t = p_t.reshape((self.n_cells, self.n_genes)).T

        return x_t, p_t

    def run_sim(self, n_steps):
        logging.info("Running simulator...")
        gene_conc_history = []
        prot_conc_history = []
        for _ in tqdm(range(n_steps)):
            self.gene_conc, self.prot_conc = self.jit_x_t()
            gene_conc_history.append(self.gene_conc)
            prot_conc_history.append(self.prot_conc)
        logging.info("Simulation ended...")
        return gene_conc_history, prot_conc_history
