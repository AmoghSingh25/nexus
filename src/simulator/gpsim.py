# Assumptions
# - The nodes are topologically sorted while being passed as input, i.e., parents of a node should have a lower idx than the child node
# - Uniform decay for all genes
# - Simulation starts from steady state calculation
# - hill coefficient is 1 for all interactions

import networkx as nx
import jax.numpy as jnp
from jax import vmap, random, lax, jit
from .utils.verify_network import _verify_network
from .utils.read_network import _read_data
from tqdm import tqdm
import logging


class simulator:
    def __init__(
        self, gene_data=None, mr_data=None, config_file="", n_cells=1, protein_sim=True
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
        """

        node_set, edges_set = _read_data(gene_data, mr_data, config_file, n_cells)

        logging.info("Running network checks...")
        if _verify_network(node_set, edges_set, n_cells):
            logging.info("Network checks passed")
        self.key, self.sub_key = random.split(random.key(42))

        self.n_genes = len(node_set)
        self.n_cells = n_cells
        self.protein_sim = protein_sim

        self.basal_rates = []
        self.ki_values = []
        self.is_mr = []
        self.gene_conc = jnp.zeros((self.n_genes, self.n_cells, 1))
        self.steady_states = jnp.zeros_like(self.gene_conc)
        self.ki_matrix = jnp.zeros((self.n_cells, self.n_genes, self.n_genes))

        self.prot_conc = jnp.zeros_like(self.gene_conc)
        self.prot_steady_state = jnp.zeros_like(self.gene_conc)
        self.prot_tran_rates = jnp.zeros_like(self.gene_conc)
        self.prot_decay = jnp.zeros_like(self.gene_conc)
        self.prot_half_lives = jnp.zeros_like(self.gene_conc)

        self.g = nx.DiGraph()

        for i in range(len(node_set)):
            node = node_set[i]
            self.is_mr.append(True if node["type"] == "mr" else False)
            if node["type"] == "mr":
                self.g.add_node(i)
                self.basal_rates.append(
                    jnp.array(node["basal_rate"]).reshape((n_cells, 1))
                )
                self.ki_values.append(jnp.array([0]))

            else:
                basal_rate_i = random.uniform(
                    self.sub_key, (n_cells, 1), minval=0.1, maxval=1.0
                )
                self.key, self.sub_key = random.split(self.key)
                self.g.add_node(i)
                self.basal_rates.append(basal_rate_i)
                self.ki_values.append(jnp.array(node["ki"]))

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

        self.g.add_edges_from(edges_set)

        for i in self.g.nodes(data=True):
            regs = sorted(self.g.predecessors(i[0]))
            if self.n_cells > 1:
                for cell in range(self.n_cells):
                    for idx in range(len(regs)):
                        self.ki_matrix = self.ki_matrix.at[cell, i[0], regs[idx]].set(
                            self.ki_values[i[0]][cell][idx]
                        )
            else:
                cell = 0
                for idx in range(len(regs)):
                    self.ki_matrix = self.ki_matrix.at[cell, i[0], regs[idx]].set(
                        self.ki_values[i[0]][idx]
                    )

        del self.ki_values, self.g

        self.basal_rates = jnp.array(self.basal_rates).reshape(
            self.n_genes, self.n_cells
        )
        self.is_mr = jnp.array(self.is_mr)
        self.decay = jnp.array([0.8])

        ## Create JIT functions
        self.jit_pij = jit(self.calc_pij)
        self.jit_x_t = jit(self.calc_x_t)

        self.gene_conc, self.prot_conc = self.calc_steady_states()
        self.steady_states = self.gene_conc
        self.prot_steady_state = self.prot_conc

        logging.info("Steady state concentrations calculated.")

    def calc_steady_state_mr(self, b, decay):
        """Steady state calculation for MRs"""
        return b / decay, jnp.array([0.0])

    def calc_steady_state_g(
        self, is_mr, idx, basal_rates, decay, steady_state, gene_conc, k_i, p_kt, p_kd
    ):
        """Steady state calculation for genes and proteins"""
        e_x = (
            self.jit_pij(is_mr, idx, basal_rates, steady_state, gene_conc, k_i) / decay
        )
        if self.protein_sim:
            p_c = p_kt * e_x / p_kd
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

        def _single_cell_steady_state(
            n_genes,
            is_mr,
            basal_rates,
            decay,
            ki_matrix,
            gene_conc,
            steady_states,
            prot_trans,
            prot_decay,
            prot_conc,
            prot_ss,
        ):
            for i in range(n_genes):
                steady_val = lax.cond(
                    is_mr[i],
                    lambda _: self.calc_steady_state_mr(basal_rates[i], decay),
                    lambda _: self.calc_steady_state_g(
                        is_mr[i],
                        i,
                        basal_rates[i],
                        decay,
                        steady_states,
                        gene_conc,
                        ki_matrix[i],
                        prot_trans[i],
                        prot_decay[i],
                    ),
                    operand=None,
                )
                gene_conc = gene_conc.at[i].set(steady_val[0])
                if self.protein_sim:
                    prot_conc = prot_conc.at[i].set(steady_val[1])

                else:
                    prot_conc = jnp.zeros_like(gene_conc)

                steady_states = gene_conc
            return gene_conc, prot_conc

        auto_vec_single_cell = vmap(
            _single_cell_steady_state,
            in_axes=(None, None, 1, None, 0, 1, 1, 0, 0, 1, 1),
        )
        gene_cells_conc, prot_cells_conc = auto_vec_single_cell(
            self.n_genes,
            self.is_mr,
            self.basal_rates,
            self.decay,
            self.ki_matrix,
            self.gene_conc,
            self.steady_states,
            self.prot_tran_rates,
            self.prot_decay,
            self.prot_conc,
            self.prot_steady_state,
        )
        gene_cells_conc = gene_cells_conc.reshape((self.n_cells, self.n_genes)).T
        prot_cells_conc = prot_cells_conc.reshape((self.n_cells, self.n_genes)).T
        return gene_cells_conc, prot_cells_conc

    def calc_pij(self, is_mr, idx, basal_rates, steady_state, gene_conc, k_i, _hill=1):
        """Calculates the production rate of each gene as a function of its regulator genes as given in Equation 5, Equation 6 and Equation 7 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        Current assumption -
            Half life is same as E[x] for genes

        """

        def _calc_pij_g(steady_state, gene_conc, k_i, _hill=1):
            steady_state = steady_state.reshape(-1)
            gene_conc = gene_conc.reshape(-1)
            frac = gene_conc**_hill / (steady_state**_hill + gene_conc**_hill)
            frac = jnp.nan_to_num(frac, nan=0.0, neginf=0.0, posinf=0.0).reshape(-1)
            frac = frac * k_i
            return frac.sum(axis=0)

        return lax.cond(
            is_mr,
            lambda x: basal_rates,
            lambda x: _calc_pij_g(steady_state, gene_conc, k_i),
            operand=None,
        )

    def calc_x_t(self, delta=0.1):
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
            steady_state,
            prot_conc,
            prot_kt,
            prot_kd,
        ):
            # gene_conc, delta, decay, steady_state, is_mr - Entire arrays passed for all genes in a cell
            p_i = (
                self.jit_pij(is_mr, idx, basal_rates, steady_state, gene_conc, k_i)
                + basal_rates
            )
            x_t_gene = gene_conc[idx] + (p_i - decay * gene_conc[idx]) * delta

            x_t_prot = (
                prot_conc + (prot_kt * gene_conc[idx] - prot_kd * prot_conc) * delta
            )
            return x_t_gene, x_t_prot

        # Auto vectorization over all the genes
        auto_vec_genes = vmap(
            _single_gene_x_t, in_axes=(0, None, None, None, 0, 0, 0, 0, 0, 0, 0)
        )

        # Auto vectorization over auto_vec_genes for all cells
        auto_vec_cells = vmap(
            auto_vec_genes, in_axes=(None, 1, None, None, 1, 0, None, 1, 1, 0, 0)
        )

        x_t, p_t = auto_vec_cells(
            jnp.arange(self.n_genes),
            self.gene_conc,
            delta,
            self.decay,
            self.basal_rates,
            self.ki_matrix,
            self.is_mr,
            self.steady_states,
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
            # self.calc_x_t()
            self.gene_conc, self.prot_conc = self.jit_x_t()
            gene_conc_history.append(self.gene_conc)
            prot_conc_history.append(self.prot_conc)
        logging.info("Simulation ended...")
        return gene_conc_history, prot_conc_history
