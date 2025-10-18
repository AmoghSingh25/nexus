# Assumptions
# - The nodes are topologically sorted while being passed as input, i.e., parents of a node should have a lower idx than the child node
# - Uniform decay for all genes
# - Simulation starts from steady state calculation
# - hill coefficient is 1 for all interactions

import networkx as nx
import jax.numpy as jnp
from jax import vmap, random, lax


class simulator:
    def __init__(self, node_set, edges_set, n_cells=1):
        self.key, self.sub_key = random.split(random.key(42))
        self.n_cells = n_cells
        self.basal_rates = []
        self.ki_values = []
        self.regs = []
        self.transcription_rates = []
        self.n_genes = len(node_set)
        self.is_mr = []

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
        self.g.add_edges_from(edges_set)

        self.gene_conc = jnp.zeros((self.n_genes, self.n_cells, 1))
        self.steady_states = jnp.zeros_like(self.gene_conc)
        self.conn_matrix = jnp.zeros((self.n_genes, self.n_genes))
        self.ki_matrix = jnp.zeros((self.n_cells, self.n_genes, self.n_genes))
        for i in self.g.nodes(data=True):
            regs = sorted(self.g.predecessors(i[0]))
            if self.n_cells > 1:
                for cell in range(self.n_cells):
                    for idx in range(len(regs)):
                        self.ki_matrix = self.ki_matrix.at[cell, i[0], regs[idx]].set(
                            self.ki_values[i[0]][cell][idx]
                        )
                        # self.ki_matrix = self.ki_matrix.at[i[0], regs[idx]].set(self.ki_values[i[0]][idx])
                        self.conn_matrix = self.conn_matrix.at[i[0], regs[idx]].set(1)
            else:
                cell = 0
                for idx in range(len(regs)):
                    self.ki_matrix = self.ki_matrix.at[cell, i[0], regs[idx]].set(
                        self.ki_values[i[0]][idx]
                    )
                    # self.ki_matrix = self.ki_matrix.at[i[0], regs[idx]].set(self.ki_values[i[0]][idx])
                    self.conn_matrix = self.conn_matrix.at[i[0], regs[idx]].set(1)
            self.regs.append(jnp.array(regs))

        self.conn_matrix = jnp.repeat(
            self.conn_matrix.reshape((self.n_genes, self.n_genes, 1)), n_cells, axis=2
        )
        del self.ki_values

        self.basal_rates = jnp.array(self.basal_rates).reshape(
            self.n_genes, self.n_cells
        )
        self.transcription_rates = jnp.array(self.transcription_rates)
        self.is_mr = jnp.array(self.is_mr)
        self.decay = jnp.array([0.8])

        # print("START")
        # print(self.basal_rates.shape)
        # print(self.ki_matrix.shape)
        # print(self.conn_matrix.shape)
        # print(self.gene_conc.shape)
        # print(self.steady_states.shape)
        # print(self.is_mr.shape)
        # print("END")

        # if n_cells > 1:
        #     self.decay = jnp.repeat(self.decay, n_cells, axis=0)

        self.gene_conc = self.calc_steady_states()
        self.steady_states = self.gene_conc
        print("Steady state concentrations calculated.")

    def calc_steady_state_mr(self, b, decay):
        """Steady state calculation for MRs"""
        return b / decay

    def calc_steady_state_g(
        self, is_mr, idx, basal_rates, decay, steady_state, gene_conc, k_i
    ):
        """Steady state calculation for genes"""
        e_x = (
            self.calc_pij(is_mr, idx, basal_rates, steady_state, gene_conc, k_i) / decay
        )
        # p_kd = jnp.log(2) / p_gm
        # p_c = p_kt * e_x / p_kd
        # return e_x, p_kd, p_c
        return e_x

    def calc_steady_states(self):
        """Calculates the steady state concentrations for the MR and Gene nodes.
        The steady state concentrations are calculated using the method mentioned in Equation 8 and Equation 10 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.
        """
        print("Steady state calculations")

        def _single_cell_steady_state(
            n_genes, is_mr, basal_rates, decay, ki_matrix, gene_conc, steady_states
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
                    ),
                    operand=None,
                )
                gene_conc = gene_conc.at[i].set(steady_val)
                steady_states = gene_conc
            return gene_conc

        auto_vec_single_cell = vmap(
            _single_cell_steady_state, in_axes=(None, None, 1, None, 0, 1, 1)
        )
        ret = auto_vec_single_cell(
            self.n_genes,
            self.is_mr,
            self.basal_rates,
            self.decay,
            self.ki_matrix,
            self.gene_conc,
            self.steady_states,
        )
        ret = ret.reshape((self.n_cells, self.n_genes)).T
        return ret

    def calc_pij(self, is_mr, idx, basal_rates, steady_state, gene_conc, k_i, _hill=1):
        """Calculates the production rate of each gene as a function of its regulator genes as given in Equation 5, Equation 6 and Equation 7 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.

        Current assumption -
            Half life is same as E[x] for genes

        """

        def _calc_pij_g(steady_state, gene_conc, k_i, _hill=1):
            steady_state = steady_state.reshape(-1)
            gene_conc = gene_conc.reshape(-1)
            # debug.print("ss = {x}, \n gc = {y}", x=steady_state, y=gene_conc)
            frac = gene_conc**_hill / (steady_state**_hill + gene_conc**_hill)
            frac = jnp.nan_to_num(frac, nan=0.0, neginf=0.0, posinf=0.0).reshape(-1)
            frac = frac * k_i
            # debug.print("k = {x}, \n s_pij = {y}", x=k_i, y=frac.sum(axis=0))
            return frac.sum(axis=0)

        return lax.cond(
            is_mr,
            lambda x: basal_rates,
            lambda x: _calc_pij_g(steady_state, gene_conc, k_i),
            (),
        )

    def calc_x_t(self, delta=0.1):
        """Estimate the concentration of each gene at the next time step using Euler-Maruyama method given in Equation 3 in
        Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.
        """
        # # node = self.g.nodes()[idx]
        # p_i = self.calc_pij(idx) + self.basal_rates[idx]
        # x_t = self.gene_conc[idx] + (p_i - self.decay * self.gene_conc[idx]) * delta
        # self.gene_conc = self.gene_conc.at[idx].set(x_t)

        def _single_gene_x_t(
            idx, gene_conc, delta, decay, basal_rates, k_i, is_mr, steady_state
        ):
            """
            gene_conc, delta, decay, steady_state, is_mr - Pass cell wide arrays
            """
            p_i = (
                self.calc_pij(is_mr, idx, basal_rates, steady_state, gene_conc, k_i)
                + basal_rates
            )
            x_t_gene = gene_conc[idx] + (p_i - decay * gene_conc[idx]) * delta
            return x_t_gene

        auto_vec_genes = vmap(
            _single_gene_x_t, in_axes=(0, None, None, None, 0, 0, 0, 0)
        )
        auto_vec_cells = vmap(
            auto_vec_genes, in_axes=(None, 1, None, None, 1, 0, None, 1)
        )
        x_t = auto_vec_cells(
            jnp.arange(self.n_genes),
            self.gene_conc,
            delta,
            self.decay,
            self.basal_rates,
            self.ki_matrix,
            self.is_mr,
            self.steady_states,
        )
        x_t = x_t.reshape((self.n_cells, self.n_genes)).T
        self.gene_conc = x_t

    def run_sim(self, n_steps):
        print("Starting simulator...")
        gene_conc_history = []
        # prot_conc_history = []
        for _ in range(n_steps):
            self.calc_x_t()
            gene_conc_history.append(self.gene_conc)
            # prot_conc_history.append()
        # return gene_conc_history, prot_conc_history
        print("Simulation ended...")
        return gene_conc_history, []
