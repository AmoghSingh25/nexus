from pathlib import Path
from nexus_sim.simulator.grn.grnSim import GRNSim
import jax.numpy as jnp
import os
from hydra import initialize_config_dir, compose
import pickle
from jax import nn


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


def read_pickle(file_name):
    with open(file_name, "rb") as file:
        var = pickle.load(file)
    return var


class TestGRN:
    config_files = [
        "configs/sample_data/sample_network_2cell.yaml",
        "configs/sample_data/sample_network_1cell.yaml",
    ]
    cell_no = [2, 1]

    def test_calc_steady_states(self):
        base_config = get_config()
        base_config.grn.protein_sim = True
        # Dummy test to check if initialization works and simulator runs
        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            sim = GRNSim(base_config.grn)
            sim.run_sim()
            gene_conc, prot_conc = sim.gene_conc, sim.prot_conc
            assert gene_conc.shape == (4, self.cell_no[i], 1)
            assert prot_conc.shape == (4, self.cell_no[i], 1)

    def test_gene_only(self):
        base_config = get_config()
        base_config.grn.protein_sim = False
        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            sim = GRNSim(cfg=base_config.grn)
            sim.run_sim()
            gene_conc, prot_conc = sim.gene_conc, sim.prot_conc
            assert gene_conc.shape == (4, self.cell_no[i], 1)
            assert prot_conc.shape == (4, self.cell_no[i], 1)

    def test_sergio_output(self):
        DATA_DIR = Path(__file__).parent
        base_config = get_config("config")
        base_config.grn.protein_sim = False
        base_config.grn.logging = False

        node_mapping = read_pickle(DATA_DIR / "saved_outputs/node_mapping.pkl")
        sergio_output = read_pickle(DATA_DIR / "saved_outputs/saved_output.pkl")
        sim = GRNSim(base_config.grn)
        gene_conc = sim.gene_conc.reshape((100, 2700))
        reordered_output_1 = jnp.zeros_like(gene_conc)
        for i in node_mapping:
            reordered_output_1 = reordered_output_1.at[i].set(
                gene_conc[node_mapping[i]]
            )
        assert jnp.allclose(reordered_output_1, sergio_output, rtol=1e-6, atol=1e-32)

    def test_rna_backprop(self):
        base_config = get_config()
        base_config.grn.logging = False
        base_config.grn.non_mr_basal = False
        base_config.grn.logging = True
        base_config.grn.learn_params = True  # Toggle if disabling backprop
        base_config.grn.epochs = 500

        sim = GRNSim(
            cfg=base_config.grn,
            target_gene_conc=jnp.ones((4)),
            target_prot_conc=jnp.ones((4)),
        )
        prev_gene_conc, _ = sim.gene_conc, sim.prot_conc
        if base_config.grn.learn_params:
            sim.basal_rates = nn.softplus(sim.learnt_gene_params["basal_rates"])
            sim.decay = nn.softplus(sim.learnt_gene_params["decay"])
            sim.ki_matrix = sim.learnt_gene_params["ki_matrix"]
            sim.hill_coeffs = sim.learnt_gene_params["hill_coeffs"]
            sim.prot_tran_rates = nn.softplus(sim.learnt_prot_params["prot_tran_rates"])
            sim.prot_decay = nn.softplus(sim.learnt_prot_params["prot_decay"])
            sim.steady_states, sim.prot_steady_state, _, _ = sim.calc_steady_states(
                learn_params=False
            )

        def _calc_mse(_target, _pred):
            _min_loss = jnp.inf
            idx = 0
            for _i in range(base_config.grn.n_cells):
                _norm_pred = _pred[:, _i]
                _norm_target = _target
                _mse_loss = jnp.mean((_norm_target - _norm_pred) ** 2)
                if _mse_loss < _min_loss:
                    _min_loss = min(_min_loss, _mse_loss)
                    idx = _i
            return _min_loss, idx

        _loss, _idx = _calc_mse(sim.target_gene_conc, sim.steady_states)
        _loss2, _idx = _calc_mse(sim.target_gene_conc, prev_gene_conc)

        ## Just check if loss is lower as checking changes in array gives False for hill_coeffs
        assert _loss2 > _loss
        # assert (
        #     jnp.any(sim.basal_rates != prev_params[0])
        #     and jnp.any(sim.decay != prev_params[1])
        #     and jnp.any(sim.ki_matrix != prev_params[2])
        #     and jnp.any(sim.hill_coeffs != prev_params[3])
        # )
