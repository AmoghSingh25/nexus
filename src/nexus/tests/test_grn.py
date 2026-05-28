from utils import log_cleanup
from pathlib import Path
from nexus.simulator.grn.grnSim import GRNSim
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

    @log_cleanup
    def test_calc_steady_states(self):
        base_config = get_config()
        base_config.grn.protein_sim = True
        # Dummy test to check if initialization works and simulator runs
        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            self.sim = GRNSim(base_config.grn)
            self.sim.run_sim()
            gene_conc, prot_conc = self.sim.gene_conc, self.sim.prot_conc
            self.sim.cleanup()
            assert gene_conc.shape == (4, self.cell_no[i], 1)
            assert prot_conc.shape == (4, self.cell_no[i], 1)

    @log_cleanup
    def test_gene_only(self):
        base_config = get_config()
        base_config.grn.protein_sim = False
        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            self.sim = GRNSim(cfg=base_config.grn)
            self.sim.run_sim()
            gene_conc, prot_conc = self.sim.gene_conc, self.sim.prot_conc
            self.sim.cleanup()
            assert gene_conc.shape == (4, self.cell_no[i], 1)
            assert prot_conc.shape == (4, self.cell_no[i], 1)

    @log_cleanup
    def test_sergio_output(self):
        DATA_DIR = Path(__file__).parent
        base_config = get_config("config")
        base_config.grn.protein_sim = False
        base_config.grn.logging = False

        node_mapping = read_pickle(DATA_DIR / "saved_outputs/node_mapping.pkl")
        sergio_output = read_pickle(DATA_DIR / "saved_outputs/saved_output.pkl")
        self.sim = GRNSim(base_config.grn)
        gene_conc = self.sim.gene_conc.reshape((100, 2700))
        reordered_output_1 = jnp.zeros_like(gene_conc)
        for i in node_mapping:
            reordered_output_1 = reordered_output_1.at[i].set(
                gene_conc[node_mapping[i]]
            )
        self.sim.cleanup()
        assert jnp.allclose(reordered_output_1, sergio_output, rtol=1e-6, atol=1e-32)

    @log_cleanup
    def test_rna_backprop(self):
        base_config = get_config()
        base_config.grn.non_mr_basal = False
        base_config.grn.logging = False
        base_config.grn.learn_params = True  # Toggle if disabling backprop
        base_config.grn.epochs = 500

        self.sim = GRNSim(
            cfg=base_config.grn,
            target_gene_conc=jnp.ones((4)),
            target_prot_conc=jnp.ones((4)),
        )
        prev_gene_conc, _ = self.sim.gene_conc, self.sim.prot_conc
        if base_config.grn.learn_params:
            self.sim.basal_rates = nn.softplus(
                self.sim.learnt_gene_params["basal_rates"]
            )
            self.sim.decay = nn.softplus(self.sim.learnt_gene_params["decay"])
            self.sim.ki_matrix = self.sim.learnt_gene_params["ki_matrix"]
            self.sim.hill_coeffs = self.sim.learnt_gene_params["hill_coeffs"]
            self.sim.prot_tran_rates = nn.softplus(
                self.sim.learnt_prot_params["prot_tran_rates"]
            )
            self.sim.prot_decay = nn.softplus(self.sim.learnt_prot_params["prot_decay"])
            self.sim.steady_states, self.sim.prot_steady_state, _, _ = (
                self.sim.calc_steady_states(learn_params=False)
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

        _loss, _idx = _calc_mse(self.sim.target_gene_conc, self.sim.steady_states)
        _loss2, _idx = _calc_mse(self.sim.target_gene_conc, prev_gene_conc)
        self.sim.cleanup()

        assert _loss2 > _loss

    @log_cleanup
    def test_noise_replay(self):
        ## Test on default configs
        base_config = get_config()
        base_config.grn.protein_sim = False
        base_config.grn.random_key = 100
        base_config.grn.noise = True

        def _compare_runs(inp_cfg):
            run1 = GRNSim(cfg=inp_cfg.grn)
            ret1 = run1.run_sim()
            run1_gene_traj = jnp.array(ret1.gene_traj)
            run1_prot_traj = jnp.array(ret1.prot_traj)
            run1.cleanup()

            run2 = GRNSim(cfg=inp_cfg.grn)
            ret2 = run2.run_sim()
            run2_gene_traj = jnp.array(ret2.gene_traj)
            run2_prot_traj = jnp.array(ret2.prot_traj)
            run2.cleanup()

            ## Check using `noise_trace` gives same output
            inp_cfg.grn.random_key = 42
            run3 = GRNSim(cfg=inp_cfg.grn)
            ret3 = run3.run_sim(noise_trace=ret2.noise_trace)
            run3_gene_traj = jnp.array(ret3.gene_traj)
            run3_prot_traj = jnp.array(ret3.prot_traj)
            run3.cleanup()

            assert (
                jnp.all(run1_gene_traj == run2_gene_traj)
                and jnp.all(run1_prot_traj == run2_prot_traj)
                and jnp.all(ret1.noise_trace == ret2.noise_trace)
                and jnp.all(run3_gene_traj == run2_gene_traj)
                and jnp.all(run3_prot_traj == run2_prot_traj)
            )

        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            _compare_runs(base_config)

        ## Test on SERGIO config
        base_config = get_config("config")
        base_config.grn.protein_sim = False
        base_config.grn.logging = False
        base_config.grn.noise = True
        base_config.grn.random_key = 101

        _compare_runs(base_config)
