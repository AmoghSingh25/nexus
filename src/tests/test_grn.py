from simulator.grn.grnSim import GRNSim
import numpy as np
import jax.numpy as jnp
import os
from hydra import initialize_config_dir, compose
import pickle


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


def read_pickle(file_name):
    with open(file_name, "rb") as file:
        var = pickle.load(file)
    return var


class TestFile:
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
            gene_conc, prot_conc = sim.run_sim()
            gene_conc, prot_conc = np.array(gene_conc), np.array(prot_conc)
            assert gene_conc.shape == (10, 4, self.cell_no[i])
            assert prot_conc.shape == (10, 4, self.cell_no[i])

    def test_gene_only(self):
        base_config = get_config()
        base_config.grn.protein_sim = False
        for i in range(len(self.config_files)):
            base_config.grn.config_file = self.config_files[i]
            base_config.grn.n_cells = self.cell_no[i]
            sim = GRNSim(cfg=base_config.grn)
            gene_conc, prot_conc = sim.run_sim()
            gene_conc, prot_conc = jnp.array(gene_conc), jnp.array(prot_conc)
            assert gene_conc.shape == (10, 4, self.cell_no[i])
            assert prot_conc.shape == (10, 4, self.cell_no[i])

    def validate_sergio_output(self):
        base_config = get_config()
        base_config.grn.protein_sim = False
        base_config.grn.logging = False

        node_mapping = read_pickle("saved_outputs/node_mapping.pkl")
        sergio_output = read_pickle("saved_outputs/saved_output.pkl")
        sim = GRNSim(base_config.grn)
        gene_conc = sim.gene_conc.reshape((100, 2700))
        reordered_output_1 = jnp.zeros_like(gene_conc)
        for i in node_mapping:
            reordered_output_1 = reordered_output_1.at[i].set(
                gene_conc[node_mapping[i]]
            )
        assert jnp.allclose(reordered_output_1, sergio_output, rtol=1e-6, atol=1e-32)
