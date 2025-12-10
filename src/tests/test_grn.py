from simulator import grnSim
import numpy as np
import jax.numpy as jnp
import os
from hydra import initialize_config_dir, compose


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


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
            sim = grnSim.GRNSim(base_config.grn)
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
            sim = grnSim.GRNSim(cfg=base_config.grn)
            gene_conc, prot_conc = sim.run_sim()
            gene_conc, prot_conc = jnp.array(gene_conc), jnp.array(prot_conc)
            assert gene_conc.shape == (10, 4, self.cell_no[i])
            assert prot_conc.shape == (10, 4, self.cell_no[i])
