from simulator import gpsim
import numpy as np
import jax.numpy as jnp


class TestFile:
    config_files = [
        "configs/sample_data/sample_network_2cell.yaml",
        "configs/sample_data/sample_network_1cell.yaml",
    ]
    cell_no = [2, 1]

    def test_calc_steady_states(self):
        # Dummy test to check if initialization works and simulator runs
        for i in range(len(self.config_files)):
            sim = gpsim.simulator(
                config_file=self.config_files[i], n_cells=self.cell_no[i]
            )
            gene_conc, prot_conc = sim.run_sim(10)
            gene_conc, prot_conc = np.array(gene_conc), np.array(prot_conc)
            print(gene_conc.shape, prot_conc.shape)
            assert gene_conc.shape == (10, 4, self.cell_no[i])
            assert prot_conc.shape == (10, 4, self.cell_no[i])

    def test_gene_only(self):
        for i in range(len(self.config_files)):
            sim = gpsim.simulator(
                config_file=self.config_files[i],
                n_cells=self.cell_no[i],
                protein_sim=False,
            )
            gene_conc, prot_conc = sim.run_sim(10)
            gene_conc, prot_conc = jnp.array(gene_conc), jnp.array(prot_conc)
            print(gene_conc.shape, prot_conc.shape)
            assert gene_conc.shape == (10, 4, self.cell_no[i])
            assert prot_conc.shape == (10, 4, self.cell_no[i])
