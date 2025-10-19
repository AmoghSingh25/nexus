from simulator import gpsim
import numpy as np


def test_calc_steady_states():
    # Dummy test to check if initialization works and simulator runs
    sim = gpsim.simulator(
        config_file="configs/sample_data/sample_network_2cell.yaml", n_cells=2
    )
    gene_conc, prot_conc = sim.run_sim(10)
    gene_conc, prot_conc = np.array(gene_conc), np.array(prot_conc)
    print(gene_conc.shape, prot_conc.shape)
    assert gene_conc.shape == (10, 4, 2)
    assert prot_conc.shape == (10, 4, 2)
