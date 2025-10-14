from simulator import gpsim


def test_calc_steady_states():
    # Dummy test to check if initialization works and simulator runs
    sim = gpsim.simulator()
    gene_conc, prot_conc = sim.run_sim(10)
    print(gene_conc.shape, prot_conc.shape)
    assert gene_conc.shape == (10, 4)
    assert prot_conc.shape == (10, 2)
