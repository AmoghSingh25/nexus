from simulator import gpsim
import numpy as np


def test_calc_steady_states():
    # Dummy test to check if initialization works and simulator runs
    node_set = [
        {"basal_rate": [0.20165, 0.19165], "type": "mr"},
        {"basal_rate": [0.25316, 0.21132], "type": "mr"},
        {
            "ki": [
                [1, 2],  # Cell 1
                [3, 4],
            ],  # Cell 2
            "prot_half_life": [3, 4],
            "prot_transcription_rate": [10, 11],
            "type": "g",
        },
        {
            "ki": [[2.34309], [1.64324]],
            "prot_half_life": [2, 1],
            "prot_transcription_rate": [12, 13],
            "type": "g",
        },
    ]
    edges_set = [[0, 2], [1, 2], [2, 3]]
    sim = gpsim.simulator(node_set=node_set, edges_set=edges_set, n_cells=2)
    gene_conc, prot_conc = sim.run_sim(10)
    gene_conc, prot_conc = np.array(gene_conc), np.array(prot_conc)
    print(gene_conc.shape, prot_conc.shape)
    assert gene_conc.shape == (10, 4, 2)
    assert prot_conc.shape == (10, 4, 2)
