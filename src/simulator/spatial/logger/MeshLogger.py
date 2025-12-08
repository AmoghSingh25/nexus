import os
import tiledb
import numpy as np
from typing import List
from simulator.spatial.cell.gridCell import GridCell


class DataLogger:
    def __init__(self, log_dir, file_name, n_steps, n_cells, n_chems, n_reactions):
        self.log_dir = log_dir
        self.chem_arr = os.path.join(self.log_dir, file_name + "_chem.tldb")
        self.reaction_arr = os.path.join(self.log_dir, file_name + "_reaction.tldb")
        self.diffusion_arr = os.path.join(self.log_dir, file_name + "_diffusion.tldb")

        self.n_steps = n_steps
        self.n_cells = n_cells
        self.n_chems = n_chems
        self.n_reactions = n_reactions

        self._create_chem_arr()
        self._create_diffusion_arr()
        self._create_reaction_arr()

    def _create_diffusion_arr(self):
        d1 = tiledb.Dim(name="t", domain=(0, self.n_steps), tile=2, dtype=np.int32)
        d2 = tiledb.Dim(
            name="cells", domain=(0, self.n_cells - 1), tile=2, dtype=np.int32
        )
        d3 = tiledb.Dim(
            name="post_diffusion",
            domain=(0, self.n_chems - 1),
            tile=2,
            dtype=np.float32,
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="conc", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.diffusion_arr, sch)

    def _create_chem_arr(self):
        d1 = tiledb.Dim(name="t", domain=(0, self.n_steps), tile=2, dtype=np.int32)
        d2 = tiledb.Dim(
            name="cells", domain=(0, self.n_cells - 1), tile=2, dtype=np.int32
        )
        d3 = tiledb.Dim(
            name="chem", domain=(0, self.n_chems - 1), tile=2, dtype=np.float32
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="conc", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.chem_arr, sch)

    def _create_reaction_arr(self):
        d1 = tiledb.Dim(name="t", domain=(0, self.n_steps), tile=2, dtype=np.int32)
        d2 = tiledb.Dim(
            name="cells", domain=(0, self.n_cells - 1), tile=2, dtype=np.int32
        )
        d3 = tiledb.Dim(
            name="reaction_id", domain=(0, self.n_reactions - 1), tile=2, dtype=np.int32
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="chem_conc", dtype=np.float32, var=True)

        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.reaction_arr, sch)

    def get_cells_conc(self, cells: np.ndarray[GridCell]):
        cells = cells.reshape(-1)
        chem_conc = np.zeros((self.n_cells, self.n_chems))
        pos = []
        for idx in range(len(cells)):
            chem_conc[idx] = cells[idx].chem.chem_mass.reshape(-1)
            pos.append(cells[idx].pos)
        return chem_conc, pos

    """ Logger functions """

    def log_chem_state(self, step, cells: np.ndarray[GridCell]):
        chem_conc, pos = self.get_cells_conc(cells=cells)
        cell_idx, chem_idx = np.indices(chem_conc.shape, sparse=False)

        chem_conc = chem_conc.ravel()
        cell_idx = cell_idx.ravel()
        chem_idx = chem_idx.ravel()
        step_idx = list([step]) * len(chem_idx)
        with tiledb.open(self.chem_arr, "w") as _A:
            _A[step_idx, cell_idx, chem_idx] = chem_conc
            _A.close()

    def log_diffusion_state(self, step, cells: np.ndarray[GridCell]):
        chem_conc, pos = self.get_cells_conc(cells=cells)
        with tiledb.open(self.diffusion_arr, "w") as _A:
            _A[step, :, :] = chem_conc
            _A.close()

    def log_reaction_state(self, step, cell_id, reaction_ids, chem_concs):
        if type(step) is not List:
            step = list([step]) * len(reaction_ids)
        if type(cell_id) is not List:
            cell_id = list([cell_id]) * len(reaction_ids)

        print(chem_concs)
        with tiledb.open(self.reaction_arr, "w") as _A:
            _A[step, cell_id, reaction_ids] = chem_concs
            _A.close()

    """ Retrieval functions """

    def retrieve_chem_data(self, step=None):
        if step is None:
            step = list(range(self.n_steps))

        with tiledb.open(self.chem_arr, "r") as _A:
            ret = _A[step]
            _A.close()

        return ret
