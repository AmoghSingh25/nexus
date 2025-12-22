import shutil
import os
import tiledb
import numpy as np
from typing import List
from simulator.spatial.field.gridField import GridField


class DataLogger:
    """
    Logger class for storing data to tiledb
    """

    def __init__(
        self,
        log_dir,
        file_name,
        n_steps=None,
        n_cells=None,
        n_chems=None,
        n_reactions=None,
        read_only=False,
    ):
        """
        Initialize Data Logger

        :param self: DataLogger
        :param log_dir: Directory of the logs
        :param file_name: Name of the tiledb array to be saved to
        :param n_steps: Number of simulation steps
        :param n_cells: Number of cells in the simulation
        :param n_chems: Number of chemicals in the simulation
        :param n_reactions: Number of reactions in the simulation
        """
        self.log_dir = log_dir
        if not os.path.exists(os.path.join(self.log_dir, file_name)):
            os.mkdir(os.path.join(self.log_dir, file_name))
        base_path = os.path.join(self.log_dir, file_name)
        self.chem_arr = os.path.join(base_path, "chem.tldb")
        self.reaction_arr = os.path.join(base_path, "reaction.tldb")
        self.reaction_order_arr = os.path.join(base_path, "reaction_order.tldb")
        self.diffusion_arr = os.path.join(base_path, "diffusion.tldb")

        if not read_only:
            self.n_steps = n_steps
            self.n_cells = n_cells
            self.n_chems = n_chems
            self.n_reactions = n_reactions
            self.chem_dtype = np.dtype(",".join(["float32"] * self.n_chems))
            self.reaction_order_dtype = np.dtype(",".join(["i4"] * self.n_reactions))

            self.time_tile = min(2, self.n_steps)
            self.cell_tile = min(2, self.n_cells)
            self.chem_tile = min(2, self.n_chems)
            self.reaction_tile = min(2, self.n_reactions)

            self._create_chem_arr()
            self._create_diffusion_arr()
            self._create_reaction_arr()
        else:
            self.initialize_values()

    """ Helper functions """

    def initialize_values(self):
        with tiledb.open(self.reaction_arr, "r") as _A:
            ret = _A[:, :, :]
            _A.close()
        self.n_cells = len(set(ret["cells"]))
        self.n_steps = len(set(ret["t"]))
        self.n_reactions = len(set(ret["reaction_id"]))

        with tiledb.open(self.chem_arr, "r") as _A:
            ret = _A[:, :, :]
            _A.close()
        self.n_chems = len(set(ret["chem"]))

    def _create_diffusion_arr(self):
        """
        Initialize the diffusion array filestore with the schema.

        :param self: DataLogger
        """
        d1 = tiledb.Dim(
            name="t", domain=(0, self.n_steps), tile=self.time_tile, dtype=np.int32
        )
        d2 = tiledb.Dim(
            name="cells",
            domain=(0, self.n_cells - 1),
            tile=self.cell_tile,
            dtype=np.int32,
        )
        d3 = tiledb.Dim(
            name="post_diffusion",
            domain=(0, self.n_chems - 1),
            tile=self.chem_tile,
            dtype=np.float32,
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="conc", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.diffusion_arr, sch)

    def _create_chem_arr(self):
        """
        Initialize the chemical array filestore with the schema.

        :param self: DataLogger
        """
        d1 = tiledb.Dim(
            name="t", domain=(0, self.n_steps), tile=self.time_tile, dtype=np.int32
        )
        d2 = tiledb.Dim(
            name="cells",
            domain=(0, self.n_cells - 1),
            tile=self.cell_tile,
            dtype=np.int32,
        )
        d3 = tiledb.Dim(
            name="chem",
            domain=(0, self.n_chems - 1),
            tile=self.chem_tile,
            dtype=np.float32,
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="conc", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.chem_arr, sch)

    def _create_reaction_arr(self):
        """
        Initialize the reaction array filestore with the schema.

        :param self: DataLogger
        """
        d1 = tiledb.Dim(
            name="t", domain=(0, self.n_steps), tile=self.time_tile, dtype=np.int32
        )
        d2 = tiledb.Dim(
            name="cells",
            domain=(0, self.n_cells - 1),
            tile=self.cell_tile,
            dtype=np.int32,
        )
        d3 = tiledb.Dim(
            name="reaction_id",
            domain=(0, self.n_reactions - 1),
            tile=self.reaction_tile,
            dtype=np.int32,
        )
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="chem_conc", dtype=self.chem_dtype)

        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1])

        tiledb.Array.create(self.reaction_arr, sch)

        dom2 = tiledb.Domain(d1, d2)
        att2 = tiledb.Attr(name="reaction_order", dtype=self.reaction_order_dtype)
        sch2 = tiledb.ArraySchema(domain=dom2, sparse=True, attrs=[att2])
        tiledb.Array.create(self.reaction_order_arr, sch2)

    def get_cells_conc(self, cells: np.ndarray[GridField]):
        """
        Helper function to get chemical concentrations of cells.

        :param self: DataLogger
        :param cells: List of cells
        :type cells: np.ndarray[GridField]
        """
        chem_conc = []
        pos = []
        for idx in cells.keys():
            chem_conc.append(cells[idx].chem.chem_mass.reshape(-1))
            pos.append(cells[idx].pos)
        return np.array(chem_conc), pos

    """ Logger functions """

    def log_chem_state(self, step, cells: np.ndarray[GridField]):
        """
        Logs the chemical concentrations.

        :param self: DataLogger
        :param step: Simulation step index
        :param cells: List of cells to store chemical concentration
        :type cells: np.ndarray[GridField]
        """
        chem_conc, pos = self.get_cells_conc(cells=cells)
        cell_idx, chem_idx = np.indices(chem_conc.shape, sparse=False)

        chem_conc = chem_conc.ravel()
        cell_idx = cell_idx.ravel()
        chem_idx = chem_idx.ravel()
        step_idx = list([step]) * len(chem_idx)
        with tiledb.open(self.chem_arr, "w") as _A:
            _A[step_idx, cell_idx, chem_idx] = chem_conc
            _A.close()

    def log_diffusion_state(self, step, cells: np.ndarray[GridField]):
        """
        Logs the chemical concentrations post diffusion.

        :param self: DataLogger
        :param step: Simulation step index
        :param cells: List of cells to store chemical concentration
        :type cells: np.ndarray[GridField]
        """
        chem_conc, pos = self.get_cells_conc(cells=cells)
        with tiledb.open(self.diffusion_arr, "w") as _A:
            _A[step, :, :] = chem_conc
            _A.close()

    def log_reaction_state(self, step, cell_id, reaction_ids, chem_concs):
        """
        Logs the reactions performed along with chemical concentrations post the reactions.

        :param self: DataLogger
        :param step: Simulation step index
        :param cell_id: Index of the cell
        :param reaction_ids: List of reaction IDs that have have taken place
        :param chem_concs: Concentration of chemicals post reactions
        """
        if type(step) is not List:
            step = list([step]) * len(reaction_ids)
        if type(cell_id) is not List:
            cell_id = list([cell_id]) * len(reaction_ids)
        chem_concs = chem_concs.view(self.chem_dtype)
        with tiledb.open(self.reaction_arr, "w") as _A:
            _A[step, cell_id, reaction_ids] = chem_concs
            _A.close()

    def log_reaction_order(self, step, cell_id, reaction_order):
        """
        Log the reaction order at step_i and cell_i

        :param self: DataLogger
        :param step: Simulation step index
        :param cell_id: Index of the cell
        :param reaction_ids: List of reaction IDs in order of execution
        """
        reaction_order = reaction_order.view(self.reaction_order_dtype)

        with tiledb.open(self.reaction_order_arr, "w") as _A:
            _A[step, cell_id] = reaction_order
            _A.close()

    """ Retrieval functions """

    def retrieve_chem_data(self, step=None, cell_id=None, chem_id=None):
        """
        Return chemical concentrations from the tiledb array. If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: DataLogger
        :param step: Simulation step index being queried
        :param cell_id: ID of the cell being queried
        :param chem_id: ID of the chemical being queried
        """
        if step is None:
            step = list(range(self.n_steps))
        if cell_id is None:
            cell_id = list(range(self.n_cells))
        if chem_id is None:
            chem_id = list(range(self.n_chems))

        with tiledb.open(self.chem_arr, "r") as _A:
            ret = _A[step, cell_id, chem_id]
            _A.close()

        return ret

    def retrieve_diffusion_data(self, step=None):
        """
        Return post diffusion concentrations from the tiledb array. If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: DataLogger
        :param step: Simulation step index being queried
        """
        if step is None:
            step = list(range(self.n_steps))

        with tiledb.open(self.diffusion_arr, "r") as _A:
            ret = _A[step]
            _A.close()

        return ret

    def retrieve_reaction_data(self, step=None, reaction_id=None, cell_id=None):
        """
        Returns reaction-related data from the tiledb array. If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: DataLogger
        :param step: Simulation step index being queried
        :param reaction_id: ID of the reaction being queried
        :param cell_id: ID of the cell being queried
        """
        if step is None:
            step = list(range(self.n_steps))
        if reaction_id is None:
            reaction_id = list(range(self.n_reactions))
        if cell_id is None:
            cell_id = list(range(self.n_cells))

        with tiledb.open(self.reaction_arr, "r") as _A:
            ret = _A[step, cell_id, reaction_id]
            _A.close()

        return ret

    def retrieve_reaction_order(self, step=None, cell_id=None):
        """
        Returns the reaction order from the tiledb array.If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: DataLogger
        :param step: Simulation step index being queried
        :param cell_id: ID of the cell being queried
        """
        if step is None:
            step = list(range(self.n_steps))
        if cell_id is None:
            cell_id = list(range(self.n_cells))

        with tiledb.open(self.reaction_order_arr, "r") as _A:
            ret = _A[step, cell_id]
            _A.close()

        return ret

    def cleanup(self):
        if os.path.exists(self.chem_arr):
            shutil.rmtree(self.chem_arr)
        if os.path.exists(self.reaction_arr):
            shutil.rmtree(self.reaction_arr)
        if os.path.exists(self.diffusion_arr):
            shutil.rmtree(self.diffusion_arr)
