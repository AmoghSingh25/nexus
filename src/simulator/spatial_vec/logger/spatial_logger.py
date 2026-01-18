import shutil
import os
import tiledb
import numpy as np


class SpatialLogger:
    """
    Logger class for storing data to tiledb
    """

    def __init__(
        self,
        log_dir,
        file_name,
        n_steps=None,
        n_cells=None,
        read_only=False,
    ):
        """
        Initialize Data Logger

        :param self: SpatialLogger
        :param log_dir: Directory of the logs
        :param file_name: Name of the tiledb array to be saved to
        :param n_steps: Number of simulation steps
        :param n_cells: Number of cells in the simulation
        :param n_chems: Number of chemicals in the simulation
        :param n_reactions: Number of reactions in the simulation
        """
        self.log_dir = log_dir
        if not os.path.exists(os.path.join(self.log_dir, file_name)) and not read_only:
            os.makedirs(os.path.join(self.log_dir, file_name), exist_ok=True)
        self.base_path = os.path.join(self.log_dir, file_name)
        self.pos_arr = os.path.join(self.base_path, "pos.tldb")

        if not read_only:
            self.n_steps = n_steps
            self.n_cells = n_cells
            self.pos_dtype = np.dtype(",".join(["float32"] * 3))

            self.cell_tile = min(2, self.n_cells)
            self.time_tile = min(2, self.n_steps)

            self._create_pos_arr()

        else:
            self.initialize_values()

    """ Helper functions """

    def initialize_values(self):
        with tiledb.open(self.pos_arr, "r") as _A:
            ret = _A[:, :, :]
            _A.close()
        self.n_cells = len(set(ret["cells"]))
        self.n_steps = len(set(ret["t"]))

    def _create_pos_arr(self):
        """
        Initialize the diffusion array filestore with the schema.

        :param self: SpatialLogger
        """
        d1 = tiledb.Dim(
            name="t", domain=(0, self.n_steps), tile=self.time_tile, dtype=np.int32
        )
        d2 = tiledb.Dim(
            name="cells",
            domain=(0, 10**10),
            tile=self.cell_tile,
            dtype=np.int64,
        )
        dom = tiledb.Domain(d1, d2)

        att1 = tiledb.Attr(name="pos", dtype=self.pos_dtype)
        att2 = tiledb.Attr(name="radius", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1, att2])

        tiledb.Array.create(self.pos_arr, sch)

    """ Logger functions """

    def log_cell_pos(self, step, pos, radius):
        """
        Logs the cell position

        :param self: SpatialLogger
        :param step: Simulation step index
        :param pos: Positions of the cell
        """

        step_idx = list([step]) * len(pos)
        cell_idx = np.indices(pos.shape[0:-1], sparse=False).ravel()
        pos = np.array(pos).view(self.pos_dtype)
        with tiledb.open(self.pos_arr, "w") as _A:
            _A[step_idx, cell_idx] = {"pos": pos, "radius": radius}
            _A.close()

    def retrieve_pos_data(self, mesh, step=None, cell=None):
        """
        Return post diffusion concentrations from the tiledb array. If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: SpatialLogger
        :param step: Simulation step index being queried
        """
        if step is None:
            step = list(range(self.n_steps))
        if cell is None:
            cell = list(range(mesh.n_cells))

        with tiledb.open(self.pos_arr, "r") as _A:
            ret = _A[step, cell]
            _A.close()

        return ret

    def cleanup(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
