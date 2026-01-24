import pickle
import shutil
import os
import tiledb
import numpy as np


class GRNLogger:
    """
    Logger class for storing GRN simulator data to tiledb
    """

    def __init__(
        self,
        log_dir,
        file_name,
        n_steps=None,
        n_cells=None,
        n_genes=None,
        read_only=False,
    ):
        """
        Initialize GRN Logger

        :param self: GRNLogger
        :param log_dir: Directory of the logs
        :param file_name: Name of the tiledb array to be saved to
        :param n_steps: Number of simulation steps
        :param n_cells: Number of cells in the simulation
        :param read_only: Boolean flag to indicate if the logger must be created or read from arrays
        """
        self.log_dir = log_dir
        if not os.path.exists(os.path.join(self.log_dir, file_name)) and not read_only:
            os.makedirs(os.path.join(self.log_dir, file_name), exist_ok=True)
        self.base_path = os.path.join(self.log_dir, file_name)
        self.prop_pkl = os.path.join(self.base_path, "properties.pkl")
        self.conc_arr = os.path.join(self.base_path, "conc_arr.tldb")

        if not read_only:
            self.n_steps = n_steps
            self.n_cells = n_cells
            self.n_genes = n_genes

            self.cell_tile = min(2, self.n_cells)
            self.gene_tile = min(2, self.n_genes)
            self.step_tile = min(2, self.n_steps)

            self._create_prop_file()
            self._create_conc_arr()
        else:
            self.initialize_values()

    """ Helper functions """

    def initialize_values(self):
        with open(self.prop_pkl, "rb") as file:
            metadata_dict = pickle.load(file)

        self.n_steps = metadata_dict["n_steps"]
        self.n_cells = metadata_dict["n_cells"]

    def _create_prop_file(self):
        """
        Creates the pickle file containing the metadata of the logs.

        :param self: GRNLogger
        """
        metadata_dict = {
            "n_cells": self.n_cells,
            "n_steps": self.n_steps,
        }
        with open(self.prop_pkl, "wb") as file:
            pickle.dump(metadata_dict, file)

    def _create_conc_arr(self):
        """
        Initialize the concentration array filestore with the schema.

        :param self: GRNLogger
        """
        d1 = tiledb.Dim(
            name="t",
            domain=(0, self.n_steps),
            tile=self.step_tile,
            dtype=np.int32,
        )
        d2 = tiledb.Dim(
            name="cell_id",
            domain=(0, self.n_cells),
            tile=self.cell_tile,
            dtype=np.int32,
        )
        d3 = tiledb.Dim(name="gene_id", domain=(0, self.n_genes), dtype=np.int32)
        dom = tiledb.Domain(d1, d2, d3)

        att1 = tiledb.Attr(name="gene_conc", dtype=np.float32)
        att2 = tiledb.Attr(name="prot_conc", dtype=np.float32)
        sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[att1, att2])

        tiledb.Array.create(self.conc_arr, sch)

    """ Logger functions """

    def log_conc(self, step, cell, gene_conc, prot_conc):
        """
        Logs the concentrations of protein and gene.

        :param self: GRNLogger
        :param step: Current step index
        :param cell: Cell ID
        :param gene_id: Gene ID
        :param gene_conc: Gene concentration
        :param prot_conc: Protein concentration
        """
        gene_id = np.indices(gene_conc.shape, sparse=False)[0].ravel()
        cell_ids = np.indices(gene_conc.shape, sparse=False)[1].ravel()
        step = list([step]) * len(gene_id)

        with tiledb.open(self.conc_arr, "w") as _A:
            _A[step, cell_ids, gene_id] = {
                "gene_conc": gene_conc,
                "prot_conc": prot_conc,
            }
            _A.close()

    def retrieve_conc(self, step=None, cells=None, gene_id=None):
        """
        Returns the reaction order from the tiledb array.If any of the parameters are not set, data is returned across the entire possible range for this parameter.

        :param self: GRNLogger
        :param step: Simulation step index being queried
        :param field_id: ID of the cell being queried
        """
        if step is None:
            step = list(range(self.n_steps))
        if cells is None:
            cells = list(range(self.n_cells))
        if gene_id is None:
            genes = list(range(self.n_genes))

        with tiledb.open(self.conc_arr, "r") as _A:
            ret = _A[step, cells, genes]
            _A.close()

        return ret

    def cleanup(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
