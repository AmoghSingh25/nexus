import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import tiledb
    import numpy as np
    import shutil
    import os

    print(tiledb.version())
    print(tiledb.libtiledb.version())
    return np, os, shutil, tiledb


@app.cell
def _():
    return


@app.cell
def _(np):
    _temp = np.ones((10, 10))
    _temp2 = np.zeros((10, 10))
    uri = "src/simulator/spatial/logs/test.tldb"
    return (uri,)


@app.cell
def _(os, shutil, uri):
    if os.path.exists(uri):
        shutil.rmtree(uri)
    return


@app.cell
def _():
    n_steps = 10
    n_cells = 3
    n_chems = 3
    return n_cells, n_chems, n_steps


@app.cell
def _(n_cells, n_chems, n_steps, np, tiledb, uri):
    d1 = tiledb.Dim(name="t", domain=(0, n_steps - 1), tile=2, dtype=np.int32)
    d2 = tiledb.Dim(name="cells", domain=(0, n_cells - 1), tile=2, dtype=np.int32)
    d3 = tiledb.Dim(name="chems", domain=(0, n_chems - 1), tile=2, dtype=np.int32)

    dom = tiledb.Domain(d1, d2, d3)

    a = tiledb.Attr(name="conc", dtype=np.int32)
    sch = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[a])

    tiledb.Array.create(uri, sch)
    return


@app.cell
def _(np):
    d1_data = np.array([0, 0], dtype=np.int32)
    d2_data = np.array([0, 0], dtype=np.int32)
    d3_data = np.array([0, 2], dtype=np.int32)
    a_data = np.array([1, 2], dtype=np.int32)
    return a_data, d1_data, d2_data, d3_data


@app.cell
def _(a_data, d1_data, d2_data, d3_data, tiledb, uri):
    # Open the array in write mode and write the data in COO format
    with tiledb.open(uri, "w") as _A:
        _A[d1_data, d2_data, d3_data] = a_data
    return


@app.cell
def _(d1_data, d2_data, tiledb, uri):
    # Open the array in write mode and write the data in COO format
    with tiledb.open(uri, "r") as _A:
        print(_A[d1_data, d2_data])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
