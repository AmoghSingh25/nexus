import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import tiledb
    import numpy as np

    print(tiledb.version())
    print(tiledb.libtiledb.version())
    return np, tiledb


@app.cell
def _(np, tiledb):
    _temp = np.ones((10, 10, 20))
    A = tiledb.from_numpy("src/simulator/spatial/logs/test.tldb", _temp, timestamp=1)
    return (A,)


@app.cell
def _(A):
    A.shape
    return


@app.cell
def _(A):
    A[:5, :5].shape
    return


if __name__ == "__main__":
    app.run()
