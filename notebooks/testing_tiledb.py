import marimo

__generated_with = "0.18.0"
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
    _temp = np.ones((10, 10))
    _temp2 = np.zeros((10, 10))
    uri = "src/simulator/spatial/logs/test.tldb"
    with tiledb.from_numpy(uri, _temp, timestamp=1, mode="append") as A:
        print(A)
        pass
    return (uri,)


@app.cell
def _(tiledb, uri):
    with tiledb.open(uri, mode="r", timestamp=1) as _A:
        arr = _A[:]
        print(_A.timestamp_range)
        print(arr)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
