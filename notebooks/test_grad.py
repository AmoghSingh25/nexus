import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from jax import grad, debug
    import matplotlib.pyplot as plt
    import marimo as mo
    return grad, jnp, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\tanh\left(x-\frac{\left(U+L\right)}{2}\right)\cdot\frac{\left(U-L\right)}{2}+\frac{U+L}{2}$
    """)
    return


@app.cell
def _(grad, jnp):
    a = 4.0
    b = 3.0
    bounds = [[0, 5], [-10, 10]]
    a_hist = [a]
    b_hist = [b]


    def tanh_transform(inp, idx):
        ub = bounds[idx][1]
        lb = bounds[idx][0]
        if inp >= ub or inp <= lb:
            y = jnp.tanh(inp - ((ub + lb) / 2)) * ((ub - lb) / 2) + ((ub + lb) / 2)
        else:
            y = inp
        return y


    def func(a, b):
        trf_a = tanh_transform(a, idx=0)
        trf_b = tanh_transform(b, idx=1)
        y = trf_a * trf_b
        return y


    def calc_loss(a, b, target=-1):
        # target = 2
        y = func(a, b)
        loss = (y - target) ** 2
        return loss


    g = grad(calc_loss, argnums=(0, 1))

    for _ in range(100):
        da, db = g(a, b)
        a -= 0.01 * da
        b -= 0.01 * db
        a_hist.append(a)
        b_hist.append(b)
    return a, a_hist, b, b_hist, bounds, func


@app.cell
def _(a_hist, b_hist, plt):
    plt.plot(a_hist, label="a")
    plt.plot(b_hist, label="b")
    plt.legend()
    plt.show()
    return


@app.cell
def _(a, b, func):
    func(a, b)
    return


@app.cell
def _(a, b, bounds):
    def check_bounds():
        if (
            a >= bounds[0][0]
            and a <= bounds[0][1]
            and b >= bounds[1][0]
            and b <= bounds[1][1]
        ):
            return True
        else:
            return False


    check_bounds()
    return


if __name__ == "__main__":
    app.run()
