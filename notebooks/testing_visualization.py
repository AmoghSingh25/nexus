import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from dash import Dash, html

    return Dash, html


@app.cell
def _(Dash, html):
    app = Dash()
    app.layout = [html.Div(children="Hello world")]
    return (app,)


@app.cell
def _(app):
    app.run(debug=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
