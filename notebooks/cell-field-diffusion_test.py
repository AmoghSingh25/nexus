import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("default")
    return jnp, plt


@app.cell
def _(jnp):
    from simulator.utils.file_manager import _read_file

    cell_conc = _read_file("src/simulator/spatial_vec/logs/cell_concs.pkl")
    field_conc = _read_file("src/simulator/spatial_vec/logs/field_concs.pkl")
    cell_field_assoc = _read_file("src/simulator/spatial_vec/logs/field_cell_assgn.pkl")

    cell_conc = jnp.array(cell_conc)
    field_conc = jnp.array(field_conc)
    return cell_conc, cell_field_assoc, field_conc


@app.cell
def _():
    selected_cell = 0
    return (selected_cell,)


@app.cell
def _(cell_conc, cell_field_assoc, field_conc, plt):
    def plot_cells_in_field(field_id, selected_chem=0, title=""):
        cells_in_field = cell_field_assoc[0][field_id]
        field_conc_i = field_conc[:, field_id, selected_chem]
        plt.plot(field_conc_i, label="Field conc.", color="r")
        for i in cells_in_field:
            plt.plot(cell_conc[:, selected_chem, i], linestyle="--", color="b")
        plt.plot([], [], linestyle="--", color="b", label="Cells Conc.")
        plt.xlabel("Steps")
        plt.ylabel("Chemical concentration")
        plt.xticks(range(len(field_conc)))
        plt.title(title)
        plt.legend()
        # plt.show()

    return (plot_cells_in_field,)


@app.cell
def _(plot_cells_in_field, plt):
    plot_cells_in_field(
        0,
        title="Diffusion of chemical between fields and cells - Single cell field",
    )
    plt.savefig("outputs/images/diffusion_field_cell_1cell.pdf", dpi=1200)
    plt.show()
    return


@app.cell
def _(plot_cells_in_field, plt):
    plot_cells_in_field(
        field_id=7,
        title="Diffusion of chemical between fields and cells - Multi cell field",
    )
    plt.savefig("outputs/images/diffusion_field_cell_multicell.pdf", dpi=1200)
    plt.show()
    return


@app.cell
def _(cell_field_assoc, selected_cell):
    for i in cell_field_assoc:
        for j in i:
            if selected_cell in i[j]:
                print("Current field - ", j)
    return


@app.cell
def _():
    selected_field = 0
    selected_chem = 0
    return selected_chem, selected_field


@app.cell
def _(cell_conc, selected_cell, selected_chem):
    cell_conc[:, selected_chem, selected_cell]
    return


@app.cell
def _(field_conc, plt, selected_chem, selected_field):
    plt.plot(field_conc[:, selected_field, selected_chem])
    return


@app.cell
def _(field_conc, selected_chem, selected_field):
    field_conc[:, selected_field, selected_chem]
    return


@app.cell
def _(cell_conc, field_conc, selected_cell, selected_chem, selected_field):
    (
        cell_conc[:, selected_chem, selected_cell]
        + field_conc[:, selected_field, selected_chem]
    )
    return


@app.cell
def _():
    return


@app.cell
def _(
    cell_conc,
    field_conc,
    plt,
    selected_cell,
    selected_chem,
    selected_field,
):
    plt.plot(field_conc[:, selected_field, selected_chem], label="Field conc")
    plt.plot(cell_conc[:, selected_chem, selected_cell], label="Protein conc")
    plt.legend()
    plt.show()
    return


@app.cell
def _(cell_conc, field_conc, plt, selected_chem, selected_field):
    plt.plot(field_conc[:, selected_field, -1], label="Field conc")
    plt.plot(cell_conc[:, selected_chem, -1], label="Protein conc")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
