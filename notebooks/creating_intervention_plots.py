import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from simulator.spatial_vec.logger.mesh_logger import FieldLogger
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("default")
    return FieldLogger, plt


@app.cell
def _(FieldLogger):
    # 1770312238 - Looped
    # 1770311561 - Pulsed
    # 1770311998 - Scheduled
    logger = FieldLogger(
        log_dir="src/simulator/spatial_vec/logs/",
        # file_name="1770312238",  # Looped
        # file_name="1770311561", # Pulsed
        # file_name="1770311998",  # Scheduled
        file_name="1770313369",  # Normal
        read_only=True,
    )
    return (logger,)


@app.cell
def _(logger):
    conc = logger.retrieve_chem_data(field_id=0, chem_id=2)["conc"]
    return (conc,)


@app.cell
def _(conc, plt):
    # plt.figure(figsize=(14, 6))
    plt.plot(conc, color="r")
    plt.title("Looped intervention on D")

    plt.xlabel("Steps")
    plt.ylabel("Conc.")

    plt.xticks(ticks=list(range(len(conc) + 1)))

    _positions = [1.5, 7.5, 13.5, 19.5]

    for _x in _positions:
        plt.annotate(
            "D=2",
            (_x, 0.535),
            arrowprops=dict(
                arrowstyle="]-[, widthA=1.0, widthB=1.0",
                lw=1.5,
                connectionstyle="bar,angle=0",
            ),
            xytext=(-10, -12),
            textcoords="offset points",
            annotation_clip=False,
        )

    plt.ylim(0.54, 0.61)
    plt.show()
    # plt.savefig("outputs/images/looped_intervention_D.pdf", dpi=1200)
    return


@app.cell
def _(conc, plt):
    plt.plot(conc, color="r")
    plt.title("Pulsed intervention on D")

    plt.xlabel("Steps")
    plt.ylabel("Conc.")

    plt.xticks(ticks=list(range(len(conc) + 1)))

    plt.annotate(
        "D=2",
        (12.5, 0.535),
        arrowprops=dict(
            arrowstyle="]-[, widthA=4.5, widthB=4.5",
            lw=1.5,
            connectionstyle="bar,angle=0",
        ),
        xytext=(-10, -12),
        textcoords="offset points",
        annotation_clip=False,
    )

    plt.ylim(0.54, 0.61)
    plt.show()
    # plt.savefig("outputs/images/pulsed_intervention_D.pdf", dpi=1200)
    return


@app.cell
def _(conc, plt):
    plt.plot(conc, color="r")
    plt.title("Scheduled intervention on D")

    plt.xlabel("Steps")
    plt.ylabel("Conc.")

    plt.xticks(ticks=list(range(len(conc) + 1)))

    plt.annotate(
        "D=2",
        (15, 0.535),
        arrowprops=dict(
            arrowstyle="]-[, widthA=9.0, widthB=9.0",
            lw=1.5,
            connectionstyle="bar,angle=0",
        ),
        xytext=(-10, -12),
        textcoords="offset points",
        annotation_clip=False,
    )

    plt.ylim(0.54, 0.61)
    plt.show()
    # plt.savefig("outputs/images/scheduled_intervention_D.pdf", dpi=1200)
    return


@app.cell
def _(conc, plt):
    plt.plot(conc, color="r")
    plt.title("No intervention on D")

    plt.xlabel("Steps")
    plt.ylabel("Conc.")

    plt.xticks(ticks=list(range(len(conc) + 1)))
    plt.ylim(0.54, 0.61)
    # plt.show()
    plt.savefig("outputs/images/no_intervention_D.pdf", dpi=1200)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
