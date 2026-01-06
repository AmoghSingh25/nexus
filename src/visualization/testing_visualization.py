import os
import argparse
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, callback
from simulator.spatial.logger.meshLogger import DataLogger
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


def format_data(inp_data, chem_id, multicell=False, cell_id=None):
    prefix = "Chem- "
    if multicell:
        prefix = "Cell - " + str(cell_id) + "." + prefix
    chem_arr = np.zeros((logger_inst.n_steps, logger_inst.n_chems))
    for t_i, chem_i, idx in zip(
        inp_data["t"], inp_data["chem"], list(range(len(inp_data["conc"])))
    ):
        if int(chem_i) in chem_id:
            chem_arr[t_i][int(chem_i)] = inp_data["conc"][idx]
    chem_arr = chem_arr[:, chem_id]
    ret_df = pd.DataFrame(chem_arr, columns=[prefix + str(i) for i in chem_id])
    ret_df.index.name = "Step"
    return ret_df


@callback(
    Output(component_id="cell_figure", component_property="figure"),
    Input(component_id="cell-id", component_property="value"),
    Input(component_id="chem-id", component_property="value"),
)
def cell_graph(cell_id, chem_id):
    if type(chem_id) is None:
        chem_id = list(range(logger_inst.n_chems))
    elif type(chem_id) is int:
        chem_id = [chem_id]
    if type(cell_id) is not int:
        fig = go.Figure()
        for i in cell_id:
            cell_i_data = logger_inst.retrieve_chem_data(cell_id=i, chem_id=chem_id)
            temp_fig = px.line(
                format_data(
                    inp_data=cell_i_data, multicell=True, cell_id=i, chem_id=chem_id
                ),
                title="Multi Cell Plot",
            )
            for t in temp_fig.data:
                t.line.color = None
                fig.add_trace(t)
        fig.update_layout(
            colorway=px.colors.qualitative.Plotly, title="Multi Cell Plot"
        )
    else:
        cell_data = logger_inst.retrieve_chem_data(cell_id=cell_id, chem_id=chem_id)
        fig = px.line(
            format_data(inp_data=cell_data, chem_id=chem_id),
            title="Cell-" + str(cell_id),
        )

    fig.update_yaxes(title="Chemical Value")
    fig.update_traces(mode="markers+lines", hovertemplate="%{y}<br>")
    fig.update_layout(hovermode="x unified", template="plotly_dark")
    return fig


@callback(
    Output(component_id="reaction-table", component_property="rowData"),
    Input(component_id="cell-id-reaction", component_property="value"),
)
def reaction_order(cell_id):
    reaction_order_struct = []
    reaction_order_data = logger_inst.retrieve_reaction_order(cell_id=cell_id)
    for t_i, cell_i, idx in zip(
        reaction_order_data["t"],
        reaction_order_data["cells"],
        list(range(len(reaction_order_data["reaction_order"]))),
    ):
        # print(type(reaction_order_data['reaction_order'][idx]))
        reaction_order_i = [str(x) for x in reaction_order_data["reaction_order"][idx]]
        reaction_order_struct.append([int(t_i), ",".join(reaction_order_i)])
    ret_df = pd.DataFrame(reaction_order_struct, columns=["Step", "Reaction Order"])
    return ret_df.to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--File", help="Name of logfile to run visualization")
    args = parser.parse_args()

    if args.File:
        ## TODO: Allow user to select log file to run from webpage
        inp_file_name = args.File
    else:
        inp_file_name = "1767656604"

    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../simulator/spatial/logs"
    )
    if not os.path.exists(os.path.join(base_dir, inp_file_name)):
        raise FileNotFoundError("Log file does not exist")
    logger_inst = DataLogger(log_dir=base_dir, file_name=inp_file_name, read_only=True)

    app = Dash()
    app.layout = [
        html.Div(
            children=[
                html.Div(
                    children="Chemical concentration graph",
                    style={"font-size": "30px"},
                    className="ag-theme-material-dark",
                ),
                dcc.Graph(
                    figure={},
                    id="cell_figure",
                    className="ag-theme-material-dark",
                ),
                dbc.Row(
                    [
                        html.Div(
                            children="Choose Cell",
                            className="ag-theme-material-dark",
                        ),
                        dcc.Dropdown(
                            options=[
                                {"label": "Cell - " + str(x), "value": x}
                                for x in range(logger_inst.n_cells)
                            ],
                            value=0,
                            id="cell-id",
                            multi=True,
                            style={"width": "fit-content"},
                            className="ag-theme-material-dark",
                        ),
                        html.Div(
                            children="Choose Chemical",
                            className="ag-theme-material-dark",
                        ),
                        dcc.Dropdown(
                            options=[
                                {"label": "Chem - " + str(x), "value": x}
                                for x in range(logger_inst.n_chems)
                            ],
                            value=0,
                            id="chem-id",
                            multi=True,
                            style={"width": "fit-content"},
                            className="ag-theme-material-dark",
                        ),
                    ],
                    id="select_row",
                ),
            ],
            style={"backgroundColor": "#000000", "padding": "20px", "height": "100%"},
            className="ag-theme-material-dark",
        ),
        html.Div(style={"width": "100%", "height": "10px", "backgroundColor": "white"}),
        html.Div(
            children=[
                html.Div(
                    children="Reaction order view",
                    style={"font-size": "30px"},
                    className="ag-theme-material-dark",
                ),
                dbc.Row(
                    [
                        html.Div(
                            children="Choose Cell",
                            style={"color": "white", "font-family": "Arial"},
                            className="ag-theme-material-dark",
                        ),
                        dcc.Dropdown(
                            options=[
                                {"label": "Cell - " + str(x), "value": x}
                                for x in range(logger_inst.n_cells)
                            ],
                            value=0,
                            id="cell-id-reaction",
                            className="ag-theme-material-auto-dark",
                            style={"width": "30%"},
                        ),
                    ],
                    id="select_row_reaction",
                ),
                dag.AgGrid(
                    rowData=reaction_order(0),
                    columnDefs=[{"field": i} for i in ["Step", "Reaction Order"]],
                    id="reaction-table",
                    className="ag-theme-material-dark",
                ),
            ],
            style={"backgroundColor": "#000000", "padding": "20px", "height": "100%"},
            className="ag-theme-material-dark",
        ),
    ]
    app.run(debug=True)
