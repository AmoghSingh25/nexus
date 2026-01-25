from flask import Flask, request, make_response
from flask_cors import CORS
import os
import argparse
import pandas as pd
import numpy as np
from dash import Input, Output, callback

from simulator.spatial_vec.logger.mesh_logger import FieldLogger as DataLoggerVec
from simulator.spatial_vec.logger.spatial_logger import SpatialLogger
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
CORS(app, origins="*")
data_base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../simulator/spatial/logs"
)
spatial_base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../simulator/spatial_vec/logs"
)
logger_inst = None
spatial_logger_inst = None


def structure_field_chem_data(resp):
    ret = np.zeros((logger_inst.n_chems, logger_inst.n_steps))
    for i in range(len(resp["t"])):
        ret[int(resp["chem"][i])][int(resp["t"][i])] = resp["conc"][i]
    return ret


def structure_positions_data(pos):
    no_steps = len(set(pos["t"]))
    pos_arr = []
    for i in range(len(pos["cells"])):
        step_i = int(pos["t"][i])
        state_i = int(pos["state"][i])
        radius_i = float(pos["radius"][i])
        pos_i = np.array(np.array(pos["pos"][i]).tolist())
        pos_i = pos_i
        pos_i = pos_i.tolist()
        color_i = [255, 255, 0]
        if state_i == -1:
            color_i = [255, 0, 0]
        elif state_i == -2:
            color_i = [0, 0, 255]

        if len(pos_arr) < no_steps:
            pos_arr.append([])

        pos_arr[step_i].append(
            {
                "position": pos_i,
                "color": color_i,
                "radius": radius_i if radius_i > 0 else 0.2,
            }
        )
    return {"data": pos_arr}


def structure_field_pos(resp, n_fields):
    struct_pos = []
    for i in range(n_fields):
        struct_pos.append(resp["position"][i].tolist())
    ret = (np.array(logger_inst.axis_divs) / logger_inst.field_res).tolist()
    return {"pos_data": struct_pos, "field_divs": ret}


@app.route("/")
def hello_world():
    return "<p>Hello World</p>"


@app.route("/positions", methods=["GET"])
def get_positions():
    global spatial_logger_inst
    file_name = request.args.get("file_name")
    if file_name is None:
        return "<p>File name parameter invalid</p>"

    spatial_logger_inst = SpatialLogger(
        log_dir=spatial_base_dir, file_name=file_name, read_only=True
    )
    pos = spatial_logger_inst.retrieve_pos_data()
    resp_pos = make_response(structure_positions_data(pos))
    return resp_pos


@app.route("/field_positions", methods=["GET"])
def get_field_positions():
    global logger_inst
    file_name = request.args.get("file_name")
    if file_name is None:
        return "<p>File name parameter invalid</p>"

    logger_inst = DataLoggerVec(
        log_dir=spatial_base_dir, file_name=file_name, read_only=True
    )
    field_pos = logger_inst.retrieve_field_pos_data()
    resp_pos = make_response(structure_field_pos(field_pos, logger_inst.n_fields))
    return resp_pos


@app.route("/get_field_conc", methods=["GET"])
def get_field_conc():
    global logger_inst

    file_name = request.args.get("file_name")
    field_id = int(request.args.get("field_id"))

    if file_name is None:
        return "<p>File name parameter invalid</p>"
    if logger_inst is None:
        logger_inst = DataLoggerVec(
            log_dir=spatial_base_dir, file_name=file_name, read_only=True
        )
    field_conc = logger_inst.retrieve_chem_data(field_id=field_id)
    resp = structure_field_chem_data(field_conc)
    resp = make_response(resp.tolist())
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--File", help="Name of logfile to run visualization")
    args = parser.parse_args()

    if args.File:
        inp_file_name = args.File
    else:
        inp_file_name = "1767656604"

    if not os.path.exists(os.path.join(data_base_dir, inp_file_name)):
        raise FileNotFoundError("Log file does not exist")
    # logger_inst = DataLogger(
    #     log_dir=data_base_dir, file_name=inp_file_name, read_only=True
    # )

    app.run(port=8080, debug=True)


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
