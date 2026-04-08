from flask import Flask, request, make_response
from flask_cors import CORS
import os
import argparse
import pandas as pd
import numpy as np

from nexus.simulator.spatial.logger.mesh_logger import FieldLogger as DataLoggerVec
from nexus.simulator.spatial.logger.spatial_logger import SpatialLogger
from nexus.simulator.grn.logger.grnLogger import GRNLogger

app = Flask(__name__)
CORS(app, origins="*")
# data_base_dir = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "../simulator/spatial/logs"
# )
spatial_base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../src/nexus/simulator/spatial/logs"
)
grn_base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../src/nexus/simulator/grn/logs"
)
logger_inst = None
spatial_logger_inst = None
grn_logger_inst = None


def structure_field_chem_data(resp):
    ret = np.zeros((logger_inst.n_chems, logger_inst.n_steps))
    for i in range(len(resp["t"])):
        ret[int(resp["chem"][i])][int(resp["t"][i])] = resp["conc"][i]
    return ret


def structure_cell_chem_data(resp):
    ret = np.zeros((grn_logger_inst.n_genes, grn_logger_inst.n_steps))
    for i in range(len(resp["t"])):
        ret[int(resp["gene_id"][i])][int(resp["t"][i])] = resp["gene_conc"][i]
    return ret


def structure_positions_data(pos):
    no_steps = len(set(pos["t"]))
    pos_arr = []
    opacity = 255
    for i in range(len(pos["cells"])):
        step_i = int(pos["t"][i])
        state_i = int(pos["state"][i])
        radius_i = float(pos["radius"][i])
        pos_i = np.array(np.array(pos["pos"][i]).tolist())
        pos_i = pos_i
        pos_i = pos_i.tolist()
        color_i = [255, 255, 0, opacity]
        if state_i == -1:
            color_i = [255, 0, 0, opacity]
        elif state_i == -2:
            color_i = [0, 0, 255, opacity]

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


@app.route("/get_logs")
def get_logs():
    log_files = filter(
        os.path.isdir,
        [os.path.join(spatial_base_dir, x) for x in os.listdir(spatial_base_dir)],
    )
    log_files = list(log_files)
    log_file_names = [x[len(str(spatial_base_dir)) + 1 :] for x in log_files]
    resp_files = make_response({"files": log_file_names})
    return resp_files


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
    field_conc["conc"] = np.nan_to_num(field_conc["conc"])
    resp = structure_field_chem_data(field_conc)
    resp = make_response(resp.tolist())
    return resp


@app.route("/get_cell_conc", methods=["GET"])
def get_cell_conc():
    global grn_logger_inst

    file_name = request.args.get("file_name")
    cell_id = int(request.args.get("cell_id"))

    if file_name is None:
        return "<p>File name parameter invalid</p>"
    if grn_logger_inst is None:
        grn_logger_inst = GRNLogger(
            log_dir=grn_base_dir, file_name=file_name, read_only=True
        )
    cell_conc = grn_logger_inst.retrieve_conc(cells=cell_id)
    cell_conc["gene_conc"] = np.nan_to_num(cell_conc["gene_conc"])
    resp = structure_cell_chem_data(cell_conc)
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

    # if not os.path.exists(os.path.join(data_base_dir, inp_file_name)):
    #     raise FileNotFoundError("Log file does not exist")

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
