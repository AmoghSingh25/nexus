"use client";
import React, { Suspense, useEffect } from "react";
import dynamic from "next/dynamic";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import { DeckGL } from "@deck.gl/react";
import {
  COORDINATE_SYSTEM,
  OrbitView,
} from "@deck.gl/core";
import { SimpleMeshLayer, ZoomWidget } from "deck.gl";
import {
  CircularProgress,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Slider,
} from "@mui/material";
import { CubeGeometry, SphereGeometry } from "@luma.gl/engine";
import { useSearchParams } from "next/navigation";

var file_name = null;

const options = { method: "GET" };

var data = [];
var field_pos = [];
var field_divs = [];

var sphere_timesteps = [];
var field_cubes = [];

async function get_field_conc(field_id) {
  var field_conc = await fetch(
    `http://localhost:8080/get_field_conc?field_id=${field_id}&file_name=${file_name}`,
    options,
  ).then((response) => response.json());
  var time_steps = [];
  for (let i = 0; i < field_conc[0].length; i++) time_steps.push(i);
  return {
    conc: field_conc,
    time_steps: time_steps,
  };
}

async function get_cell_conc(cell_id) {
  var cell_conc = await fetch(
    `http://localhost:8080/get_cell_conc?cell_id=${cell_id}&file_name=${file_name}`,
    options,
  ).then((response) => response.json());
  var time_steps = [];
  for (let i = 0; i < cell_conc[0].length; i++) time_steps.push(i);
  return {
    conc: cell_conc,
    time_steps: time_steps,
  };
}

async function get_data(indicateReady, currVis) {
  data = await fetch(
    `http://127.0.0.1:8080/positions?file_name=${file_name}`,
    options,
  )
    .then((response) => response.json())
    .then((response) => response.data)
    .catch((err) => console.error(err));
  field_pos = await fetch(
    `http://127.0.0.1:8080/field_positions?file_name=${file_name}`,
    options,
  )
    .then((response) => response.json())
    .catch((err) => console.error(err));
  compute_ds(currVis);
  field_divs = field_pos["field_divs"];
  field_pos = field_pos["pos_data"];
  indicateReady(true);
}

function compute_ds(currVis) {
  sphere_timesteps = [];
  field_cubes = [];

  for (let t = 0; t < data.length; t++) {
    var spheres = [];
    for (let i = 0; i < data[t].length; i++) {
      spheres.push(
        new SimpleMeshLayer({
          id: `sphere-${t}-${i}`,
          data: [0],
          mesh: new SphereGeometry({
            radius: data[t][i].radius,
            nlat: 10,
            nlong: 10,
          }),
          getPosition: data[t][i].position,
          getColor: data[t][i].color,
          coordinateOrigin: [0, 0, 0],
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: currVis == 1,
          onClick: (d) => {
            console.log("PICKED SPHERE");
          },
        }),
      );
    }
    sphere_timesteps.push(spheres);
  }
  const sphere_mesh = new CubeGeometry({});

  const field_cube_layer = new SimpleMeshLayer({
    id: "field-cubes",
    data: field_pos,
    mesh: sphere_mesh,
    getPosition: (d) => d,
    getScale: (d) => [1, 1, 0.5],
    getColor: [255, 0, 0, 50],
    coordinateOrigin: [0, 0, 0],
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    pickable: currVis == 0,
    onClick: (d) => {},
  });
  field_cubes = field_cube_layer;
}

function Visualization() {
  const [step_id, setStepid] = new React.useState(0);
  const [dataReady, setDataReady] = new React.useState(false);
  const [currVis, setCurrVis] = new React.useState(0); // 0 - Field, 1 - Cell
  const [selectedID, setSelectedId] = new React.useState(0); // ID of item selected
  const [fieldConcLoading, setFieldConcLoading] = new React.useState(false);
  const [fieldConcData, setFieldConcData] = new React.useState(null);
  const [traces, setTraces] = new React.useState(null);
  const searchParams = useSearchParams();
  const [cellConcLoading, setCellConcLoading] = new React.useState(false);
  const [cellConcData, setCellConcData] = new React.useState(null);

  file_name = searchParams.get("file_name");
  if (file_name == null) {
    return <></>;
  }

  const layers = React.useMemo(() => {
    if (!dataReady) {
      return [];
    }

    const sphereLayers = sphere_timesteps[step_id]?.map((sphere, i) => {
      return new SimpleMeshLayer({
        id: `${i}`,
        data: [0],
        mesh: new SphereGeometry({
          radius: data[step_id][i].radius,
          nlat: 10,
          nlong: 10,
        }),
        getPosition: data[step_id][i].position,
        getColor: data[step_id][i].color,
        coordinateOrigin: [0, 0, 0],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: currVis == 1,
        onClick: (d) => {
          setSelectedId(d.layer.id);
        },
      });
    });

    const fieldLayer = new SimpleMeshLayer({
      id: "field-cubes",
      data: field_pos,
      mesh: new CubeGeometry({}),
      getPosition: (d) => d,
      getScale: field_divs,
      getColor: [255, 0, 0, 50],
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      pickable: currVis === 0,
      onClick: (d) => {
        setSelectedId(d.index);
      },
    });

    const cubeLayer = new SimpleMeshLayer({
      id: "bg",
      data: {
        position: [0, 0, 0],
      },
      mesh: new CubeGeometry({}),
      getPosition: (d) => d,
      getScale: [5, 5, 5],
      getColor: [255, 255, 255, 50],
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      pickable: false,
    });
    return [...sphereLayers, fieldLayer];
  }, [currVis, step_id, dataReady]);

  useEffect(() => {
    if (currVis !== 0) {
      setFieldConcLoading(false);
    }
    if (currVis !== 1) {
      setCellConcLoading(false);
    }

    if (selectedID == null) return;

    async function fetchFieldConc() {
      setFieldConcLoading(true);
      const resp = await get_field_conc(selectedID);

      setFieldConcData(resp);
      setFieldConcLoading(false);
      var traceArr = [];
      for (let i = 0; i < resp["conc"].length; i++) {
        traceArr.push({
          x: resp["time_steps"],
          y: resp["conc"][i],
          mode: "lines",
          name: `Chem - ${i + 1}`,
        });
      }
      setTraces(traceArr);
    }

    async function fetchCellConc() {
      setCellConcLoading(true);
      const resp = await get_cell_conc(selectedID);

      setFieldConcData(resp);
      setCellConcLoading(false);
      var traceArr = [];
      for (let i = 0; i < resp["conc"].length; i++) {
        traceArr.push({
          x: resp["time_steps"],
          y: resp["conc"][i],
          mode: "lines",
          name: `RNA id - ${i + 1}`,
        });
      }
      setTraces(traceArr);
    }

    if (currVis == 0) fetchFieldConc();
    else {
      fetchCellConc();
    }
  }, [currVis, selectedID]);

  if (!dataReady) {
    get_data(setDataReady, currVis);
    return <CircularProgress />;
  }

  // const ambient_light = new AmbientLight({
  //   color: [255, 255, 255],
  //   intensity: 10.0,
  // });
  // const lighting_effect = new LightingEffect({ ambient_light });

  const view = new OrbitView({
    orbitAxis: "Y",
  });

  return (
    <div
      style={{
        display: "flex",
        height: "150vh",
        flexDirection: "column",
        overflow: "scroll",
      }}
    >
      <Slider
        min={0}
        max={data.length - 1}
        step={1}
        marks
        defaultValue={0}
        valueLabelDisplay="on"
        style={{ width: "90vw", zIndex: 2, margin: "2%" }}
        value={step_id}
        onChange={(e, v) => {
          setStepid(v);
        }}
      />
      <div
        style={{
          position: "relative",
          height: "80vh",
        }}
      >
        <DeckGL
          layers={layers}
          // widgets={[new ZoomWidget(), new ResetViewWidget()]}
          // effects={[lighting_effect]}
          initialViewState={{
            target: [0, 0, 0],
            zoom: 5,
            rotationOrbit: 145,
            rotationX: 65,
            minRotationX: -90,
            maxRotationX: 90,
            minZoom: -10,
            maxZoom: 10,
          }}
          views={view}
          controller={true}
          style={{ height: "80vh", backgroundColor: "white" }}
        />
      </div>
      <div>
        <FormControl>
          <FormLabel
            id="demo-radio-buttons-group-label"
            style={{ color: "lightblue" }}
          >
            Choose the component to visualize
          </FormLabel>
          <RadioGroup
            row
            aria-labelledby="demo-radio-buttons-group-label"
            defaultValue="field"
            name="radio-buttons-group"
          >
            <FormControlLabel
              value="field"
              control={<Radio onClick={() => setCurrVis(0)} />}
              label="Field"
            />
            <FormControlLabel
              value="cell"
              control={<Radio onClick={() => setCurrVis(1)} />}
              label="Cell"
            />
          </RadioGroup>
        </FormControl>
      </div>
      {currVis == 0 && !fieldConcLoading ? (
        <div style={{ height: "50vh", width:"50wh" }}>
          <Plot
            data={traces}
            layout={{
              title: { text: `Field - ${selectedID}` },
              autosize: true,
              xaxis: {
                title: {
                  text: "Steps",
                },
              },
              yaxis: {
                title: {
                  text: "Conc.",
                },
              },
            }}
            style={{ height: "100%", width: "100%" }}
          />
        </div>
      ) : (
        <></>
      )}
      {currVis == 1 && !cellConcLoading ? (
        <div style={{ height: "50vh" }}>
          <Plot
            data={traces}
            layout={{
              title: { text: `Cell - ${selectedID}` },
              autosize: true,
            }}
            style={{ height: "100%", width: "100%" }}
          />
        </div>
      ) : (
        <></>
      )}
    </div>
  );
}

export default function App(){
  return <Suspense>
    <Visualization />
  </Suspense>
}