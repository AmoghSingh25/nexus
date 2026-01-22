"use client";
import React, { useEffect } from "react";
import { DeckGL } from "@deck.gl/react";
import {
  AmbientLight,
  COORDINATE_SYSTEM,
  Deck,
  LightingEffect,
  OrbitView,
} from "@deck.gl/core";
import { LineLayer, PointCloudLayer, ScatterplotLayer } from "@deck.gl/layers";
import { SimpleMeshLayer, ZoomWidget } from "deck.gl";
import { ResetViewWidget } from "@deck.gl/widgets";
import { CircularProgress, Slider } from "@mui/material";
import {
  CubeGeometry,
  CylinderGeometry,
  SphereGeometry,
} from "@luma.gl/engine";
import { Sampler } from "@luma.gl/core";

const options = { method: "GET", headers: { "User-Agent": "insomnia/12.1.0" } };

var data = [];
var field_pos = [];

var sphere_timesteps = [];
var field_cubes = [];

async function get_data(indicateReady) {
  data = await fetch(
    "http://127.0.0.1:8080/positions?file_name=1769036978",
    options,
  )
    .then((response) => response.json())
    .then((response) => response.data)
    .catch((err) => console.error(err));
  field_pos = await fetch(
    "http://127.0.0.1:8080/field_positions?file_name=1769036978",
    options,
  )
    .then((response) => response.json())
    .catch((err) => console.error(err));
  compute_ds();
  indicateReady(true);
}

function compute_ds() {
  for (let t = 0; t < data.length; t++) {
    var spheres = [];
    for (let i = 0; i < data[t].length; i++) {
      spheres.push(
        new SimpleMeshLayer({
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
        }),
      );
    }
    sphere_timesteps.push(spheres);
  }
  const sphere_mesh = new CubeGeometry({
    size: 0.1,
    length: 0.1
  });

  const field_cube_layer = new SimpleMeshLayer({
    id: "field-cubes",
    data: field_pos,
    mesh: sphere_mesh,
    getPosition: (d) => d,
    getScale: d=> [1, 1, 0.5],
    getColor: [255, 0, 0, 50],
    coordinateOrigin: [0, 0, 0],
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
  });
  field_cubes.push(field_cube_layer)
}

export default function App() {
  const [step_id, setStepid] = new React.useState(0);
  const [ref, refreshVis] = new React.useState(0);
  const [dataReady, setDataReady] = new React.useState(false);

  if (!dataReady) {
    get_data(setDataReady);
    return <CircularProgress />;
  }

  const ambient_light = new AmbientLight({
    color: [255, 255, 255],
    intensity: 10.0,
  });
  const lighting_effect = new LightingEffect({ ambient_light });

  const view = new OrbitView({
    orbitAxis: "Y",
  });

  return (
    <div>
      <DeckGL
        // layers={[sphere_timesteps[step_id], field_cubes]}
        layers={[sphere_timesteps[step_id], field_cubes]}
        widgets={[new ZoomWidget(), new ResetViewWidget()]}
        effects={[lighting_effect]}
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
        style={{ height: "90%", display: "block" }}
      />
      <Slider
        min={0}
        max={data.length - 1}
        step={1}
        marks
        defaultValue={0}
        valueLabelDisplay="auto"
        style={{ position: "absolute", bottom: 10, margin: 40, width: "90%" }}
        value={step_id}
        onChange={(e, v) => {
          setStepid(v);
        }}
      />
    </div>
  );
}
