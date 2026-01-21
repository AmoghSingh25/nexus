"use client";
import React, { useEffect } from "react";
import { DeckGL } from "@deck.gl/react";
import { AmbientLight, COORDINATE_SYSTEM, Deck, LightingEffect, OrbitView } from "@deck.gl/core";
import { LineLayer, PointCloudLayer, ScatterplotLayer } from "@deck.gl/layers";
import { SimpleMeshLayer, ZoomWidget } from "deck.gl";
import { ResetViewWidget } from "@deck.gl/widgets";
import { CircularProgress, Slider } from "@mui/material";
import { SphereGeometry } from "@luma.gl/engine";

const options = { method: "GET", headers: { "User-Agent": "insomnia/12.1.0" } };

var data = []
var sphere_timesteps = []

async function get_data(indicateReady){
  data = await fetch(
  "http://127.0.0.1:8080/positions?file_name=1769006175",
  options,
)
  .then((response) => response.json())
  .then((response) => response.data)
  // .then(()=> indicateReady(true))
  .catch((err) => console.error(err));
  compute_spheres_ds();
  indicateReady(true);
}

function compute_spheres_ds() {
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
}


export default function App() {
  const [step_id, setStepid] = new React.useState(0);
  const [ref, refreshVis] = new React.useState(0);
  const [dataReady, setDataReady] = new React.useState(false);

  if(!dataReady)
  {
    get_data(setDataReady);
    return <CircularProgress/>
  }

  const ambient_light = new AmbientLight({
    color: [255,255,255],
    intensity: 10.0
  })
  const lighting_effect = new LightingEffect({ambient_light})

  const view = new OrbitView({
    orbitAxis:"Y"
  })

  return (
    <div>
    <DeckGL
      layers={[sphere_timesteps[step_id]]}
      widgets={[new ZoomWidget(), new ResetViewWidget()]}
      effects={[lighting_effect]}
      initialViewState={{
        target:[0,0,0],
        zoom:3,
        rotationOrbit: 145,
        rotationX: 65,
        minRotationX: -90,
        maxRotationX: 90,
        minZoom: -10,
        maxZoom: 10
      }}
      views={view}
      controller={true}
      style={{"height":"90%", "display":"block"}}
    />
      <Slider min={0} max={data.length-1} step={1} marks defaultValue={0} valueLabelDisplay="auto" style={{"position":"absolute", "bottom":10, "margin":40, width:"90%"}}
      value={step_id}
      onChange={(e, v)=>{setStepid(v)}}
      />
    </div>
  );
}