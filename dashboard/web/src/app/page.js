"use client";
import React, { useEffect, useState } from "react";
import {
  FormControl,
  MenuItem,
  InputLabel,
  Select,
  ThemeProvider,
} from "@mui/material";
import { createTheme } from "@mui/material/styles";
import { useRouter } from "next/navigation";

const options = { method: "GET" };

const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});

async function get_logs(setFileNames) {
  var log_files = await fetch(`http://localhost:8080/get_logs`, options).then(
    (response) => response.json(),
  );
  setFileNames(log_files["files"]);
  return log_files["files"];
}

export default function App() {
  const [fileNames, setFileNames] = useState(null);
  const router = useRouter()

  useEffect(() => {
    get_logs(setFileNames);
  }, [fileNames]);
  if (fileNames == null) {
    return <></>;
  } else {
    return (
      <div style={{ padding: "20px" }}>
        <ThemeProvider theme={darkTheme}>
          <FormControl fullWidth color="white">
            <InputLabel>File Name</InputLabel>
            <Select label="Select file name" onChange={(x)=>
                router.push("/vis?file_name="+x.target.value)
            }>
              {fileNames.map((x, idx) => {
                return <MenuItem value={x}>{x}</MenuItem>;
              })}
            </Select>
          </FormControl>
        </ThemeProvider>
      </div>
    );
  }
}
