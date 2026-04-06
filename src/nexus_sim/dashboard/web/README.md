# Visualization dashboard

The visualization can be run by the running the two commands in seperate terminal windows.

```bash
## Runs the API for fetching the data from TileDB
uv run src/nexus_sim/dashboad/retrieve_logger_api.py

## Runs the website to visualize the data
cd src/nexus_sim/dashboard/web/
npm install # If the packages are not installed
npm run dev
```

After running these two commands, open the link `http://localhost:3000` and selecting a log file from the dropdown. This log file should be present inside `src/simulator/grn/logs/` and `src/simulator/spatial_vec/logs/`.

The colors indicate the states of the cell, yellow indicating live cells, blue indicating cells undergoing programmed cell death and red are the cells undergoing sudden cell death.

A time slider is given on the top to select the time step for visualization. This can be used to view the change in the cell positions with time.

The bottom section also provides visualization of chemical concentrations in a Field or a Cell, depending on the selection. To visualize a different Field/Cell, it can be selected in the 3D view to update the plot.