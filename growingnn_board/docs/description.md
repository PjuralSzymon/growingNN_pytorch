# GrowingNN Board – Implementation Documentation

## 1. Purpose

GrowingNN Board is a read-only dashboard application used to visualize training and simulation data produced by the GrowingNN training algorithm. The training algorithm runs separately and periodically writes data files to a selected output directory. The board application does not control training, does not modify model parameters, and does not execute simulations itself. Its only responsibility is to read JSON and PDF files from the selected directory, refresh the displayed data every few seconds, and present the current state of training, simulation results, architecture graphs, and generation history in a clear UI.

Most important are files: 
growingnn-board\mainPage.png
growingnn-board\mainSimulationBoard.png
growingnn-board\mainTrainingBoard.png
are describing how the applicaiton should work analyse those and based on those plan the implmentation and what data should be saved 

## 2. General Workflow

The typical workflow is:

1. Start the GrowingNN training algorithm.
2. The training algorithm writes JSON files and generated graph PDFs into an experiment directory.
3. Start the GrowingNN Board application.
4. On the start page, select the experiment directory.
5. The board loads `main.json`.
6. The board refreshes the selected directory every 5 seconds.
7. If JSON or PDF files change, the UI updates automatically.

The application should treat the selected directory as the source of truth.

## 3. Recommended Technology Stack

The implementation should be mostly Python.

Recommended backend:

```text
Python 3.11+
FastAPI
Pydantic
watchdog or polling-based file watcher
```

Recommended frontend:

```text
React / Vue / simple HTML templates
PDF.js for PDF rendering
Chart.js / Plotly / Recharts for graphs
```

Recommended simple architecture:

```text
GrowingNN Board
│
├── Python backend
│   ├── reads JSON files
│   ├── validates schemas
│   ├── serves API endpoints
│   ├── serves PDF files
│   └── polls directory every 5 seconds
│
└── frontend
    ├── start page
    ├── main overview board
    └── simulation details board
```

## 4. Experiment Directory Structure

A selected experiment directory should look like this:

```text
experiment_001/
│
├── main.json
│
├── graphs/
│   ├── overview_graph.pdf
│   ├── gen_1_graph.pdf
│   ├── gen_2_graph.pdf
│   └── gen_5_simulation_graph.pdf
│
├── generations/
│   ├── generation_1.json
│   ├── generation_2.json
│   └── generation_5.json
│
├── simulations/
│   ├── simulation_gen_5.json
│   └── simulation_gen_6.json
│
└── metrics/
    ├── loss_generation.json
    └── global_loss.json
```

The exact filenames can be changed, but the structure should stay predictable.
in the growingNN main module implmeent one class that will be responsible for saving allof this data live for example it will be passed
to the trianing funciton as an object and every epoch it will be runned as growingnnboardcontroller.updateTrainingParams(epoch,accuracy, ...) (Names are random please create better ones)
PUT A BIG FOCUS ON DOING AS LITTLE CHANGES IN GROWINGNN MODULE AS POSSIBLE MAKE IT SIMPLE AND EASY TO SEE 

## 5. Refresh Logic

The app should poll the selected directory every 5 seconds.

Pseudo-logic:

```python
while app_is_running:
    scan_selected_directory()
    read_main_json()
    check_last_update()
    read_generation_jsons()
    read_simulation_jsons()
    update_api_cache()
    sleep(5)
```

The board should not crash if a file is temporarily incomplete because the training process may be writing it. JSON loading should be protected with error handling.

Recommended behavior:

```text
valid JSON      -> update cache
invalid JSON    -> keep previous valid cache
missing file    -> show warning
changed PDF     -> reload PDF viewer
```

## 6. Main JSON Schema

`main.json` is the central file used by the board. Have in mind that not all of those params should be updated at once in the json
Analyse the image: mainPage.png and based on this decide what data should be saved with this json: 

Example:

```json
{
  "experimentId": "exp_2024_05_18_001",
  "experimentName": "GrowingNN GPT experiment",
  "lastUpdate": "2024-05-18T10:42:11Z",
  "experimentStartedOn": "2024-05-18T09:42:11Z",
  "trainingTimeElapsedSec": 12932,
  "status": "running",

  "trainingParameters": {
    "totalGenerations": 8,
    "currentGeneration": 5,
    "currentEpoch": 12,
    "epochsPerGeneration": 20,
    "totalEpochs": 160,
 ...
  },

 ...
}
```

## 7. Directory Status Logic

The start page should show recent experiment directories and their status based on the `lastUpdate` field from `main.json`.

Status rules:

```text
lastUpdate < 1 hour old   -> green / Active
lastUpdate < 6 hours old  -> yellow / Recent
lastUpdate >= 6 hours old -> gray / Inactive
```

The frontend should display:

```text
Directory path
Last update
Status dot
Status label
```

## 8. Training JSON Schema

Keep a seperate json file for all data realted accuracy, loss and others and how it was changing during training 
Analyse the image: mainTrainingBoard.png and based on this decide what data should be saved with this json


## 9. Simulation JSON Schema

Simulation data should describe what action was tested, what action was chosen, and the simulation tree. Focus mostly on what monte carlo simujaliton is saving but also update teh remianing ones 
Analyse the image: mainSimulationBoard.png and based on this decide what data should be saved with this json

## 10. API Endpoints

The Python backend should expose simple read-only endpoints.

Example endpoints:

```text
GET /api/experiments/recent
GET /api/experiment/load?path=...
GET /api/experiment/main
GET /api/generations
GET /api/generation/{generationNumber}
GET /api/simulation/{generationNumber}
GET /api/files/pdf?path=...
```

The frontend should never read files directly. It should request data through the backend.

## 11. PDF Handling

PDF files should be displayed in the browser using PDF.js.

Requirements:

```text
Always show vertical scrollbar
Always show horizontal scrollbar
Show file path above or near the PDF viewer
Support zoom in/out
Support page number
Support download/open button
```

The PDF viewer title should be contextual:

```text
Simulation Graph for Generation: 5 (PDF)
File path: /logs/exp_2024_05_18/graphs/gen_5/simulation_graph.pdf
```

## 12. Error Handling

The app should handle missing or invalid files gracefully.

Examples:

```text
main.json missing
→ show "This folder does not contain a valid GrowingNN experiment."

PDF missing
→ show "Graph PDF is not available yet."

JSON invalid
→ keep showing last valid data and display a small warning.

lastUpdate too old
→ show gray inactive status.
```

## 13. Important Implementation Rules

The board must be read-only.

It should not:

```text
start training
stop training
change learning rate
change optimizer
change simulation settings
edit JSON files
delete files
modify PDFs
```

It should only:

```text
read files
validate files
cache parsed data
display charts
display PDFs
refresh every 5 seconds
show warnings when files are stale or invalid
```

## 14. Minimal Python Backend Structure

Recommended files:

```text
growingnn_board/
│
├── app.py
├── config.py
├── schemas.py
├── file_reader.py
├── cache.py
├── pdf_server.py
└── api.py
```

Responsibilities:

```text
schemas.py      -> Pydantic schemas for JSON validation
file_reader.py  -> safe JSON and PDF path reading
cache.py        -> stores latest valid parsed data
api.py          -> FastAPI endpoints
app.py          -> starts the server
```

## 15. Example Polling Service

```python
import time
from pathlib import Path

class ExperimentWatcher:
    def __init__(self, experiment_path: Path, interval_sec: int = 5):
        self.experiment_path = experiment_path
        self.interval_sec = interval_sec
        self.cache = {}

    def read_loop(self):
        while True:
            try:
                self.refresh()
            except Exception as exc:
                print(f"Refresh failed: {exc}")
            time.sleep(self.interval_sec)

    def refresh(self):
        main_path = self.experiment_path / "main.json"
        if not main_path.exists():
            return

        # Read and validate main.json here.
        # Then read generation and simulation files referenced by main.json.
```

## 16. Summary

Most important are files: 
growingnn-board\mainPage.png
growingnn-board\mainSimulationBoard.png
growingnn-board\mainTrainingBoard.png
are describing how the applicaiton should work analyse those and based on those plan the implmentation and what data should be saved 