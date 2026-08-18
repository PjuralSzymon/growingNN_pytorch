# GrowingNN Board

Read-only dashboard backend for GrowingNN training runs. Training writes JSON and PDF files; the board reads them and exposes a REST API. UI mockups live in `growingnn-board/`.

## Prerequisites

- Python 3.11+
- Repo root on `PYTHONPATH` (run commands from the repository root)
- A virtualenv with project dependencies (`torch`, etc.) plus board extras

## Install

From the repository root:

```bash
pip install -r requirments.txt
pip install -r tools/growingnn_board/requirements.txt
```

## Start the server

From the repository root, double-click or run:

```bat
start_board.bat
```

Or:

```powershell
python tools/growingnn_board/run_server.py
```

Default URL: http://127.0.0.1:8765 — open this in a browser for the UI (not just `/docs`).

Interactive API docs: http://127.0.0.1:8765/docs

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROWINGNN_BOARD_HOST` | `127.0.0.1` | Bind address |
| `GROWINGNN_BOARD_PORT` | `8765` | Port |
| `GROWINGNN_BOARD_POLL_SEC` | `5` | Refresh interval for a loaded experiment |
| `GROWINGNN_EXPERIMENTS_ROOT` | `experiments/output` | Root scanned by “recent experiments” |

## Produce experiment data (training side)

Attach `ExperimentBoard` when building `RunningConfig`:

```python
from growingnn.board import ExperimentBoard

board = ExperimentBoard(
    "experiments/output/my_run/board",
    experiment_name="My experiment",
    dataset="CIFAR-10",
)
cfg = RunningConfig(..., experiment_board=board)
train_generations(model, train_loader, val_loader, cfg)
```

Example script with board enabled by default:

```bash
python experiments/train_cifar10.py --board true
```

Artifacts appear under `experiments/output/train_cifar10/board/`:

```text
main.json
metrics/training.json
generations/generation_N.json
simulations/simulation_gen_N.json
graphs/*.pdf
```

## Load an experiment in the board

1. Start the server (see above).
2. Point the board at any directory. The board uses `main.json` in that directory. If it is not
   present, the board searches subdirectories and loads the most recently modified valid
   `main.json`. Absolute paths may be outside `GROWINGNN_EXPERIMENTS_ROOT`:

```bash
curl -X POST "http://127.0.0.1:8765/api/experiment/load?path=D:/repos/growingNN_pytorch/experiments/output/train_cifar10/board"
```

3. Poll read endpoints (the server reloads files every 5 seconds):

```bash
curl http://127.0.0.1:8765/api/experiment/main
curl http://127.0.0.1:8765/api/experiment/training
curl http://127.0.0.1:8765/api/simulation/0
```

List recent experiment folders under `GROWINGNN_EXPERIMENTS_ROOT`:

```bash
curl http://127.0.0.1:8765/api/experiments/recent
```

Open a graph PDF (path relative to the loaded experiment or absolute):

```bash
curl "http://127.0.0.1:8765/api/files/pdf?path=graphs/gen_0_simplified.pdf" --output graph.pdf
```

## API summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/experiments/recent` | Recent experiment directories |
| POST | `/api/experiment/load?path=...` | Load one experiment |
| GET | `/api/experiment/main` | Overview (`main.json`) |
| GET | `/api/experiment/training` | Epoch metrics |
| GET | `/api/generations` | Generation index |
| GET | `/api/generation/{n}` | One generation snapshot |
| GET | `/api/simulation/{n}` | Simulation details + candidates |
| GET | `/api/files/pdf?path=...` | Serve a PDF |

The board never starts, stops, or edits training. It only reads files from disk.
