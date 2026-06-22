# GrowingNN Board – implementation plan

This document records the integration design. UI mockups live in `growingnn-board/*.png`.

## Design goal

GrowingNN Board is read-only. Training writes files; the board polls and displays them. On the **growingnn** side there is exactly **one class**: `ExperimentBoard` in `growingnn/board/experiment_board.py`.

## growingnn changes (minimal)

| File | Change |
|------|--------|
| `RunningConfig` | Optional `experiment_board` field |
| `gradient_descent` | Optional `on_epoch_end` callback (one hook) |
| `train_generations` | Calls board lifecycle methods if set |
| `montecarlo_alg` / `greedy_alg` | Report simulation via board; skip duplicate report in trainer |

No board logic outside `growingnn/board/`.

## Experiment directory layout

```text
experiment_dir/
├── main.json                 # overview + lastSimulation + trainingParameters
├── graphs/
│   ├── start_simplified.pdf
│   ├── gen_0_full.pdf
│   └── gen_0_simulation_simplified.pdf
├── generations/
│   └── generation_0.json
├── simulations/
│   └── simulation_gen_0.json
└── metrics/
    └── training.json         # all epoch rows for charts
```

JSON writes use temp-file + replace so the board can poll safely during training.

## Board usage in training

```python
from growingnn.board import ExperimentBoard

board = ExperimentBoard(
    "experiments/output/my_run",
    experiment_name="CIFAR-10 minimal",
    dataset="CIFAR-10",
    device="cpu",
)
cfg = RunningConfig(..., experiment_board=board)
train_generations(model, train_loader, val_loader, cfg)
```

## growingnn_board package

Read-only FastAPI server (`growingnn_board/`):

- Polls loaded experiment every 5 s
- Validates JSON with Pydantic
- Serves PDFs for PDF.js frontend

Run:

```bash
pip install -r growingnn_board/requirements.txt
python -m growingnn_board.app
```

## Mockup → data mapping

### Start page (`mainPage.png`)

- `main.json` → `lastUpdate`, `experimentName`, path listing via `/api/experiments/recent`
- Status: active / recent / inactive from `lastUpdate` age

### Training board (`mainTrainingBoard.png`)

- Sidebar: `main.json` `trainingParameters`, `dataset`, `device`, elapsed time
- Loss charts: `metrics/training.json` epochs (generation + global)
- Timeline: `generationTimeline` + epoch rows
- Last simulation summary: `main.json` `lastSimulation`
- Graph viewer: `graphs/gen_N_simplified.pdf`

### Simulation board (`mainSimulationBoard.png`)

- Settings: `trainingParameters` simulation fields
- Candidate cards: `simulations/simulation_gen_N.json` `candidates`
- Chosen action: `actionChosen`, `scoreChosen`
- Graph: `graphs/gen_N_simulation_simplified.pdf`

## Future (not in this PR)

- React frontend + PDF.js viewer
- Per-candidate mini-graph PDFs
- Random search board reporting
