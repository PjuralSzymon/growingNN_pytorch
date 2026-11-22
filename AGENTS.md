# GrowingNN project guide

## Project overview

GrowingNN is a PyTorch project that trains a model while evolving its architecture. The core implementation is in `growingnn/`; changes there should follow the algorithm below.

```text
Algorithm 1 Training algorithm.
1: Input: Dataset
2: Output: Model
3: Initialization:
4:     SimSet <- Create simulation set
5:     Model <- Create a model with basic structure
6:     for each generation do
7:         GradientDescent(Model, Dataset, epochs)
8:         if canSimulate() then
9:             Action <- MCTS.simulation(Model)
10:            Model <- Action.execute(Model)
11:        end if
12:    end for
```

## Main folders

- `growingnn/` — the algorithm: model/configuration, generation training, simulation/MCTS, graph actions, and `torch.fx` graph tools.
- `experiments/` — experiment definitions and scripts that run configured training studies.
- `documentation/` — project notes and the website, including published experiment reports and their data/charts.
- `tests/` — unit, integration, and regression checks for the implementation.
- `growingnn_board/` — experiment board UI and support code for inspecting runs.
- `scripts/` — helper scripts for repository and experiment workflows.
- `article/` — article or paper material related to the project.
- `testResults/` — test result artifacts.

## Experiments and reports

Add and run experiment scripts in `experiments/`. `experiments/createsummary.py` processes run output into summaries. Website content and chart-generation scripts live in `documentation/website/scripts/` (for example, `generate_content.py` and `generate_experiment_005_charts.py`). Published report pages are in `documentation/website/content/experiments/`; reusable snapshots are in `documentation/website/data/experiments/`, with generated charts in the website assets.

A typical report should include:

- A short purpose and the research question(s).
- Tested and fixed parameters, plus links or paths to the run script and result data.
- Measured charts with labeled axes and a caption directly below each chart.
- Grouped result tables and a concise explanation of important training-history patterns.
- Limitations, including incomplete runs or seed variation where relevant.
- Evidence-based conclusions and useful next experiments.

Keep claims traceable to recorded results; distinguish training from validation measurements and explain how the reported data was selected or grouped.
