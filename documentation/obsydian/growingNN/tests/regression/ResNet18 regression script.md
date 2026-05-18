File: `tests/regression/resnet_regression_test.py`.

---

## What it does

Loads pretrained ResNet-18 (`ResNet18_Weights.DEFAULT`), `eval()`, traces with `fx.symbolic_trace` to `gm`.

Constants (lines 34 to 42): all five action flags on, `BATCH_SIZE = 2`, `INPUT_SHAPE = (3, 64, 64)`, `ITERATIONS = 50`.

Each iteration:

1. `_generate_actions(gm)` merges actions from `AddResLayer` (EYE only), `AddResConvLayer`, `AddSeqLayer`, `AddSeqConvLayer`, `DelLayer`.
2. If the list is empty, logs a warning and breaks (early stop before 50 steps).
3. Picks one action at random (`random.Random(42)`).
4. Appends `type(action).__name__` to `used_action_types`.
5. `execute`, forward `gm(x)`, log output delta norm vs initial output.
6. Writes PDF graphs via `growingnn/utils/fx_graph_drawer.py` into `testResults/regression/`.

After the loop: `plot_norms_and_parameter_count`, then an action summary table in the log (`action` / `count` columns).

CLI: `parse_regression_cli` from `tests/regression/regression_utils.py` (`--save-output`).

---

## Why

Stress test on a real torchvision model with dotted submodule names (`layer1.0.conv1`, etc.). Catches shape bugs from `growingnn/actions/utils/layer_analyser.py` on deep nets.

---

## Early stop (not a crash)

The loop exits before 50 only when:

- `len(actions) == 0` — warning `No actions to execute for iteration N`
- `execute` or forward raises — `logger.exception` then break
- Uncaught error in graph export (outside the execute `try` today)

If the run looks frozen, check DEBUG volume from `module_sequential_pairs` on a large mutated graph. Set `LOG_LEVEL` to `INFO` in `growingnn/core/config.py` to see `idx:` and `action used` lines.

---

## Comparison with the original growingNN paper

Same high-level idea as the paper’s architecture search loop: propose moves, apply one, keep training signal. This script is a manual random walk for debugging, not full MCTS.

---

## Known limitations

1. Forward at 64×64 but shape probe often uses 224×224 (see `documentation/obsydian/growingNN/Actions/utils/Layer Analyser.md`).
2. Action generation cost grows as `seq_conv_*` and `res_conv__*` modules accumulate.
3. PDF export every step is slow.

---

## Related code

Action modules: `growingnn/actions/add_res_layer.py`, `add_res_conv_layer.py`, `add_seq_layer.py`, `add_seq_conv_layer.py`, `delete_layer.py`. Helpers: `tests/regression/regression_utils.py`, `growingnn/utils/fx_graph_drawer.py`.
