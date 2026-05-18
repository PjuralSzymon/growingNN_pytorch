
This note is about `tests/regression/utils_testing.py`.

What it does. It adds the repo root to `sys.path`, loads `ModelFactory.simple_chain_3()`, traces with `torch.fx.symbolic_trace`, then writes two PDFs under `testResults/regression/` using `draw_filtered_fx_graph` and `draw_torch_fx_graph`.

Why. Quick manual check of FX graph drawing without downloading ImageNet weights. Where. Run as `python tests/regression/utils_testing.py` from a cwd where imports resolve; CLI uses `parse_regression_cli` from `tests/regression/regression_utils.py`.

### Related

`tests/model_factory.py`, `tests/regression/resnet_regression_test.py`, [[FX graph drawer]].
