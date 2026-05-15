
This note is about `tests/regression/utils_testing.py`.

What it does. It adds the repo root to `sys.path`, defines nested blocks `InnerBlock`, `MiddleBlock`, `OuterBlock`, and `ModelDeeplyNested` (same nesting idea as `ModelFactory.deeply_nested_submodules` in `tests/model_factory.py`), traces with `torch.fx.symbolic_trace`, then writes two PDFs under `testResults/regression/` using `draw_filtered_fx_graph` and `draw_torch_fx_graph`.

Why. Quick manual check of dotted FX names on a small nested net without downloading ImageNet weights. Where. Run as `python tests/regression/utils_testing.py` from a cwd where imports resolve; CLI uses `parse_regression_cli` from `tests/regression/regression_utils.py`.

### Related

[[Model factory]], [[Dotted Module Names in torch.fx]], [[ResNet18 regression script]].
