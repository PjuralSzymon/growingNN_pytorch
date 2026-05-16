Unit tests live under `tests/unit/`. Pytest collects them when run from repo root or from `tests/`. Example command: `python -m pytest tests/unit/actions/utils/fx_shape_probe_test.py -q`.

For drivers see `tests/Test runners.md` and `tests/run_all_test.py`.

### Notable files

`tests/unit/actions/utils/model_analyser_test.py` exercises Model Analyser helpers (`growingnn/actions/utils/model_analyser.py`).

`tests/unit/actions/utils/fx_shape_probe_test.py` exercises FX Shape Probe (`growingnn/actions/utils/fx_shape_probe.py`) with `ModelFactory.simple_conv_chain_2` from `tests/model_factory.py`.

Other action tests may live under `tests/unit/actions/` as the tree grows.

### Related

`tests/model_factory.py`, regression notes in `tests/regression/Adding residual layers.md` and `tests/regression/Adding Sequential Layers.md`.
