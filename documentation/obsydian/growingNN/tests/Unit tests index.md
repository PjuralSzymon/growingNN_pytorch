Unit tests live under `tests/unit/`. Pytest collects them when run from repo root or from `tests/`. Example command: `python -m pytest tests/unit/actions/utils/fx_shape_probe_test.py -q`.

For drivers see [[Test runners]].

### Notable files

`tests/unit/actions/utils/model_analyser_test.py` exercises [[Model Analyser]] helpers.

`tests/unit/actions/utils/fx_shape_probe_test.py` exercises [[FX Shape Probe]] with [[Model factory]] `ModelFactory.simple_conv_chain_2`.

Other action tests may live under `tests/unit/actions/` as the tree grows.
