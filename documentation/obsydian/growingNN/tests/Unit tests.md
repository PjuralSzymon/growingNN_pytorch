Unit tests live under `tests/unit/`. Run from repo root: `python -m pytest tests/unit -q`.

Drivers: `tests/Test runners.md`, `tests/run_all_test.py`.

---

### Action and utils tests

`tests/unit/actions/utils/model_analyser_test.py` — [[Model Analyser]].

`tests/unit/actions/utils/layer_analyser_test.py` — `LayerShapeAnalyser`, `LayerBridgeFinder` (bridge sizes, `uniform_activation_shape`, conv/linear rules).

`tests/unit/actions/add_seq_layer_shape_test.py` — `AddSeqLayer.generate_all_actions` on linear chains and `ModelFactory.simple_conv_chain_2`.

`tests/unit/actions/delete_layer_test.py` — delete shape helpers and `DelLayer.generate_all_actions`.

`tests/unit/actions/add_seq_layer_test.py`, `add_res_layer_test.py` — execute loops on small models.

`tests/unit/actions/utils/model_transformations_test.py` — [[Model Transformer]] `add_new_seq_layer`, `delete_layer`.

---

### Factory

`tests/model_factory.py` — `simple_chain_3`, `simple_conv_chain_2`, residual variants for tests.

---

### Related

Regression notes under `tests/regression/` (no wiki links per vault rules). Vault pages: [[Layer Analyser]], [[Model Analyser]].
