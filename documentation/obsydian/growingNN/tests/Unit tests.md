Unit tests live under `tests/unit/`. Run from repo root: `python -m pytest tests/unit -q`.

Drivers: `tests/run_all_test.py` (see `documentation/obsydian/growingNN/tests/Test runners.md` for a short note on that script).

---

### Action and utils tests

`tests/unit/actions/utils/model_analyser_test.py` — covers `growingnn/actions/utils/model_analyser.py` (Model Analyser).

`tests/unit/actions/utils/layer_analyser_test.py` — `LayerShapeAnalyser`, `LayerBridgeFinder` in `growingnn/actions/utils/layer_analyser.py`.

`tests/unit/actions/add_seq_layer_shape_test.py` — `AddSeqLayer.generate_all_actions` on linear chains and `ModelFactory.simple_conv_chain_2`.

`tests/unit/actions/delete_layer_test.py` — delete shape helpers and `DelLayer.generate_all_actions`.

`tests/unit/actions/add_seq_layer_test.py`, `add_res_layer_test.py` — execute loops on small models.

`tests/unit/actions/utils/model_transformations_test.py` — `add_new_seq_layer`, `delete_layer` in `growingnn/actions/utils/model_transformations.py`.

---

### Factory

`tests/model_factory.py` — `simple_chain_3`, `simple_conv_chain_2`, residual variants.

---

### Related code (not vault links)

Regression scripts under `tests/regression/`. Product docs under `documentation/obsydian/growingNN/Actions/` and `Actions/utils/`.
