Regression script: `tests/regression/adding_seq_layers.py`. Action code: `growingnn/actions/add_seq_layer.py` (`AddSeqLayer`). Vault doc: Sequentail Linear Actions (`documentation/obsydian/growingNN/Actions/Sequentail Linear Actions.md`).

---

## What we test

Small and medium models from `tests/model_factory.py`. `AddSeqLayer.generate_all_actions` then random `execute` steps.

Early runs used only rank-2 bridges (`find_bridge_linear_sizes`). The port now also proposes conv→linear sequential moves via `find_seq_linear_after_conv_sizes` and `LinearFactory.create_linear` (no extra pool inside the new module).

---

## Results (historical)

Quasi-identity square linears on a tiny model kept outputs stable across many inserts:

![](Pasted%20image%2020260404192146.png)

Simple start graph:

![](Pasted%20image%2020260404192219.png)

After many sequential layers:

![](Pasted%20image%2020260404192232.png)

ResNet-scale behaviour: `tests/regression/resnet_regression_test.py` (see `tests/regression/ResNet18 regression script.md`).

---

## Related code

`LayerShapeAnalyser`, `LayerBridgeFinder` in `growingnn/actions/utils/layer_analyser.py`. Graph insert: `add_new_seq_layer` in `growingnn/actions/utils/model_transformations.py`.
