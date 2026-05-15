This vault folder documents the `growingnn` Python package in repo root `growingNN_pytorch`. Start here for links. The code trains networks whose graph can change during training. The main reference chapter is DOI `10.1007/978-3-031-63749-0_25` (see [[Paper and references]]).

### Core

- [[Config]] in `growingnn/core/config.py`
- [[Logger]] in `growingnn/core/logger.py`
- [[Package version]] in `growingnn/__init__.py`

### Actions (growth and shrink)

- [[Base action and Layer Type]] in `growingnn/actions/action.py` and `Layer_Type` in `layer_Factory.py`
- [[Residual Linear Actions]]
- [[Residual Conv Action]]
- [[Sequentail Linear Actions]] (file name keeps the spelling typo from early drafts)
- [[Sequential Conv Action]]
- [[Del Layer Action]]
- [[Delete neurons action]] stub in `growingnn/actions/delete_neurons.py`

### Action utilities

- [[Model Analyser]] in `growingnn/actions/utils/model_analyser.py`
- [[Model Transformer]] in `growingnn/actions/utils/model_transformations.py`
- [[Layer Factory]] in `growingnn/actions/utils/layer_Factory.py`
- [[Name factory]] in `growingnn/actions/utils/name_factory.py`
- [[Conv to linear adapter]] in `growingnn/actions/utils/conv_to_linear_adapter.py`
- [[Quasi identity]] in `growingnn/actions/utils/quaziIdentity.py`
- [[Dotted Module Names in torch.fx]]
- [[FX Shape Probe]] in `growingnn/actions/utils/fx_shape_probe.py`

### Repo utilities

- [[FX graph drawer]] in `growingnn/utils/fx_graph_drawer.py`

### Tests

- [[Unit tests index]]
- [[Test runners]]
- [[Model factory]] in `tests/model_factory.py`
- [[ResNet18 regression script]]
- [[Utils testing script]]
- [[Regression utils]]
- [[Adding residual layers]]
- [[Adding Sequential Layers]]

### Lab notes

- [[Part 1]] dated log
- [[PLAN]] and [[TODO]] in Reports
