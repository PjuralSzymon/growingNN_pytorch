This page covers `growingnn/actions/action.py` and the enum `Layer_Type` defined in `growingnn/actions/utils/layer_Factory.py` (lines 16 to 19).

### Base class `Action`

Fields. `__init__` stores `self.params = _params` at line 8 in `action.py`.

Methods. `execute(self, model)` is a stub at lines 11 to 12. Subclasses override it. `can_be_infulenced(self, by_action)` is a stub at lines 14 to 15. `generate_all_actions(model)` at lines 17 to 20 returns an empty list on the base class; real actions override with `@staticmethod` patterns that take `nn.Module` or `fx.GraphModule`.

### Enum `Layer_Type`

Values. `ZERO = 1`, `RANDOM = 2`, `EYE = 3` in `layer_Factory.py`.

Use. [[Residual Linear Actions]] passes `layer_types=[Layer_Type.EYE]` in regression. [[Layer Factory]] maps each value to a different init for new `nn.Linear` rows.

### Comparison with the original growingNN paper

The paper speaks about several init modes for new weights. This repo encodes three discrete modes in `Layer_Type`. Residual add paths in this repo prefer zero or small random init rather than quasi-identity for new weights, while `Layer_Type.EYE` remains available for explicit eye init via [[Layer Factory]].

### Known limitations

The base `generate_all_actions` is not used in production; each action class defines its own. `DelNeurons` in `delete_neurons.py` is still a stub (see [[Delete Neurons action]]).
