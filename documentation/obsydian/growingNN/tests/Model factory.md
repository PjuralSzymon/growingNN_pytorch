
This note is about `tests/model_factory.py` and the class `ModelFactory`.

What it does. It builds small `torch.nn.Module` objects for unit tests and for drawings in regression scripts. Why. Tests need fixed width, short forward paths, and known edge cases. Where. Imported from `tests/unit/actions/utils/layer_analyser_test.py`, from `tests/regression/utils_testing.py`, and from other tests under `tests/`. Layout notes: `documentation/obsydian/growingNN/tests/Unit tests.md`.

---

### New factory `deeply_nested_submodules`

Added near line 309 in `tests/model_factory.py`. It returns `ModelDeeplyNested()` with nested classes `InnerBlock`, `MiddleBlock`, `OuterBlock` inside the factory method. Forward path is `stem` then `outer` then `head`. Each block holds `nn.Linear` and `nn.ReLU` submodules.

Why. FX `call_module` targets use dotted paths such as `outer.middle.inner.l1`. That string is a single qualified name, not one Python attribute. Tests for Model Analyser (`get_layer_module` in `growingnn/actions/utils/model_analyser.py`) use this layout to match real nets like `torchvision.models.resnet18`.

Technicalities. Input and output width stay `4` so it can replace flat chains in some tests. Names include `outer.middle.inner.act`, `outer.middle.l1`, `outer.act`, and `head`.

### Related

Dotted Module Names in torch.fx (`growingnn/actions/utils/model_analyser.py` and action generators), Model Analyser, `tests/regression/utils_testing.py` script that traces a nested model and writes PDF graphs under `testResults/regression/`.
