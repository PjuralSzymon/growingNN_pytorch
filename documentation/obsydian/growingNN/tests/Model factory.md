
This note is about `tests/model_factory.py` and the class `ModelFactory`.

What it does. It builds small `torch.nn.Module` objects for unit tests and for drawings in regression scripts. Why. Tests need fixed width, short forward paths, and known edge cases. Where. Imported from `tests/unit/actions/utils/layer_analyser_test.py`, from `tests/regression/utils_testing.py`, and from other tests under `tests/`. Layout notes: `documentation/obsydian/growingNN/tests/Unit tests.md`.

### Related

Model Analyser, `tests/regression/utils_testing.py`, `tests/regression/resnet_regression_test.py`.
