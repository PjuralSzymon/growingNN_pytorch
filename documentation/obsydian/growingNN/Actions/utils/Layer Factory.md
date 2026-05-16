This page is about `growingnn/actions/utils/layer_Factory.py`. It builds `nn.Linear` and `nn.Conv2d` (and small `nn.Sequential` stacks) for actions.

### `Layer_Type`

Enum at lines 16 to 19: `ZERO`, `RANDOM`, `EYE`. See [[Base action and Layer Type]].

### `LinearFactory`

`create_linear(in_features, out_features, type)` dispatches at lines 24 to 32.

`create_zero_linear` sets weight and bias to zero (lines 35 to 38).

`create_random_linear` draws normal weights using mean and std from `ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE` in [[Config]] (lines 43 to 46).

`create_eye_linear` builds a near-identity map using `quaziIdentity.eye_stretch` then transposes to match `nn.Linear` weight layout `(out_features, in_features)` (lines 50 to 55). Used by [[Residual Linear Actions]] when `Layer_Type.EYE` is selected.

### `ConvFactory`

`create_conv` supports `ZERO` and `EYE` only at lines 62 to 67; `RANDOM` raises.

`create_zero_conv` zero initialises weight and bias (lines 70 to 74).

`create_eye_conv` places a 1 on the centre tap of the kernel on the channel diagonal (lines 77 to 91). Used by [[Sequential Conv Action]].

`create_zero_conv_before_linear` wraps a zero conv with `AdaptiveMaxPool2d(1)` or `AdaptiveAvgPool2d(1)` depending on `RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE` in [[Config]], then `Flatten` (lines 95 to 111). Used by [[Residual Conv Action]] for conv-to-linear skips.

### Known limitations

`create_conv` does not implement `Layer_Type.RANDOM` for conv. `create_zero_conv_before_linear` returns `nn.Sequential`, not a bare `Conv2d`; FX `call_module` for the inserted name must point at that submodule as a whole.

### Related

[[Residual Linear Actions]], [[Residual Conv Action]], [[Sequential Conv Action]], [[Config]], [[Quasi identity]].
