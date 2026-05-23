This page is about `growingnn/actions/utils/conv_to_linear_adapter.py` and the function `can_insert_conv_before_linear`.

### What it does

It returns a boolean. Inputs are `conv_out_channels` and `linear_in_features` (both positive integers). It is true when `linear_in_features % conv_out_channels == 0` (lines 2 to 6 in the source file).

### Why

After adaptive pool and flatten, a conv map with `out_channels` produces a vector length equal to `out_channels`. A following linear expects `in_features`. Exact divisibility is a cheap static check that those lengths can line up for some spatial layouts. [[Residual Conv Action]] calls this before proposing `ConvFactory.create_zero_conv_before_linear` in `layer_Factory.py`.

### Known limitations

Divisibility is necessary but not sufficient for every graph shape. Spatial size and batch layout still matter at runtime. [[FX Shape Probe]] does not filter conv-to-linear pairs today.

### Related

[[Residual Conv Action]], [[Layer Factory]], [[FX Shape Probe]].
