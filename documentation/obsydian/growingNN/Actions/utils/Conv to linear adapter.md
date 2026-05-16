File: `growingnn/actions/utils/conv_to_linear_adapter.py`. Function: `can_insert_conv_before_linear(conv_out_channels, linear_in_features)`.

---

## What it does

Returns true when `linear_in_features % conv_out_channels == 0` (positive integers).

---

## Why

After global pool and flatten, a conv with `C` output channels gives a length-`C` vector per spatial cell (when pooled to 1×1). A following linear expects `in_features`. Divisibility is a cheap static check.

---

## Where it is used today

Production actions use [[Layer Analyser]] `LayerBridgeFinder.find_conv_before_linear_sizes` instead (same `%` rule at lines 167 to 170 in `layer_analyser.py`).

`can_insert_conv_before_linear` remains for direct tests or future callers. [[Residual Conv Action]] does not import this file anymore.

---

## Generating actions

Not an action class. No `generate_all_actions`.

---

## Executing actions

Not applicable.

---

## Comparison with the original growingNN paper

The paper does not name this helper. It is an implementation detail for conv-to-linear bridges.

---

## Known limitations

1. Divisibility is necessary, not sufficient for every spatial layout.
2. Prefer shape-based checks in [[Layer Analyser]] for new code.

---

## Related

[[Layer Analyser]], [[Residual Conv Action]], [[Layer Factory]].
