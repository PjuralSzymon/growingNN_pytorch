# Matrix rescaling / `propagate_neuron_change` issue

## Summary

Neuron add/delete rescales a `nn.Linear` output, then must propagate the new feature width through the traced FX graph so every downstream consumer (linears, convs, norms, residual adds) stays shape-consistent.

The old `propagate_neuron_change` in `growingnn/actions/utils/layer_resize.py` used a recursive depth-first walk with ad-hoc early returns. That design silently dropped propagation at several module types and left the graph in a broken state that only surfaced during longer training/simulation runs.

## Root causes

### 1. LayerNorm was not a width-propagating module

`PASSTHROUGH_MODULES_TO_UPDATE` only listed `BatchNorm{1,2,3}d`. Models wrapped with `nn.LayerNorm` (see `ModelFactory.complex_residual_many_widths_with_activation`) hit a dead end:

- `merge` shrank from 11 → 5 features
- `norm_merge` stayed at `normalized_shape=(11,)`
- downstream `r4_a` kept `in_features=11`
- forward pass failed: `expected input [*, 11], got [2, 5]`

The same failure occurred for `stem` and `expand`, which sit behind `LayerNorm` in the activation stack.

BatchNorm-backed layers (`r1_up`, `r2_a`, …) worked because BatchNorm was already in the update set.

### 2. `call_module` passthrough ops were skipped (ReLU, Dropout, …)

The redesigned `_propagate_edge` initially handled `call_module` nodes before checking `NodeTypeChecker.is_passthrough`. ReLU/Dropout live in `PASSTHROUGH_MODULES` (not `PASSTHROUGH_MODULES_TO_UPDATE`), so they hit the “skip non-resizable module” branch and **never re-enqueued**.

Effect on `complex_residual_many_widths`: shrinking `stem` updated `stem.out_features` but stopped at the first `act` (`nn.ReLU`) node. `r1_up.in_features` stayed at 12 while the trunk carried width 6.

Fix: evaluate `is_passthrough` before the generic `call_module` resizable/norm branches.

### 3. Square-layer output rescale used post-input shape

When a square `nn.Linear` (e.g. `256→256`) received a shrunk add input (`230`), input rescale ran first (`230→256`). The follow-up check used `_is_square_resizable(updated)` on the **already rescaled** module (`230×256`), so it no longer looked square and output stayed at `256`.

On `cifar_minimal_res_conv_fork_hidden`, shrinking `hidden` to `230` left `seq_linear_0` output at `256`, then sibling sync at `add_1` **expanded `hidden` back to `256`**, undoing the shrink.

Fix: capture `was_square` on the original module before input rescale (same semantics as the pre-redesign code).

### 4. Silent propagation stop on unknown modules (design flaw)

The old forward walk treated unknown `call_module` nodes as non-resizable and **did not enqueue their users**:

```python
else:
    logger.debug("propagate_neuron_change --- skip non-resizable module: %s", name)
    # propagation stopped here — no forward continuation
```

Any norm/width-preserving module missing from the update set broke the entire downstream chain, not only the norm node itself.

### 5. Restrictive pre-checks masked the problem

`can_resize_linear_output` could return `True` while runtime propagation still failed, because the unsizable pre-check did not model LayerNorm gaps. Actions were generated and executed, then simulation crashed mid-rollout.

### 6. Recursive walk + scattered `seen` keys

Forward (`"p"`), backward sync (`"s"`), and input-align (`"b"`) passes shared one `seen` set but used different visit semantics. The logic was hard to extend and easy to break when adding new topology handlers.

## Fix (redesign)

1. **Extend width-updating modules** — add `nn.LayerNorm` to `PASSTHROUGH_MODULES_TO_UPDATE` in `growingnn/core/config.py`.

2. **Norm rescaling** — `_rescale_layer_norm` reprojects `weight`/`bias` with `get_reshsper`, mirroring BatchNorm handling. `_norm_feature_width` unifies width queries across norm types.

3. **Queue-based forward propagation** — `propagate_neuron_change` now drives a `deque` work queue:
   - each visit processes all users through `_propagate_edge`
   - width-updating norms rescale and re-enqueue
   - passthrough ops re-enqueue without mutation
   - residual adds run sibling sync (`_prepare_add_node`) before continuing
   - resizable linears/convs rescale inputs and enqueue with their new output width

4. **Width analysis** — `NodeWidthAnalyser.node_output_width` reads `LayerNorm.normalized_shape[0]`.

## Reproduction (before fix)

```python
import torch.fx as fx
from growingnn.actions.delete_neurons import DelNeurons
from tests.model_factory import ModelFactory

gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths_with_activation())
DelNeurons(["merge", 0.5]).execute(gm)
gm(torch.randn(2, 4))  # LayerNorm shape mismatch
```

## Expected behavior (after fix)

- Shrinking `merge` updates `norm_merge.normalized_shape` to match `merge.out_features`.
- Downstream linears (`r4_a`, `r4_b`, `head`) receive consistent `in_features`.
- Mixed grow/shrink simulation loops over norm-wrapped residual models complete without shape errors.

## Remaining limitations

- `GroupNorm`, `InstanceNorm`, `cat`, and `view` reshapes are still unsupported on the propagation path.
- Shared modules at multiple FX call sites still require `all_sites_match_width` before input rescale.
- Propagation pre-check (`propagation_hits_unsizable`) is conservative and may reject valid moves on exotic topologies.
