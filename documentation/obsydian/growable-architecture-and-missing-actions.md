# GrowingNN: what architectures the actions can build, and what is missing

Handoff for implementing missing architecture actions. Based on the code on branch `work/new-res-block-grow-action` (`growingnn/actions/`, `growingnn/utils/fx/`, `growingnn/core/config.py`).

## 1. How the graph is edited

- The model is a `torch.fx.GraphModule` wrapped in `TracedModel` (`growingnn/core/traced_model.py`). Every action edits this graph in place.
- **Editable modules** (`config.EDITABLE_MODULES`) are `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, and `nn.Conv3d`. Only these can be endpoints of an action. Batch norm, ReLU, pooling, flatten, and dropout are never endpoints.
- **Hidden module**: a `call_module` node that is neither the first layer (fed by the input) nor the last layer (feeding the output) (`ModuleClassifier.is_hidden_module`).
- **Sequential pair** `(A, B)`: B is the next editable module after A on a forward path; non-editable nodes between them are skipped (`module_sequential_pairs`). The input and output layers are allowed.
- **Dependency pair** `(A, B)`: B is any editable module reachable forward from A, and B must be **hidden** (`module_dependency_pairs`). Residual actions use these pairs.
- **Sequential insert** (`ModelStructureEditor.add_new_seq_layer`): the new module is placed directly in front of B, on the path from A to B.
- **Residual insert** (`connect_residual_branch` in `utils/fx/sum_nodes.py`): `new_module(A_output)` is summed into **B's raw output**, right after B's conv/linear call and before B's batch norm or activation. Several branches into the same B share one `nary_add`.
- **Width propagation** (`actions/utils/layer_resize.py`): when a Linear output is resized, consumers are rescaled (Linear and Conv2d inputs, BatchNorm features). The classifier head's output is pinned.

## 2. Every action, exactly what it inserts

| Action (class) | Where | Inserted module | Initialization | Can change width? |
| --- | --- | --- | --- | --- |
| Add seq conv (`AddSeqConvLayer`) | Between a sequential pair with identical 4-D shapes, or just before global pool/flatten | `Conv2d(C, C, k, stride=1, padding=p)`: k and p copied from A (or 1×1 before flatten) | "Eye": a 1 at the kernel centre on the channel diagonal (identity) | **No**: C in = C out = existing channels |
| Add seq conv + BN + ReLU (same class, `add_batchnorm_and_relu=True`) | Same | `Conv2d(C, C)` → `BatchNorm2d(C)` → `F.relu` | Eye conv, default BN | No |
| Add residual conv (`AddResConvLayer`) | Dependency pair with identical 4-D outputs | `Conv2d(C_A, C_B, k_A, stride=1, p_A)` summed into B's output | **Zero** weights and bias, so the output is unchanged | No (C_A = C_B is required) |
| Add residual conv → linear (same class) | A is a conv, B a linear later in the graph | `Sequential(Conv2d(C_A, out_B), AdaptiveMaxPool2d(1), Flatten)` summed into B's output | Zero | No |
| Add seq linear (`AddSeqLinearLayer`) | Sequential pair of linear-shaped layers, or after conv features before a linear | `Linear(F, F)`: F = existing feature width | Eye (quasi-identity) | No |
| Add residual linear (`AddResLinearLayer`) | Dependency pair (flattened feature widths) | `Linear(F_A, F_B)` summed into B's output | Eye or zero | Bridges F_A → F_B, but adds no new width |
| Add neurons ×1.1 / ×1.5 / ×2.0 (`AddNeurons`) | Hidden `nn.Linear` only | Rescaled copy of the Linear with more outputs; consumers rescaled | Rescaling matrix (`get_reshsper`) | **Linear only** |
| Delete neurons ×0.1 / ×0.5 / ×0.9 (`DelNeurons`) | Hidden `nn.Linear` only | Rescaled smaller Linear | Rescaling matrix | Linear only |
| Add dropout 0.1 / 0.2 / 0.5 (`AddSeqDropoutLayer`) | Sequential pair with matching shapes, no dropout already on the path | `Dropout` / `Dropout2d` | — | No |
| Delete layer (`DelLayer`) | Any hidden module whose inputs can feed its outputs directly | Removes the module | — | No |
| Residual block duplicate / remove / merge (`ResidualBlockAction`) | A closed residual region ending in an add, e.g. `relu(x + BN(conv(x)))` | Copy of the block (quasi-identity) after the block or its ReLU; or remove / merge the block | Quasi-identity: identity conv, reset BN | No |

Limits: `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE = 1_000_000` for new Linear layers. Every conv is created with `stride=1`.

## 3. What can and cannot be built

### Can be built

- **Depth**: any number of same-width convs (3×3 or 1×1), with or without BN + ReLU, and same-width linears in the head.
- **Skip connections**: zero-initialized conv skips between any two same-shape conv outputs (also long skips), conv → head skips through global max-pool, and linear skips.
- **Residual blocks**, but only by copying a block that already exists (residual block action). From scratch, the closest result is `B_out + zeroconv(A_out)` with no batch norm in the branch.
- **Head width**: hidden linears can widen or shrink.
- **Regularization**: dropout; layers can be deleted.

### Cannot be built (the gaps)

1. **Wider convolutions: the most important gap.** Nothing changes a conv's channel count. `AddNeurons` and `DelNeurons` raise for anything other than `nn.Linear`, and every new conv copies the current width. **The channel count of the first conv is the channel count of the whole network forever.** Width propagation could already rescale conv inputs and outputs (`ConvFactory.create_conv_with_rescaled_output_channels` / `create_conv_with_rescaled_input_channels` exist and are used in `layer_resize.py`), but no action triggers it for a conv.
2. **Batch norm after an existing layer.** BN only arrives as part of a new conv + BN + ReLU. There is no "add BN after conv X" action, so the first conv (stem) can never get BN.
3. **Residual block with BN in the branch from scratch.** The residual conv branch is a bare zero conv. Nothing builds `x + BN(conv(x))`, or the standard two-conv `x + BN(conv(ReLU(BN(conv(x)))))`, except duplicating an existing block.
4. **Down-sampling / resolution changes.** No action adds a stride-2 conv or a pooling layer. All convs run at full input resolution (32×32 on CIFAR) until the final global pool. This is a large compute cost, and it means there are no multi-scale stages like ResNet's 32→16→8.
5. **Activations on existing paths.** ReLU only arrives with conv + BN + ReLU. Plain seq conv and linear inserts are linear (identity), so stacking them adds no nonlinearity.
6. **Kernel size.** Fixed to the source conv's kernel (usually 3×3) or 1×1 before flatten.

## 4. Concrete example: Experiment 010 start vs Experiment 013/014 start

**Experiment 010 starter** (`TinyCifar10Starter`, `experiments/train_cifar10_exp010_mcts_meta.py`), 296 parameters:

```text
input 3×32×32
conv1: Conv2d(3→8, 3×3, pad 1, no bias)
torch.relu
adaptive_avg_pool2d(1) → flatten
linear2: Linear(8→10, no bias)            # head, output pinned
```

At the start, `conv1` is fed by the input and `linear2` feeds the output, so **neither is hidden**. Only actions that need sequential pairs are available: seq conv(8→8), seq conv + BN + ReLU(8→8), seq linear(8→8), and dropout. Residual actions and add neurons become possible only after a hidden layer exists. **Every conv stays at 8 channels for the whole run.**

**Experiment 013/014 starter** (`ResidualStarter`, `experiments/train_cifar10_exp013_residual_block_multidataset.py`), 2,970 parameters:

```text
input 3×32×32
stem: Conv2d(3→16, 3×3, pad 1, no bias) → stem_bn: BatchNorm2d(16) → relu
block_conv: Conv2d(16→16, 3×3, pad 1, no bias) → block_bn: BatchNorm2d(16)
add (stem output + block output) → relu
adaptive_avg_pool2d(1) → flatten
classifier: Linear(16→10)
```

**This network is not reachable from the Experiment 010 starter:**

| Needed | Reachable? | Missing action |
| --- | --- | --- |
| Stem 8 → 16 channels | No | Conv widening (gap 1) |
| BN directly after the stem | No | Add BN after an existing layer (gap 2) |
| `x + BN(conv(x))` block | Only as `x + zeroconv(x)` (no BN, added to a later layer's output) | Residual block from scratch (gap 3) |

In experiments, the 296-parameter start reached 46.7–59.2% CIFAR-10 validation after 200 epochs (Experiment 010). The 2,970-parameter start reached 68.5% at epoch 200 and 72.4% at epoch 300 with no residual-block actions (Experiment 014 `off`). The training recipe and search settings were the same or weaker for Experiment 014, so the starter matters a lot.

## 5. Suggested new actions (priority order)

All of them should preserve the function (the network output right after the action equals the output before it, or nearly). Tests should compare outputs before and after in eval mode, as `tests/unit/actions/residual_block_design_test.py` does.

1. **Widen conv** (`AddConvChannels`, ratios e.g. ×1.5 / ×2): grow `out_channels` of a hidden or first conv. The consumers' input channels grow (next conv input, BN features, residual partners on the same `nary_add`, the head Linear via pool). Function-preserving options: Net2Net (copy existing filters, divide outgoing weights among copies), or new filters with zero outgoing weights. Watch the sum nodes: every term of an `nary_add` must be widened together. The existing rescale helpers and `fix_graph_widths` are the place to start.
2. **Shrink conv** (counterpart, for the shrink side of the algorithm).
3. **Add BN after a conv**: insert `BatchNorm2d(C)` after a conv. To preserve the output, set `running_mean`/`running_var` to the current activation statistics and γ = √var, β = mean, so the layer is the identity at insertion.
4. **Add residual block from scratch**: at a 4-D activation `x`, insert `x + BN(conv(x))` (optionally conv-BN-ReLU-conv-BN), with the last BN's γ = 0 so the block starts as an identity. This makes the residual block action useful from a block-free starter.
5. **Down-sample stage**: insert a stride-2 conv (or pooling), possibly together with widening (ResNet stage: halve resolution, double channels). It changes output shapes downstream, so it needs width/shape propagation and cannot be exactly output-preserving. Consider it last.

## 6. Where to plug new actions in

- Implement an `Action` subclass in `growingnn/actions/` with `_execute(traced)`, `generate_all_actions(traced, ...)`, `can_be_infulenced`, and `__str__`.
- Register it in `growingnn/actions/registry.py` behind a new `RunningConfig.ACTIONS_ENABLE_*` flag in `growingnn/core/config.py`. Also add the flag to `update_grow_actions` / `update_shrink_actions`.
- The MCTS uses `generate_all_actions` output directly. Each extra candidate competes in a search with ~150 candidates and noisy one-epoch scores (Experiment 014), so keep the candidate count per action type moderate.
