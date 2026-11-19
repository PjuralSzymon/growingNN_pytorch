# General mutation support roadmap: ResNet through GPT-2

# Implemented first slice (2026-11-11)

- Graph parameter accounting now counts direct `get_attr` Parameters and deduplicates aliases by object identity in addition to parameters reached via `call_module`.
- The `hf_symbolic_editable` capture route was verified on GPT-2: it keeps embeddings functional, transposes Hugging Face `Conv1D` weights into linear orientation, and preserves the native rank-3 projection interfaces. The source model is untouched. The initial implementation flattened/restored dimensions; the later native-rank update below removes that artificial graph structure.
- On the recorded GPT-2 run, the normalized graph had 944 nodes and 63 module calls; parameters matched at 124,439,808, generated token IDs matched, and logits were within `rtol=1e-5, atol=1e-6` (maximum absolute difference `3.81e-5`). Existing generators proposed 132 residual-linear actions, 12 sequential-linear actions, and 25 deletions. One residual and one sequential proposal were executed and returned finite logits with the expected output shape.
- In the first-slice run, the regression did not evaluate dropout or neuron-action families because of a prior slow traversal. The current script runs those generators; the 25 `DelLayer` proposals still include a learned MLP projection that needs semantic review, so proposal counts do not establish that every proposal is valid.

This does not establish that arbitrary regions are semantically safe or resize Transformer width. The capture route is an architecture-specific extension point, not a general replacement for HF symbolic capture. At the time of this first slice, Qwen model-level results had not yet been collected; the later Qwen run is recorded below.

## FX region and action integration update (2026-11-11)

- `GraphRegionAnalyzer` now returns node-based nested region records, crossing edges, structured interface metadata, and distinct structural/interface diagnostics. Dominator and postdominator analysis uses parent forests; parameter/buffer `get_attr` edges remain explicit boundary inputs but do not split activation-flow dominance. A 10,000-operation functional chain verifies deep walks do not rely on Python recursion.
- The FX analysis cache is invalidated after ShapeProp updates node metadata, so asking for action sites before shape propagation cannot leave stale missing-metadata rejection results cached.
- `ModelStructureEditor.insert_module_on_edge` rewrites an exact FX edge with any endpoint node kind. `AddSeqLinearLayer` retains module-pair actions and adds identity-initialized low-rank adapters at eligible FX edges, including rank-3 activations. Ranks 4, 8, and 16 are filtered by feature width and the parameter budget. It avoids shape/dtype/device-only consumers, direct parameter/buffer values, metadata failures, and edges already covered by a legacy path. Node names (not node objects) make generated actions executable against simulation graph copies. `AddResLinearLayer` also adds zero-initialized low-rank branches at recognized FX sum nodes, starting from existing merge operands; `torch.export` ATen tensor adds are recognized.
- Latest GPT-2 regression, after preserving native-rank Linear interfaces and adding functional pointwise width propagation: 591 nodes, 99 module calls, 49 rank-3 Linear calls, exact generated-token match, maximum logit error `4.20e-5`; 738 sequential-linear, 258 residual-linear, and 774 dropout proposals. Twelve MLP neuron-growth candidates were generated each at ratios 1.1 and 1.5; ratio 2.0 exceeded the model-relative parameter budget. Split/layout-changing attention QKV projections were rejected. Representative edge, merge, dropout, and MLP neuron-growth actions executed on separate copies and returned finite logits with the expected shape. Counts are proposals, not universal proofs of semantic safety.
- The latest warm-forward Qwen2.5-0.5B export run used explicit token IDs and captured 1,901 nodes with exact eager/export logits. Region discovery found 1,800 candidates; it found 2,355 action sites, 1,584 eligible, and generated 1,377 sequential-edge plus 288 residual-merge actions. Representative sequential and residual actions both preserved logits exactly on independent copies and returned finite outputs of shape `(1, 2, 151936)`. Full decoder groups remain descriptive because positional and residual side inputs cross at multiple internal nodes. The full unit suite now passes 498 tests, and the ResNet integration suite passes all seven tests. Plain root `pytest` collection hits the repository's duplicate `test_generate_content.py` module-name collision, so importlib mode is needed for the current full-unit run. LFM3 remains unverified: its cached config expects Transformers 5.12.1 and imports `PreTrainedConfig`, while the project environment has Transformers 4.45.1. Its config describes 24 alternating convolution and attention layers, making hybrid-architecture coverage a useful next target once a compatible runtime is available.
- This remains a partial foundation, not full model-agnostic mutation support. Legacy sequential module-pair proposals can still span longer functional paths; deletion remains semantically risky; neuron and resize actions remain module-specific; convolution actions still require a verified spatial/channel-axis interface. The current GPT-2 FX-edge action family creates options, but no validation has been performed for every candidate, gradients, training-mode behavior, aliasing, or parameter ties.
- Qwen2.5-0.5B was rerun against the current export, region, and action paths as recorded above; one sequential and one residual action were applied and executed on copies. The full unit suite now passes 498 tests and all seven ResNet integration tests pass. LFM3 loading still fails under the installed Transformers version, so its capture is unverified. Multimedia and time-series families are also unverified.

## Goal

Keep the existing ResNet and small-model experiments behaviorally stable while making the action system operate on **validated computational regions and tensor boundaries**, not only named `call_module` layers. A model family should be capturable, analyzable, and able to produce a useful set of safe growth proposals when its FX graph exposes the required operations. Model-specific capture adapters may describe architecture patterns, but candidate boundaries, interface checks, mutation records, and diagnostics should be shared.

### Scope of this refactor

- Limit this refactor to action generation's graph work: graph capture integration, graph analysis, candidate discovery, tensor-interface analysis, and graph transformation. Keep almost all implementation in `growingnn/utils/fx/`.
- Make the FX layer reusable across capture routes and architectures: it should describe graph nodes and edges, nested regions, tensor interfaces, edit locations, and diagnostic reasons without requiring a particular module hierarchy.
- Change files outside `growingnn/utils/fx/` only where required to connect existing action generators or `TracedModel` to the shared FX facilities. Preserve their public APIs and existing ResNet/small-model behavior where practical; do not broaden this into a general redesign of the action layer.
- Do not turn this into a rewrite of model loading, training, the action registry/policy, or every action implementation. Keep architecture-specific knowledge in capture/operation adapters, and keep mutation policy and action semantics at the existing action layer.
- Do not add block duplication or block deletion actions.

“Generate a lot of actions” should mean many distinct, reviewable choices at supported regions (placement, rank/width, initialization, and adapter type), not one action for every pair of the 970 GPT-2 graph nodes. Graph size alone does not establish that an edit is valid. The action generator should return candidates with explicit eligibility evidence and rejection reasons, and it should cap or rank combinatorial variants.

Do not add block duplication or block deletion actions. The existing generic `DelLayer` API is already producing a risky result on GPT-2 and should be treated as a separate existing behavior to audit, not expanded into block deletion.

## Evidence from the current GPT-2 run

- `HFTracer` produced a 970-node FX graph with 51 `call_module` calls. The traced graph preserved the “Hello world” logits exactly and greedy token IDs matched the original model.
- The model has 124,439,808 parameters. The full fidelity script counted that amount from the model and traced GraphModule directly.
- Existing proposal generation produced **25 total**: 24 `DelLayer` proposals for per-block dropout modules, and one for `model.transformer.wpe` (the learned positional embedding branch). Every add, convolution, dropout insertion, and neuron resize variant produced zero.
- Mean full-prompt forward time in that run was 0.008474s original and 0.009240s traced. That is a small observed overhead for one prompt/device/run, not a stable performance benchmark.
- This is not a failure to capture the model. It is a mismatch between the captured graph representation and the assumptions inside existing candidate generators and mutators.

The `wpe` proposal is especially revealing: the current deletion check treats a module that feeds an addition as a removable merge branch. Removing the positional embedding preserves a compatible-looking tensor shape but changes GPT-2 semantics. Shape compatibility is not sufficient evidence for a safe mutation.

## Current blockers, in order of impact

### 1. Candidate endpoints are editable modules, not graph boundaries

`ModuleClassifier.is_editable_module` only recognizes `EDITABLE_MODULES` (`nn.Linear` and convolution classes). `GraphStructureQuery.module_sequential_pairs` and `module_dependency_pairs` use these module nodes as endpoints and traverse other nodes only as connectors. Functional operations are not eligible endpoints even when they are the actual projection, residual add, activation, reshape, or attention operation.

GPT-2 uses Hugging Face `Conv1D` projections, and the HF trace represents much of the attention/MLP computation as lower-level FX function/method nodes. Existing pair enumeration therefore sees only a small subset of the captured computation. In the current action design, a 970-node graph does not create 970 useful opportunities.

### 2. Action implementations assume a small operator set

- `DelNeurons` remains module-backed. `AddNeurons` now supports module Linear resizing and a narrow functional-Linear chain, but not general functional projection groups.
- Layer factories create `nn.Linear` and mostly `nn.Conv2d` bridges.
- `EDITABLE_MODULES` omits Transformer `Conv1D`, functional linear projections, attention, and other projection forms.
- Resize propagation knows `nn.Linear`, `nn.Conv2d`, BatchNorm, and some LayerNorm cases. It does not coordinate GPT-2's fused QKV/projection widths, head dimensions, residual widths, embedding width, output head, and tied parameters.
- `PASSTHROUGH_MODULES`, `PASSTHROUGH_FUNCTIONS`, and width logic are explicit allowlists. They do not form a general semantic description of operators.

Adding `Conv1D` to one type tuple would not fix the system. Its weight orientation differs from `nn.Linear`, HF tracing can inline it, and safe width changes still need coordinated updates at all consumers and residual joins.

### 3. Shape gates are rank-specific and shape-only

`LayerBridgeFinder.linear_feature_dim` accepts only rank-2 `(batch, features)` shapes. `conv_channels` accepts rank-4 NCHW shapes. GPT-2 activations are rank 3 `(batch, sequence, hidden)`, where `nn.Linear`-style projections apply to the final feature dimension while the sequence dimension is preserved.

Current `LayerShapeAnalyser` caches one input/output shape per module target and reads only the first direct node argument. This cannot fully describe multiple tensor inputs, masks, tuples, cache values, structured outputs, symbolic dimensions, strides/layout, aliases, or side effects. The recent example-input support allows valid GPT-2 action probing; it does not yet solve general tensor-interface analysis.

### 4. Parameter accounting can omit functional weights

`GraphStructureQuery._sum_graph_module_parameters` sums parameters reachable from `call_module` nodes only. A traced graph that reads projection weights with `get_attr` and uses them in function nodes can therefore undercount parameters. The GPT-2 experiment's 124,439,808 count came from direct model parameter enumeration, not that legacy graph query. Graph accounting must count unique parameter/buffer references in both module and `get_attr` paths, accounting for tied weights only once.

### 5. Mutation editors are module-pair-oriented

Legacy `ModelStructureEditor` methods still insert between `call_module` endpoint IDs. The newer `insert_module_on_edge` can edit an exact FX operand occurrence at any node boundary, while the functional-linear resizer supports only a validated single-use producer/pointwise/sink chain. General multi-edge region rewrites and structured-value transformations remain unsupported.

### 6. Eligibility checks conflate shape with meaning

Existing delete and resize gates use topology plus shape/width checks. GPT-2 demonstrates that a shape-compatible branch can encode essential positional information. Functional purity, operation role, normalization, attention-head constraints, parameter tying, train/eval behavior, and semantics must be represented separately. Unsupported semantics should reject a candidate with a reason.

### 7. The registry multiplies fixed action families, not general candidates

`RunningConfig` and `registry.generate_all_actions` hard-code Boolean switches and fixed ratios/dropout rates. They select a family of existing generators; they do not ask a shared capability system which region operations are valid. The variants are useful policy choices, but the candidate discovery beneath them is module/rank-specific.

## Proposed design

### A. Keep three concerns separate

1. **Graph structure:** capture route, FX node identity, reachability, dominators/postdominators, region boundaries, nested regions, and all crossing edges.
2. **Tensor interface:** pytree structure, number/order of values, tensor rank/shape/symbolic dimensions, dtype, device, layout/stride, and mutation/alias annotations where available.
3. **Mutation capability:** which operation/region patterns a transformation understands, how it constructs new parameters, and how it reconnects/validates the graph.

An action is eligible only when all three layers pass. Candidate data should identify entry/exit nodes, internal node IDs, boundary edges, interface contract, transformation kind/variant, cost estimate, and diagnostics. Preserve string/module-target adapters for old public actions during migration.

### B. Add a descriptor registry, not another global module allowlist

Recognize operation semantics across `call_module`, `call_function`, `call_method`, `get_attr`, and nested GraphModules. Initial descriptors should include:

- module projections: `nn.Linear`, Conv1d/2d/3d, Hugging Face `Conv1D`;
- functional projections: `aten.linear`, `addmm`, and matmul-plus-bias patterns only when weight/bias provenance and transpose orientation are verified;
- normalization and activation operations;
- residual/add/concat joins;
- reshape, transpose, view, and flatten operations that preserve a known feature axis;
- embeddings, attention/softmax, and dropout as recognized semantic operations, with safe mutation capability declared separately.

Each descriptor should expose its feature axis or output-interface rule and parameter references. Unknown operations remain in regions and receive `unsupported_for_mutation` diagnostics; they are not silently treated as passthrough.

### C. Generate safe insertion actions at regions and tensor seams

The first general GPT-2 growth action is a shape-preserving residual adapter at selected rank-3 tensor seams, not an arbitrary edit to attention internals:

`y = x + W_out(activation(W_in(x)))`

with `W_out` initialized to zero. It preserves `(batch, sequence, hidden)` shape and initially preserves model output exactly. Candidate seams can be discovered around each block's attention output and MLP output when the graph confirms single tensor input/output, feature-axis compatibility, and safe side-effect behavior.

This gives a bounded, useful action family: for GPT-2's 12 blocks, two eligible seams per block and three bottleneck ranks would yield up to 72 distinct growth proposals. This is an acceptance target after eligibility and graph edits are validated, not a requirement to create invalid actions to inflate counts. More generic shape-preserving branch/action recipes can later apply to ResNet residual sites and other architectures.

### D. Treat width changes as coordinated graph transformations

Do not generalize `AddNeurons` by adding architecture-name checks. The functional path currently follows only known unary pointwise operations between one producer and one downstream functional Linear, updating both parameter matrices together. It rejects fanout, merges, layout changes, missing shape metadata, module sinks, and tied parameters. Attention QKV/head changes may need coordinated updates to fused projections, head reshape/permutation, output projection, normalization, residual branches, embeddings, LM head, and tied parameters. Keep those width edits unsupported until their full dependency region is represented and validated.

### E. Validate edits before exposing them as actions

For a candidate mutation:

1. apply to a cloned GraphModule (candidate generation stays read-only);
2. lint/recompile and run the same example inputs;
3. verify declared input/output pytree and tensor interfaces;
4. for zero-initialized residual adapters, assert initial outputs match within a stated tolerance;
5. verify gradients reach new parameters and old parameter sharing/ties are preserved;
6. report why a candidate was rejected if any check fails.

Do not use an equality-of-shapes rule as the sole safety test. Do not run or propose GPT-2 `wpe` deletion as a general safe edit. Existing dropout deletion may be a no-op in eval but still changes train-mode regularization; annotate its mode-dependent semantics.

## Implementation plan

### Phase 0 — Baseline correctness and reporting

- Keep the current GPT-2 fidelity script as the reference: exact prompt/token IDs, HF trace, original-vs-traced logits/text, graph images, timings, and action summaries.
- Correct `GraphStructureQuery.get_amount_of_parameters` to count unique reachable `get_attr` parameters/buffers and module parameters, then compare with the source model count.
- Add explicit action rejection reasons and audit the `wpe` deletion candidate; separate eval-only/no-op dropout candidates from semantic architecture changes.
- Preserve existing ResNet action counts and execution behavior as golden baselines.

### Phase 1 — General graph and interface inventory

- Promote region analysis to an explicit graph API used by action generation; include functional nodes and all crossing edges.
- Add structured tensor-interface descriptions, including rank-3 last-feature semantics and multiple values.
- Cache shape/interface metadata using actual examples or route metadata; avoid default float probes for integer inputs.
- Inventory parameters via node provenance, not only called modules.

### Phase 2 — Generic shape-preserving mutation substrate

- Introduce typed `ActionCandidate`/`ActionSpec` values with stable FX-node boundaries and diagnostics; adapt current `Action` APIs at the boundary.
- Add GraphModule edge/region insertion and replacement helpers that work when the source/target is a function or method node.
- Implement a zero-initialized bottleneck residual adapter for configured tensor feature axes; default to the final axis for rank-2/rank-3 values and allow callers to opt into other ranks/layouts only with an explicit policy.
- Enumerate per-region ranks/placements with deduplication and configurable candidate budgets.

### Phase 3 — GPT-2 projection and block support

- Add GPT-2 `Conv1D` and functional projection descriptors, including weight orientation and `get_attr` parameter ownership.
- Identify the 12 block boundaries, attention/MLP subregions, residual additions, and eligible hidden-state seams in the HF FX graph.
- Generate attention/MLP boundary adapters and report candidate counts by block, region kind, variant, and rejection reason.
- Keep feature-width expansion disabled until attention/head/normalization/residual/tied-weight coordination is complete.

### Phase 4 — Regression and expansion

- ResNet18 plus current unit/integration action behavior remains unchanged.
- GPT-2: exact original/traced fidelity, exact zero-init adapter fidelity, expected gradients, correct parameter accounting, and at least 72 eligible proposals when all 24 seams and three selected ranks are available.
- Execute representative candidates on graph clones and verify output/interface correctness before allowing training.
- Then add Qwen and ViT; add LFM3 only with compatible Transformers/model code and explicit hybrid conv/attention descriptors. Add multimedia/time-series adapters after the tensor-pytree interface supports their inputs.

## Success criteria

- Existing ResNet tests and experiment outputs remain stable for existing actions.
- GPT-2 source and traced parameter counts both equal 124,439,808, counting tied parameters once.
- Same prompt yields matching pre-mutation logits and generated token IDs between the source and HF-traced graph.
- Existing action families no longer silently present shape-only risky deletions as safe; candidates include semantic support/rejection diagnostics.
- GPT-2 produces nonzero growth proposals at attention/MLP block seams; initial zero-output adapters preserve output exactly.
- Candidate count scales with supported region placements and selected variants, not arbitrary all-pairs enumeration. Every proposed action is traceable to a named boundary and validation status.
- Region and mutation APIs can represent module, function, method, and nested subgraph nodes without requiring a module target string.

## Recommended priority

## Latest verification (2026-11-12)

- The refactor scope remains graph capture, graph analysis, region/candidate discovery, tensor-interface analysis, and graph transformation for action generation. Almost all shared implementation stays in `growingnn/utils/fx/`; changes to `TracedModel` and action generators are adapters that route existing actions through those shared facilities.
- The GPT-2 regression captured the unquantized model with `hf_symbolic_editable` (591 graph nodes, 99 module calls, 49 rank-3 Linear calls). Original and captured greedy token IDs matched; logits were allclose with maximum absolute error `4.20e-5`. It reported 1,876 proposals: 738 sequential-linear, 258 residual-linear, 774 dropout, 24 neuron-growth, and the existing deletion candidates. Representative sequential, residual, and dropout candidates executed on copies and preserved initial logits exactly; a neuron-growth candidate executed with finite output.
- A fresh local Qwen2.5-0.5B run captured 1,861 nodes through `torch.export`; eager and exported logits matched. AddNeurons discovered 24 grouped functional width-growth actions, corresponding to the gated feed-forward regions. One action executed and returned finite logits of shape `(1, 5, 151936)`. This verifies the supported MLP pattern; it does not enable coordinated attention/head, residual-width, or tied embedding/LM-head changes.
- The focused AddNeurons suite passed 17 tests. After adding transactional FX-edge insertion rollback, nested `torch.cond` discovery, and descriptive `addmm` projection coverage, the full unit and integration suites passed 521 tests, including the existing ResNet coverage.
- LFM3 received structural-only coverage through a temporary Transformers 5.12.1 overlay. Its custom model source required process-local shims for kernel/accelerate decorators and no-cache mask APIs. A meta-device model (1,728,801,536 parameters) exported to 2,570 FX nodes; analysis found 154 functional Linear projections. The generic action generators proposed 1,706 sequential-linear and 288 residual-linear actions; AddNeurons proposed 24 gated-MLP width actions at ratio 1.1. One neuron action executed on a graph copy and returned logits metadata shape `(1, 4, 128000)`. No checkpoint weights or numerical model fidelity were tested, so this is not full runtime coverage. The Transformers 4.45.1 project environment still cannot load the model without compatibility work. Multimedia and time-series model capture are also unverified.

Next, repeat LFM3 capture and action validation with a compatible Transformers runtime and real checkpoint weights, then add a hybrid conv/attention capture adapter only if the actual graph requires one. Continue broadening functional projection descriptors and region transformations based on graph structure, while keeping attention width edits rejected until reshape, split, tied-parameter, and residual dependencies have explicit transformation rules. Multimedia and time-series captures remain additional coverage work after task-shaped interfaces are available.
