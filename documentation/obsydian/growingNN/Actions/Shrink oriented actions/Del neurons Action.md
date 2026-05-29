[[Actions]]

This page is about neuron-level shrink in growingNN. The R5 class is `DelNeurons` in `growingnn/actions/delete_neurons.py`. The behaviour below comes from the old repo class `Del_neurons`. It still needs a full refactor for torch.fx and `nn.Module` weights.

This should work as this: 
generate_all_actions:
1. find all hidden modules 
2. for only those that are capable that the outpur shape can be shrinked so for example linear layers onyl layers that have matricx multiplicateion 

execute:
1. For the given layer chagne the weight (and maybe bias) matrix with quazi identidtiy module so the output willahve diffrent shape
2. Recurisvely go over each output layer
2.1. if that's only dropout activation or other layer which only forwards the same shape adjust it so it can handle new shape
2.2. If that's layer like linear that it gets some shape and matrix inisde depends on it refactor also the matrix shape s it can handle new shape
2.3 After layer iput shape was adjusted we can treeat it like a leaf so it is ok 

Write unit test:
1. With model that start with conv layer then flatten then one linear layer then residual connections and on the path activation, dropout soething, then at the end of each conenction 2 linear layers and output layer 
2. execute the descrease neurons for the first layer and check the results 

## Pseudocode

Legacy `Del_neurons` (target shape for R5 on `fx.GraphModule` + `nn.Linear`).

```
function generate_all_actions(model, scale_neurons_ratio = 0.5):
    actions = []
    for layer_id in model.hidden_layers + model.input_layers:
        layer = model.get_layer(layer_id)
        if layer is Conv:
            continue
        new_neurons = floor(layer.neurons * scale_neurons_ratio)
        if new_neurons >= MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
            actions.append(Del_neurons([layer_id, scale_neurons_ratio]))
    return actions


function execute(model, params):
    layer_id, reduce_ratio = params[0], params[1]
    model.forward_blank()
    model.get_layer(layer_id).scale_neurons(reduce_ratio)


function scale_neurons(layer, reduce_ratio):
    new_width = max(1, int(layer.neurons * reduce_ratio))
    old_width = layer.neurons
    resheper = get_reshsper(old_width, new_width)

    layer.W = Reshape(layer.W, old_width, resheper)
    layer.B = Reshape(layer.B, old_width, resheper)
    layer.neurons = new_width

    for output_id in layer.output_layers_ids:
        out = model.get_layer(output_id)
        start, end = out.get_weight_matrix_indexes_for_layer_id(layer.id)
        if start == 0 and end == 0:
            continue

        W_before = out.W[:, :start]              if start > 0 else none
        W_mid    = out.W[:, start:end]         # columns from this layer
        W_after  = out.W[:, end:]              if end < out.W.cols else none

        W_mid_scaled = Reshape(W_mid.T, W_mid.cols, resheper).T
        out.W = hstack(W_before, W_mid_scaled, W_after)
        out.input_size = out.W.cols
```

R5 placeholder (not implemented yet):

```
function generate_all_actions(gm):
  # TODO: hidden/input call_module ids from Torch.fx GraphStructureQuery
  # TODO: nn.Linear only, same minimum width rule from Config
  return []

function execute(gm, params):
  # TODO: edit linear.weight / linear.bias + fan-out linear.weight columns
  pass
```

## Generating actions

In the old code, `generate_all_actions(model, scale_neurons_ratio=0.5)` walked `model.hidden_layers` and `model.input_layers`. For each layer id it skipped `Conv` layers. It computed `new_neurons = floor(neurons * scale_neurons_ratio)`. It emitted `Del_neurons([layer_id, scale_neurons_ratio])` only when `new_neurons >= config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL`.

In R5, `DelNeurons.generate_all_actions` at lines 20 to 22 in `delete_neurons.py` is empty. A future version should list candidate `call_module` targets from [[Torch.fx]] `GraphStructureQuery` (similar to [[Del Layer Action]]) and filter by minimum width from [[Config]].

## Executing actions

Old `execute` called `model.forward_blank()` then `model.get_layer(params[0]).scale_neurons(params[1])`.

`scale_neurons(reduce_ratio)` on one layer did three things. First it set `neurons_reduced_amount = max(1, int(self.neurons * reduce_ratio))`. Second it reshaped `self.W` and `self.B` with `Reshape` and `get_reshsper` from [[Quasi identity]]. Third it walked `self.output_layers_ids`, sliced each successor weight matrix at columns for this layer, reshaped that slice to the new width, wrote it back with `np.hstack`, and updated `output_layer.input_size`.

R5 `DelNeurons.execute` is `pass` at lines 14 to 15. FX work must replace direct `W` / `B` arrays with in-place edits on `nn.Linear.weight` and `nn.Linear.bias`, or graph surgery via [[Torch.fx]] `ModelStructureEditor` / `NodeEditor`.

## Comparison with the original growingNN paper

The Springer chapter (DOI 10.1007/978-3-031-63749-0_25) treats architecture search at several scales. Neuron removal is finer than whole-layer delete in [[Del Layer Action]]. The old code kept the network runnable by rescaling outgoing weights with the same `get_reshsper` map used for the layer itself.

## Known limitations

1. Not implemented in R5: `delete_neurons.py` is a stub; imports for `delete_layer` and seq helpers are unused.
2. Old design assumed custom layers with `.neurons`, `.W`, `.B`, and `.output_layers_ids`. torch.fx uses submodule paths and `Parameter` tensors instead.
3. Conv layers were excluded in generation; conv channel shrink is still undefined for this repo.
4. `can_be_infulenced` returned `False` in the old class; downstream actions could not chain off a neuron delete in that design.




Instruction on how to implmenet:


# Torch.FX Shape Propagation and Neuron Shrinking System

# Goal

This document describes a complete design for implementing:

- neuron shrinking
    
- output shape propagation
    
- residual/add synchronization
    
- recursive graph updates
    
- weight matrix adaptation
    
- graph traversal
    
- shape consistency validation
    
- FX-based transformation passes
    

inside a PyTorch + torch.fx dynamic architecture manipulation system.

The goal is to create a system similar in spirit to the original GrowingNN implementation but implemented using:

- torch.nn.Module
    
- torch.fx.GraphModule
    
- FX graph traversal
    
- module replacement
    
- weight slicing
    

instead of a custom graph engine.

---

# High Level Idea

When a layer changes its output shape:

```text
A.out_features: 128 -> 64
```

all downstream consumers of A must also be updated.

Example:

```text
A -> B
```

If:

```text
A output size = 64
```

then:

```text
B input size must also become 64
```

This becomes more complex with residual/add structures:

```text
A ----\
      add -> D
C ----/
```

If A shrinks to 64:

```text
C must also shrink to 64
D input must shrink to 64
```

This document explains how to build such a system.

---

# Core Philosophy

The system should NOT:

- directly mutate random tensors
    
- blindly resize weights
    
- rewrite graph structure unnecessarily
    

The system SHOULD:

1. Trace model with FX
    
2. Detect graph dependencies
    
3. Build propagation plan
    
4. Replace modules safely
    
5. Recompile graph
    
6. Validate shapes
    

---

# FX Basics

FX graph nodes:

```python
node.op
```

can be:

```text
placeholder
call_module
call_function
call_method
output
```

Important APIs:

```python
node.users
node.all_input_nodes
node.target
```

Example:

```text
x -> l1 -> relu -> l2 -> output
```

might become:

```text
placeholder(x)
call_module(l1)
call_function(relu)
call_module(l2)
output
```

---

# Shape Propagation Concept

The key abstraction:

```python
keep_idx
```

Example:

```python
keep_idx = tensor([0, 3, 5, 7])
```

means:

```text
keep only these neurons/channels/features
```

This propagates through the graph.

---

# Weight Matrix Rules

# Linear

PyTorch stores:

```python
Linear.weight.shape == (out_features, in_features)
```

## Shrink output

```python
new_weight = old_weight[keep_idx, :]
new_bias = old_bias[keep_idx]
```

## Shrink input

```python
new_weight = old_weight[:, keep_idx]
```

---

# Conv2d

PyTorch stores:

```python
Conv2d.weight.shape == (
    out_channels,
    in_channels,
    kernel_h,
    kernel_w,
)
```

## Shrink output channels

```python
new_weight = old_weight[keep_idx, :, :, :]
new_bias = old_bias[keep_idx]
```

## Shrink input channels

```python
new_weight = old_weight[:, keep_idx, :, :]
```

---

# BatchNorm

BatchNorm tracks channels.

Must shrink:

```python
weight
bias
running_mean
running_var
```

using:

```python
tensor[keep_idx]
```

---

# Passthrough Layers

These layers preserve shape:

```text
ReLU
Dropout
Identity
LeakyReLU
Tanh
Sigmoid
```

Propagation should continue through them.

---

# Flatten

Flatten is special.

Example:

```text
Conv output:
(B, 64, 8, 8)
```

Flatten:

```text
(B, 4096)
```

If channels shrink:

```text
64 -> 32
```

then flattened representation also shrinks.

Need conversion:

```python
channel_keep_idx
```

into:

```python
flatten_keep_idx
```

Example:

```python
flatten_keep_idx = []

for c in keep_channels:
    start = c * H * W
    end = (c + 1) * H * W
    flatten_keep_idx.extend(range(start, end))
```

---

# Add / Residual Handling

The hardest case.

Example:

```text
A ----\
      add -> D
C ----/
```

Constraint:

```text
shape(A) == shape(C)
```

Therefore:

```text
if A shrinks:
    C must also shrink
```

The same keep_idx must be applied.

---

# Correct Mental Model

Add nodes create:

```text
shape equality constraints
```

Meaning:

```text
mask(A) == mask(C)
```

---

# Example

Before:

```text
A.out = 128
C.out = 128
D.in = 128
```

After:

```text
A.out = 64
C.out = 64
D.in = 64
```

---

# Why Module Replacement is Better

Do NOT resize parameters in-place.

Bad:

```python
layer.weight = nn.Parameter(layer.weight[:, keep_idx])
```

Good:

```python
new_layer = nn.Linear(...)
new_layer.weight.copy_(...)
replace module
```

Reasons:

- optimizer state consistency
    
- proper metadata
    
- correct in_features/out_features
    
- simpler debugging
    

---

# Recommended Architecture

# Transformation Passes

Recommended passes:

```text
1. Discover changes
2. Build propagation plan
3. Apply replacements
4. Cleanup graph
5. Validate
```

---

# Suggested File Structure

```text
transformations/
    propagate_output_change.py
    shrink_linear.py
    shrink_conv.py
    shrink_batchnorm.py
    graph_utils.py
    add_constraints.py
    flatten_utils.py
    validation.py

actions/
    shrink_layer_output.py
```

---

# Recommended API

```python
shrink_layer_and_propagate(
    gm,
    layer_id,
    keep_idx,
)
```

---

# Core Recursive Algorithm

Pseudo:

```python
propagate_output_change(node, keep_idx):

    for user in node.users:

        if user is Linear:
            shrink Linear input

        elif user is BatchNorm:
            shrink BatchNorm
            propagate further

        elif user is passthrough:
            propagate further

        elif user is add:
            shrink all other branches
            propagate after add

        elif user is output:
            stop
```

---

# Full Pseudocode

```python
function propagate_output_change(node, keep_idx):

    if node already visited:
        return

    mark node visited

    for each user in node.users:

        if user is Linear:
            shrink input dimension

        elif user is Conv2d:
            shrink input channels

        elif user is BatchNorm:
            shrink channels
            recurse

        elif user is ReLU:
            recurse

        elif user is add:
            for each other branch:
                shrink branch output

            recurse after add

        elif user is output:
            continue

        else:
            raise unsupported
```

---

# Branch Shrinking

Pseudo:

```python
function shrink_branch_output(branch_node):

    if branch_node is Linear:
        shrink output

    elif branch_node is BatchNorm:
        shrink
        recurse backward

    elif branch_node is passthrough:
        recurse backward

    else:
        raise unsupported
```

---

# Validation System

After transformation:

```python
dummy = torch.randn(...)
out = gm(dummy)
```

If model runs:

```text
shape propagation likely succeeded
```

---

# Additional Validation

Validate:

```python
Linear.weight.shape[1] == previous_layer.out_features
```

and:

```python
add input shapes equal
```

---

# Important Design Decision

# Never Mutate During Discovery

Bad:

```python
walk graph
mutate graph immediately
continue walking
```

This creates:

- inconsistent traversal
    
- stale references
    
- recursive bugs
    

Instead:

```text
1. build plan
2. execute plan
```

---

# Transformation Plan

Example:

```python
[
    ShrinkOutput("A", keep_idx),
    ShrinkOutput("C", keep_idx),
    ShrinkInput("D", keep_idx),
]
```

---

# Suggested Operation Classes

```python
class ShrinkLinearOutput
class ShrinkLinearInput
class ShrinkConvOutput
class ShrinkConvInput
class ShrinkBatchNorm
```

---

# Cleanup Passes

After modifications:

```text
remove redundant add nodes
remove unused nodes
remove identity chains
```

---

# Redundant Add Detection

Example:

```text
add(x, identity(x))
```

can become:

```text
x
```

---

# Important Restriction

Do NOT remove:

```text
add(x, x)
```

unless intentional.

Because:

```text
x + x = 2x
```

---

# Safe Initial Scope

Initial implementation should support ONLY:

```text
Linear
BatchNorm1d
ReLU
Dropout
Identity
Residual add
```

Then later:

```text
Conv2d
Flatten
BatchNorm2d
Pooling
Concat
```

---

# Unsupported Operations Initially

Do not support initially:

```text
cat
view
reshape
permute
attention
LSTM
GRU
multihead attention
```

These complicate propagation significantly.

---

# Shape Contract Concept

Every node has:

```text
input contract
output contract
```

Example:

```text
Linear(128 -> 256)
```

Contract:

```text
input: 128
output: 256
```

Changing output contract requires updating all consumers.

---

# Dependency Graph Concept

This is effectively:

```text
constraint propagation
```

Example:

```text
A changed
↓
all consumers affected
↓
all add branches constrained
↓
all downstream consumers affected
```

---

# Comparison with Original GrowingNN

Original GrowingNN:

```text
custom graph
manual weight slicing
manual connectivity lists
```

New FX system:

```text
FX graph traversal
module replacement
recursive propagation
```

Conceptually they are equivalent.

---

# Mapping Old GrowingNN to FX

Old:

```python
output_layers_ids
```

New:

```python
node.users
```

Old:

```python
input_layers_ids
```

New:

```python
node.all_input_nodes
```

Old:

```python
get_weight_matrix_indexes_for_layer_id
```

New:

```python
tensor slicing using keep_idx
```

---

# Test Cases

# Test 1: Simple Linear Chain

```text
x -> A -> B -> output
```

Shrink A:

Expected:

```text
A output shrinks
B input shrinks
```

---

# Test 2: ReLU Passthrough

```text
x -> A -> relu -> B
```

Expected:

```text
A output shrinks
B input shrinks
```

---

# Test 3: Residual Add

```text
A ----\
      add -> D
C ----/
```

Expected:

```text
A output shrinks
C output shrinks
D input shrinks
```

---

# Test 4: Multiple Consumers

```text
        -> B
A ----|
        -> C
```

Expected:

```text
B input shrinks
C input shrinks
```

---

# Test 5: BatchNorm

```text
A -> BN -> B
```

Expected:

```text
BN shrinks
B input shrinks
```

---

# Test 6: Conv + Flatten + Linear

```text
Conv -> Flatten -> Linear
```

Expected:

```text
flatten indices remapped
linear input shrinks correctly
```

---

# Recommended Assertions

```python
assert out.shape correct
assert weight shapes correct
assert forward pass succeeds
assert no disconnected nodes
```

---

# Suggested Validation Helper

```python
def validate_model(gm, input_shape):
    x = torch.randn(input_shape)

    with torch.no_grad():
        y = gm(x)

    return y
```

---

# Debugging Helpers

```python
def print_graph(gm):
    for node in gm.graph.nodes:
        print(
            node.name,
            node.op,
            node.target,
            list(node.users),
            node.all_input_nodes,
        )
```

---

# Recommended Logging

```python
print(f"Shrinking output of {layer_id}")
print(f"Updating consumer {consumer_id}")
print(f"Propagating through add")
```

---

# Performance Notes

This system is graph traversal heavy.

Use:

```python
visited = set()
```

to avoid infinite recursion.

---

# Important Edge Cases

# Shared modules

```python
same Linear used twice
```

Must detect carefully.

---

# Cycles

FX graphs should be DAGs.

Still protect recursion.

---

# Grouped Conv

Initially reject:

```python
groups != 1
```

---

# Add Shape Mismatch

If add branches cannot shrink consistently:

Raise:

```python
RuntimeError
```

---

# Long Term Vision

Eventually this becomes:

```text
graph constraint solving system
```

where:

```text
shape changes
parameter sharing
residual equality constraints
concatenation offsets
```

are all solved automatically.

---

# Suggested Milestones

# Phase 1

Support:

```text
Linear
ReLU
BatchNorm1d
Add
```

---

# Phase 2

Support:

```text
Conv2d
BatchNorm2d
Flatten
Pooling
```

---

# Phase 3

Support:

```text
Concat
Transformer blocks
Attention
```

---

# Final Important Principle

The graph itself usually does NOT need modification.

Most transformations are:

```text
same graph
new module dimensions
new weights
```

FX is mainly:

```text
dependency discovery
```

not necessarily graph surgery.

---

# Final Summary

The system should:

```text
1. Detect output shape change
2. Propagate through graph
3. Update all consumers
4. Enforce add constraints
5. Replace modules safely
6. Validate final model
```

This is the FX equivalent of the original GrowingNN neuron shrinking logic, but generalized into a recursive graph-aware shape propagation engine.