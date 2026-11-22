---
name: function-preserving-actions
description: How to initialize weights when an architecture action (adding a layer, neurons or channels, widening, inserting a branch) changes a growingnn network, so the network keeps what it has learned. Use when writing or changing any action in growingnn/actions, its weight factories (layer_Factory.py, layer_resize.py), or when judging why an action makes the loss jump.
---

# Function-preserving actions

## The idea

An action changes the structure of a trained network. It must not make the network forget what it
learned: right after the action the network should compute (almost) the same function, so the
training history shows no jump in loss. Training then decides how to use the new capacity.

This is a goal for **weight initialization**, not a filter. Do not reject actions by comparing
outputs at generation time; make every action's initialization preserve the output by construction,
and prove it in a unit test.

## Find the operation the new part enters through

Look at where the new weights meet the old computation. There are only a few cases.

| How the new part enters | Example | Initialize so it is neutral |
| --- | --- | --- |
| **Addition** (a new term in a sum) | residual branch `y = x + g(x)`; new channels or neurons, because every consumer sums over its inputs `z = Σ_c V[:, c] · h_c` | **zero** the weights that carry the new term into the sum |
| **Composition / multiplication by a matrix** | new sequential layer `y = L(x)` between two layers | **identity** (eye matrix, delta kernel for convolutions) |
| **Element-wise multiplication** | gate or scale `y = s ⊙ x` | **ones** |
| **Concatenation** | `[x, n]` read by a later layer | **zero** the later layer's columns for `n` |

New channels and new neurons are the addition case, not the composition case: the old channels
stay as they are, and the new ones only add terms to what the next layers compute.

## Rules for new channels or neurons (widening, add neurons)

1. **Keep every old weight exactly as it is**, at the same index. Never rescale or mix old weights
   to make room (the quasi-identity reshaper does this and changes the output).
2. **New incoming weights: normal random initialization** (the layer's default init). The new
   units then carry signal and differ from each other.
3. **New outgoing weights: zero**, in every consumer of the widened output: convolutions,
   linear layers, and the classifier after global pooling (pooling keeps channels independent).
4. **Channel-wise layers in between are independent per channel**: BatchNorm (copy old
   `weight`, `bias`, `running_mean`, `running_var`; new channels `weight=1`, `bias=0`, `mean=0`,
   `var=1`), ReLU, Dropout, pooling. They need no special care beyond keeping old values.
5. **Residual additions**: all inputs of a sum must grow together. Widen every producer of the sum
   the same way (old channels first, new ones after) and zero the outgoing weights after the sum.

**Do not duplicate weights.** Copies preserve the output only with a matching split in the
consumer, start as identical units that need noise to separate, and learn slowly. Zero outgoing
weights preserve the output exactly and leave the new units free.

**Keep gradients flowing.** At least one side of a new unit must be non-zero. With random incoming
and zero outgoing weights, the outgoing weights get a gradient from the first step (it is
proportional to the new unit's activation), so the unit starts learning. If both sides are zero,
the unit never learns.

## Things that break preservation quietly

- **BatchNorm after an identity layer**: an eye convolution followed by a new BatchNorm normalizes
  with batch statistics, so the output changes. Either initialize the BatchNorm to undo it or accept
  and measure the change.
- **Activation after an identity layer**: ReLU after ReLU is identity, but ReLU after a layer that
  can output negatives is not.
- **Training vs evaluation mode**: BatchNorm and Dropout behave differently. Check both.
- **Width tracking**: the shape propagation must see through global pooling and `flatten`, or the
  classifier input is not resized.

## Verify with a test, not a filter

For each action, a unit test that:
1. builds a small network with the structures the action touches (BatchNorm, residual sum,
   classifier after global pooling);
2. gives BatchNorm non-trivial `weight`, `bias` and running statistics, like a trained network;
3. runs a fixed random batch before and after the action, in **evaluation and training mode**;
4. asserts the outputs match (`torch.allclose`, tight tolerance), and that the parameter count grew.

## State of the actions in this repository (2026-11-22)

| Action | Enters by | Current initialization | Preserves output |
| --- | --- | --- | --- |
| Residual conv / linear | addition | zero (`create_zero_conv`) | yes |
| Sequential conv / linear | composition | identity (`create_eye_conv`, `Layer_Type.EYE`) | yes, except with BatchNorm after it |
| Add neurons | addition | quasi-identity rescaling of old weights | **no** (output change about the size of the output) |
| Widen conv | addition | old weights kept, new channels random, zero weights where they are read (`resize_conv_output`) | yes, in both modes |
| Add normalization (`AddNormalization`) | composition | PyTorch default (BatchNorm: weight 1, bias 0, running mean 0, variance 1; LayerNorm: weight 1, bias 0) | BatchNorm: in evaluation mode only; LayerNorm: no |

**Add normalization is a deliberate exception.** The user chose plain default initialization so the
new BatchNorm stays a normal BatchNorm (no gate, no calibration batch), and accepted the change in
training mode, where batch statistics normalize from the first step. Do not "fix" it with a gate
or calibrated weight and bias without asking.

### To do

- **Add neurons:** replace the quasi-identity rescaling with random new rows and zero new columns,
  as in widen conv.
- **Check every other action's initialization** with the test recipe above: sequential conv/linear
  with BatchNorm after the identity layer, residual layers, dropout, delete layer, delete neurons.
  For each, confirm the output is preserved and that the new weights get a non-zero gradient (one
  side must be non-zero, or the new part never learns).

Widen conv is the reference implementation: `can_resize_conv_output` offers every convolution whose
new width is absorbed by convolutions or linear layers, and `_keep_old_weights_and_zero_new_inputs`
initializes every layer the propagation grew. On the Experiment 015 Phase 12 network all 20
widenings left the output exactly unchanged; the earlier quasi-identity rescaling changed it by up
to 1.07 (mean output 0.83).
