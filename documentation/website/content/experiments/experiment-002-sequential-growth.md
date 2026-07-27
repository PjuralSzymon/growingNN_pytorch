# Experiment 002: Sequential layer growth

This example measures one simple architecture mutation. It follows Experiment 001 and changes only the growth setting.

## Goal

Test whether adding sequential convolution layers after learning slows down improves validation accuracy.

## Setup

- Dataset: CIFAR-10
- Starting model: same model as Experiment 001
- Trigger: validation score does not improve for `5` epochs
- Enabled action: sequential convolution insertion
- Maximum actions: `3`
- Total epochs: `50`

## Results

- Validation accuracy: `80.1%`
- Final validation loss: `0.66`
- Final parameters: `1.47 M`
- Applied actions: `2`

## Finding

Two inserted layers improve the example accuracy by `1.7` percentage points. The parameter count grows by `18.5%`. The first action gives most of the improvement.

## Next step

Compare this restricted policy with Monte Carlo Tree Search over several action types.
