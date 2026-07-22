# Experiment 001: Fixed architecture baseline

This short template records the reference result before architecture growth is enabled. Replace the example values with measured values from your run.

## Goal

Measure the accuracy, loss, training time, and parameter count of a fixed model. Later experiments can use this run as a fair baseline.

## Setup

- Dataset: CIFAR-10
- Model: small fixed convolutional network
- Optimizer: SGD
- Learning rate: `0.01`
- Epochs: `50`
- Random seeds: `3`
- Architecture actions: disabled

## Results

- Validation accuracy: `78.4%`
- Final validation loss: `0.71`
- Parameters: `1.24 M`
- Mean training time: `18 min`

## Finding

The fixed model learns a stable reference solution. Its validation curve starts to flatten near epoch 35. This is the point where architecture growth may be useful.

## Next step

Enable one sequential growth action. Keep the dataset, optimizer, seed set, and training budget unchanged.
