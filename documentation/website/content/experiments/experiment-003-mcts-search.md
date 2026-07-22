# Experiment 003: MCTS architecture search

This example records a complete GrowingNN run. Monte Carlo Tree Search chooses between growth and shrink actions.

## Goal

Compare automatic architecture search with the fixed model and the restricted sequential growth policy.

## Setup

- Dataset: CIFAR-10
- Starting model: same model as Experiment 001
- Search: MCTS with UCB1
- Rollout depth: `3`
- Search time per generation: `120 s`
- Generations: `6`
- Random seeds: `3`

## Results

- Validation accuracy: `81.3%`
- Final validation loss: `0.62`
- Final parameters: `1.39 M`
- Search time: `12 min`
- Selected actions: add layer, add neurons, add residual path, remove neurons, add layer

## Finding

The example search reaches the best accuracy of the three runs. It also ends with fewer parameters than sequential-only growth. Search time is the main extra cost.

## Next step

Run an ablation for each score term. Report mean, standard deviation, and all random seeds instead of one selected run.
