# GrowingNN algorithm

GrowingNN trains a neural network and changes its structure at the same time. Weight optimization and architecture search have separate jobs. This keeps each decision small and easy to inspect.

> [!NOTE] Main idea
> Training improves the parameters inside the current graph. Simulation chooses one legal graph change for the next generation.

## One generation

A run is divided into generations. Each generation has two stages.

1. The training stage uses SGD and backpropagation for a fixed number of epochs.
2. The simulation stage compares architecture actions and selects one action.
3. The selected action changes the traced model.
4. The next generation continues training the updated model.

This cycle lets learned weights survive between architecture changes. It avoids rebuilding the whole network after every decision.

## Architecture search

The model is represented by a directed acyclic graph with `torch.fx`. Layers are nodes. Tensor flow is represented by directed edges. Both sequential and residual connections can be inspected and changed.

Monte Carlo Tree Search explores legal architecture actions. A rollout applies a short sequence of actions and training steps. A scoring function measures the result. UCB1 balances actions with good known scores against actions with little evidence.

The current search has two important rules.

1. Every legal action at the root is tested at least once.
2. Only one first-level action is returned and applied to the real model.

Read [[MCTS]] for the search details and [[Actions]] for the available graph changes.

## Safe graph mutations

An action may add a layer, add neurons, remove a layer, remove neurons, or create a residual path. Each action first checks whether it is legal for the current graph.

New layers should disturb learned behavior as little as possible. Quasi-identity initialization and width propagation help preserve compatible tensor shapes and useful weights.

The traced model invalidates cached graph data after a mutation. Shape and topology analysis can then be computed again from the new graph.

## System flow

```text
Current GraphModule
        |
        v
Train weights with SGD
        |
        v
Generate legal actions
        |
        v
Explore actions with MCTS
        |
        v
Apply one graph mutation
        |
        +----> next generation
```

## Where to continue

Open [[Training loop]] to inspect weight training. Open [[Simulation]] for action selection. Open [[TracedModel]] for the model wrapper. Open [[Torch.fx]] for graph inspection and editing.
