The GrowingNN algorithm is used to dynamically change the structure of the model during training. To achieve that, the algorithm is divided into generations. Each generation has two stages:
1. Training stage, which is a typical SGD [[Training loop]] that updates the weights in the model over a fixed number of epochs
2. Simulation stage, which uses the [[MCTS]] algorithm to find the best action; what counts as best is described by the [[Scoring function]]!


![[grafy.png]]