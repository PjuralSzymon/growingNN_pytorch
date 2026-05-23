The GrowingNN algorithm is used to dynamically change the structure of the model during training. To achive that algrotihm is divided into generation. Each generation is divided into 2 stages: 
1. Training stage which is typical SGD [[Training loop]] that updates the weights in the model and consists of some amount of epochs
2. Simulation stage which is using [[MCTS]] algorithm to find the best action, what is best is described by [[Scoring function]]
