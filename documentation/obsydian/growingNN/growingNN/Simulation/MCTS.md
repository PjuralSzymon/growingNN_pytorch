Monte Carlo Tree Search Algorithm we are using our modification of that algorithm, there were 2 main modification
1. The end of the "game" in our case in considered by checking N actions 
2. Each simulation is time limited, but always must analyze all actions at the first depth which means the simulation always will grade all actions that are possible to choose from current model and will analyze more actions if possible. 

## Algorithm
The algorithm of the simulation works as follows: 
1. For a given state of the model in the given generation 
	1. We are generating all possible actions 
	2. Each of those action is graded by [[Scoring function]] 
	3. What action is selected to be graded is chosen based on MCTS and UCB1 formula
2. Algorithm repeats according to the MCTS 
3. Algorithm returns 1 best action in the first level of the tree. 


TODO: To be filled out... 
