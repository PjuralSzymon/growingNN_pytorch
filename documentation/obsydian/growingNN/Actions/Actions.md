Actions are the building block of this algorithm those are defining how the model can change on the given simulation and how it will change. Each action inheriths from base Action class (`growingnn/actions/action.py`) which forces the action to implement 2 methods: 
- `generate_all_actions` which is static and gets the model (`model: nn.Module | fx.GraphModul`)  and returns List of Actions, each action in that list is a copy of the current class with different parameters for example `generate_all_actions` implemented in the [[Sequential Linear Actions]] will return a list of [[Sequential Linear Actions]] but each of those classes will have parameters that will define different sequential linear actions to be added a given moment. Changes generated for a given model won't work for other models.
- Execute is a function that execute changes on a model it was generated. 

## Action types
Currently in our algorithm we have 4 action types used to grow the network and 2 action types design to shrink the network (also we have 1 more shrinking action in the design process).
### Grow oriented actions:
- [[Sequential Linear Actions]]
- [[Sequential Conv Action]]
- [[Residual Linear Actions]]
- [[Residual Conv Action]]
### Shrink oriented actions:
- [[Del Layer Action]]
- [[Del neurons Action]]



