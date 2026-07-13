Actions are the building block of this algorithm those are defining how the model can change on the given simulation and how it will change. Each action inheriths from base Action class (`growingnn/actions/action.py`) which forces the action to implement 2 methods: 
- `generate_all_actions` is a static method that takes [[TracedModel]] and returns a list of actions. Each entry is a copy of the same class with different parameters. For example, `generate_all_actions` in [[Sequential Linear Actions]] returns a list of [[Sequential Linear Actions]] instances, each defining a different sequential layer to add at that moment. Actions generated for one model do not apply to another.
- `execute(traced)` calls `_execute(traced)` then `traced.invalidate()` so cached shape and topology data match the graph after the mutation.

## Action types

### Grow oriented actions

- [[Sequential Linear Actions]]
- [[Sequential Conv Action]]
- [[Residual Linear Actions]]
- [[Residual Conv Action]]
- [[Add neurons Action]]
- [[Add Seq Dropout Action]]

### Shrink oriented actions

- [[Del Layer Action]]
- [[Del neurons Action]]

Shared helpers live in `growingnn/actions/utils/` (`seq_insertion.py`, `layer_resize.py`, `layer_Factory.py`) and `growingnn/actions/registry.py` combines enabled generators into one move list.
