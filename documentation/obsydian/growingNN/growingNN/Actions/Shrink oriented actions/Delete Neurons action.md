This page is about `growingnn/actions/delete_neurons.py` and the class `DelNeurons`.

### Current state

The class subclasses [[Base action and Layer Type]] `Action`. `execute` is `pass` at lines 14 to 15. `generate_all_actions` builds an empty list and has no return or append logic completed at lines 20 to 22. It is a stub for future neuron-level shrink moves.

Imports at the top reference `get_all_hidden_modules`, `module_sequential_pairs`, `unique_call_module_name`, `add_new_residual_layer`, `add_new_seq_layer`, `delete_layer` for planned use.

### Generating actions

Not implemented yet.

### Executing actions

Not implemented yet.

### Comparison with the original growingNN paper

The paper discusses architecture search at several granularities. Neuron deletion would be a finer move than whole-layer delete in [[Del Layer Action]].

### Known limitations

The file duplicates `from .action import Action, Layer_Type` twice (lines 5 and 11). The action never appears in [[ResNet18 regression script]] flags.

### Related

[[Del Layer Action]], [[Model Analyser]], [[Part 1]] entry 09.05.2026, [[Index]].
