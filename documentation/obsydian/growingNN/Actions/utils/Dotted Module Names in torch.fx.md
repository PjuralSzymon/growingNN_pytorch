
This note explains how `torch.fx` names submodules in an `fx.GraphModule`, how to read those names, why `getattr` returns `None` for them, and which API the framework should use.

It is used by [[Model Analyser]] and [[Model Transformer]] when they walk the graph and need the actual `nn.Module` object behind a `call_module` node.

---

### What a node target looks like

A node target in an `fx.GraphModule` is a string that names a submodule.
- Flat models give a single identifier, like `"l1"` or `"stem"`.
- Nested models give a dotted path, like `"layer1.0.conv1"`.
The dotted form is how `torch.fx` encodes a path through submodules.

### How `torch.fx` builds the name

`torch.fx.symbolic_trace` walks the module tree using `nn.Module.named_modules()`. Reference: https://docs.pytorch.org/docs/stable/fx.html
The qualified name is the chain of lookups from the root model, joined with dots:
- An attribute step uses the attribute name, e.g. `conv1`.
- An `nn.Sequential` index step uses the integer position as a string, e.g. `0`, `1`.
- Each step is separated by a single dot.

### How to read `layer1.0.conv1`

Take `torchvision.models.resnet18`. Its forward path includes:
- `self.layer1` is an `nn.Sequential` of two `BasicBlock` instances.
- `self.layer1[0]` is the first block.
- `self.layer1[0].conv1` is the first `nn.Conv2d(64, 64, kernel_size=3, padding=1)` in that block.

So the dotted name `layer1.0.conv1` is read left to right as:
1. attribute `layer1`
2. index `0`
3. attribute `conv1`

### Why `getattr(gm, "layer1.0.conv1")` returns `None`

Python's `getattr` does one attribute lookup. It treats the dot as part of the attribute name, not as a path separator. There is no attribute on `gm` literally named `layer1.0.conv1`, so `getattr(gm, "layer1.0.conv1", None)` returns `None`.

This is why the early check `getattr(gm, str(node.target), None) is None` in `_is_editable_module` (`growingnn/actions/utils/model_analyser.py`, line 50) incorrectly rejected every nested submodule in ResNet-18, even though those submodules are real `nn.Conv2d` instances and `nn.Conv2d` is in `EDITABLE_MODULES` (`growingnn/core/config.py`, line 15).

### The correct API

Use `nn.Module.get_submodule(qualified_name)`. It splits the string on dots and walks the module tree step by step, using `getattr` for attribute steps and `__getitem__` for indexed containers. It raises `AttributeError` if any step is missing. Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule
- `gm.get_submodule("l1")` works for flat names.
- `gm.get_submodule("layer1.0.conv1")` works for nested names.

### What `get_submodule` returns and how to iterate

What `get_submodule` returns
It returns the exact `nn.Module` instance stored at that path. Not a copy. Changing its attributes changes the live model. For `gm.get_submodule("layer1.0.conv1")` on `torchvision.models.resnet18` the return type is `nn.Conv2d` and you can read `.weight`, `.bias`, `.in_channels`, `.out_channels`, `.kernel_size`, `.stride`, `.padding`, `.groups`.

Iterating over each submodule separately
`nn.Module` provides four iteration helpers. They all walk the same tree but differ in depth and in whether names are returned. Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
1. `gm.named_modules()` — yields `(qualified_name, module)` for every submodule, including `gm` itself with the empty name `""`. Recursive. Use this when you want the dotted name and the module together.
2. `gm.modules()` — same recursive walk as `named_modules()` but yields only the module objects, no names.
3. `gm.named_children()` — yields `(name, module)` for direct children only, one level deep. Names are single identifiers, never dotted.
4. `gm.children()` — same as `named_children()` but yields only the module objects.


Behavior on missing or empty names
- `gm.get_submodule("does.not.exist")` raises `AttributeError`. Wrap in `try`/`except AttributeError` if the name comes from outside data.
- `gm.get_submodule("")` returns `gm` itself, because the empty name is the root in `named_modules()`.

### Symptom in the log

With `LOG_LEVEL = "DEBUG"` and `LOG_TO_FILE = True` (`growingnn/core/config.py`, lines 19 and 20), `tests/regression/growingnn.log` shows many lines like:
- `layer_module: None is not an editable module` (from line 52 of `model_analyser.py`)
- `pair: conv1 -> layer1.0.conv1 not added becouse: editable: False, hidden: True` (from line 109 of `model_analyser.py`)

The pair was skipped because the editable check returned `False`. The hidden check correctly returned `True`, so the node really exists in the graph.

### Known limitations

- The bug only shows up on models whose submodules have dotted names. Flat factories in `tests/model_factory.py` (e.g. `simple_chain_3`, `complex_residual_many_widths`) have no dots and are not affected.
- After switching to `get_submodule`, the analyser will start producing edges for every nested editable module. Downstream actions in [[Sequentail Linear Actions]], [[Residual Linear Actions]], [[Sequential Conv Action]], [[Residual Conv Action]], and [[Del Layer Action]] that assume short names may need to be reviewed.
