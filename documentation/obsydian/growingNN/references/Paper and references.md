### Main chapter

Szymon Pjura et al., growing neural networks with search over architectures. Published as chapter `10.1007/978-3-031-63749-0_25` in the Springer volume (2024 era proceedings). Same DOI appears in [[Layer Analyser]], [[Residual Conv Action]], and [[Residual Linear Actions]].

### PyTorch FX

Official FX docs: `https://docs.pytorch.org/docs/stable/fx.html` . Used for `symbolic_trace`, `GraphModule`, `Node`, and passes such as `ShapeProp` in [[Layer Analyser]].

### This repo vs older growingNN package

Sequential conv→linear bridges moved from [[Sequential Conv Action]] to [[Sequentail Linear Actions]] (plain `nn.Linear` on the existing flatten path). See those pages under `documentation/obsydian/growingNN/Actions/`.

### `nn.Module.get_submodule`

Docs: `https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule` . Central to [[Dotted Module Names in torch.fx]] and `get_layer_module` in [[Model Analyser]].
