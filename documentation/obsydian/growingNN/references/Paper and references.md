### Main chapter

Szymon Pjura et al., growing neural networks with search over architectures. Published as chapter `10.1007/978-3-031-63749-0_25` in the Springer volume (2024 era proceedings). Same DOI appears in action comparison sections across the vault.

### PyTorch FX

Official FX docs: `https://docs.pytorch.org/docs/stable/fx.html` . Used for `symbolic_trace`, `GraphModule`, `Node`, and passes such as `ShapeProp` in `growingnn/utils/fx/graph_analysis.py` (`LayerShapeAnalyser`).

### `nn.Module.get_submodule`

Docs: `https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule` . Central to `ModuleResolver.get_layer_module` in `growingnn/utils/fx/node_analysis.py`.
