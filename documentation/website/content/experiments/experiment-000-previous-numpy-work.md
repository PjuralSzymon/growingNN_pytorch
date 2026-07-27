# Experiment 000: Previous NumPy work

This page explains the experiments completed before the current PyTorch implementation. They belong to the original GrowingNN project. That version implemented neural networks, gradient descent, graph changes, and simulation directly with NumPy.

The results on this page provide historical context. They are not direct benchmarks of the current PyTorch code.

## Goal

Record what was tested in the previous GrowingNN work. Separate those results from new experiments performed with this repository.

## Previous implementation

The original system was based on NumPy. It did not use PyTorch, `torch.nn.Module`, automatic differentiation, or `torch.fx`.

The implementation contained its own:

- dense and convolution operations
- forward and backward propagation
- SGD training
- directed graph representation
- architecture growth and shrinking actions
- Monte Carlo Tree Search
- quasi-identity shape adaptation

This gave full control over each architecture change. It also made larger models and standard PyTorch comparisons more difficult.

## Previous experiments

The original studies used MNIST and Fashion-MNIST classification. They examined whether:

1. architecture growth could improve a very small starting network
2. MCTS could select better changes than random or greedy selection
3. learned knowledge could survive layer insertion and removal
4. the simulation trigger and progressive learning rate could reduce instability after a change

These experiments used the NumPy implementation and its custom training loop. Their timing, memory use, and exact metrics must not be compared directly with the current PyTorch version.

## References

S. Świderski and A. Jastrzębska, "Dynamic Growing and Shrinking of Neural Networks with Monte Carlo Tree Search", Computational Science — ICCS 2024, Lecture Notes in Computer Science, volume 14832, pages 362–377. DOI: [10.1007/978-3-031-63749-0_25](https://doi.org/10.1007/978-3-031-63749-0_25).

The extended study is "Data classification with dynamically growing and shrinking neural networks", Journal of Computational Science, 2025. DOI: [10.1016/j.jocs.2025.102660](https://doi.org/10.1016/j.jocs.2025.102660).

The previous source code is available in the [original GrowingNN repository](https://github.com/PjuralSzymon/growingnn).

## Known limitations

The previous reports do not validate the current PyTorch implementation. The model representation, automatic differentiation, layer execution, shape analysis, and mutation code have changed.

Some old figures and result files are not stored in this repository. Values from the papers should be treated as published historical results.

## Next step

Experiment 001 starts the new sequence. It should establish a reproducible PyTorch baseline before architecture actions are enabled.
