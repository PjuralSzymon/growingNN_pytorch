# Experiment 006: Neuron-resize action ratio pairs

This grid is inconclusive. It cannot decide whether AddNeurons / DelNeurons should stay on. We need a rerun after we know more about simulation set size, simulation time, and the simulation algorithm.

Script: `experiments/train_mnist_exp006_neuron_resize_actions.py`

Package: `sequential_halving_beam`, `120 s` budget, `big` MNIST starter, four groups (`none`, `add11_del01`, `add15_del05`, `add20_del09`), three seeds. `12` / `12` cells finished.

## Simulation bug found while this experiment ran

`sequential_halving_beam_alg.get_action` scored every root action once. That first pass used up the `120 s` budget. Sequential Halving then ran zero times. The keep-set was the first three arms in `generate_all_actions` order, not the top scores. Residual conv is listed first, so it always won.

That was fixed during this work: sort living arms by mean before `BEAM_WIDTH`, and run at least `SIMULATION_MIN_ALGORITHM_ITERATION_RUNS = 3` halving rounds even after time is gone. The published numbers here still mix a broken search with a later patched search. They are not a clean test of neuron-resize.

## What we can see

![Final accuracy by neuron-resize group](/assets/experiments/006-final-accuracy-by-group.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy by neuron-resize group. Gray markers are individual seeds.

![Parameter growth by neuron-resize group](/assets/experiments/006-param-growth-by-group.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by group. Gray markers are individual final counts.

The `add20_del09` bar grows more than the control. That is not AddNeurons with factor `2.0`. That action never ran (`0` live executions). The extra parameters come from residual and sequential layer inserts. The simulation did not grade width-doubling by executing it.

![Chosen simulation actions by neuron-resize group](/assets/experiments/006-simulation-chosen-actions-by-group.png)

> [!CAPTION] Figure 3. Count of winning simulation actions by group.

DeleteNeurons never ran (`del01`, `del05`, and `del09` are all `0` live actions). `del01` never even entered the pool. AddNeurons ran `3` times in total (`add11` once, `add15` twice, `add20` never). That is too few events to judge the flags.

## Conclusion

Do not use this experiment to turn neuron-resize on or off.

We need more information about simulation set size, simulation time, and the simulation algorithm before this question can be graded. Then rerun the grid with a working search and a budget that can actually try AddNeurons / DelNeurons, including factor `2.0`.
