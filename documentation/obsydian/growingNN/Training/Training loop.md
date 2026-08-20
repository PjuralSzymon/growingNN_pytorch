The training loop is standard SGD, like in most PyTorch projects. The growingNN-specific orchestration lives in `train_generations` in `growingnn/training/trainer.py`. It is described at a high level in [[General]].

When you set the number of training epochs, remember two things. First, that count applies to one generation only, not to the full run. Second, the same loop runs again in every generation. Learning rate shape comes from [[Learning Rate Scheduler]] (standalone or [[Composed Learning Rate Scheduler]]).

---

## `train_generations` — one full run

Entry point used by experiment drivers (for example `experiments/train_cifar10.py`). Wires training, simulation, and architecture mutation.

Idea:

1. prepare simulation loaders (`RunningConfig.simulation_set.generate`, see [[Simulation Set]])
2. for each generation index until `config.generations`:
   2.1 run `gradient_descent` for `config.epochs` on the live model ([[Learning Rate Scheduler]], optional stopper)
   2.2 record metrics and parameter count (`GraphStructureQuery.get_amount_of_parameters`)
   2.3 if early stopper fires then exit the whole run
   2.4 if `simulation_scheduler.can_simulate` is true for this generation then:
       2.4.1 ask [[Simulation]] for one action (`config.simulation_alg.get_action` on a deep copy of the live [[TracedModel]])
       2.4.2 if an action was returned then call `action.execute(traced)` on the live wrapper (invalidates cached analysis on the graph)
       2.4.3 optional experiment board saves FX graphs after the mutation
3. clear quasi-identity cache (`clear_reshepers_cache`) after all generations

The simulation copy and the live model diverge on purpose. Search explores on the copy; only the chosen move mutates the model that continues training next generation.

---

## Inside `gradient_descent`

Per-epoch train and validation pass. Same as a normal PyTorch loop except it reports to `ExperimentBoard` when configured. Not architecture-aware beyond using whatever graph the model already has.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 alternates weight learning and architecture search. R5 maps that to generations: SGD block, then optional simulation picking one architecture move from `registry.py`.

---

## Known limitations

1. Only one action executes per generation even if rollouts explored multi-step paths.

2. Simulation uses a small stratified subset, not the full dataset.

3. `train_generations` does not re-trace the model; actions must keep the `fx.GraphModule` valid after `execute`.
