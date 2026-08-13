`ComposedLearningRateScheduler` in `growingnn/training/lr_scheduler_global.py` multiplies a global curve by GrowingNN recovery. Pass it as `RunningConfig.lr_scheduler` when you want both. The [[Training loop]] still calls `alpha_scheduler`, `structure_changed`, and `reset`.

## What it does

```text
effective_lr = max(MIN_LEARNING_RATE, global_lr(global_epoch) * recovery_factor)
```

Global schedule: absolute LR over the full run (`global_epoch`). Continues across generations when no action runs.

Recovery factor: after `structure_changed()`, starts near `0` and warms to `1`. When idle, factor stays `1`, so training is just the global schedule. Recovery comes from [[Learning Rate Scheduler]] modes with `alpha=1.0` as the peak factor, not an absolute LR.

Until the first action, `mark_warmup_schedule_as_fully_complete` primes recovery so early epochs follow the global curve only.

## Global adapters

Pure functions `lr_at(global_epoch, total_epochs)`. No optimizer required. Defined in `lr_scheduler_global.py`.

1. `CosineAnnealingLearningRate(t_max, eta_min=0, initial_lr=0.01)` — cosine decay like `torch.optim.lr_scheduler.CosineAnnealingLR`
2. `StepLearningRate(step_size, gamma=0.1, initial_lr=0.01)` — drop by `gamma` every `step_size` epochs
3. `ExponentialLearningRate(gamma, initial_lr=0.01)` — `initial_lr * gamma^epoch`
4. `LinearDecayLearningRate(t_max, eta_min=0, initial_lr=0.01)` — linear decay from start to floor
5. `ConstantLearningRate(lr)` — fixed global rate

`total_epochs` on the composed object is usually `generations * epochs` from `RunningConfig`.

Factory helpers in the same module: `build_global_learning_rate_schedule` and `build_composed_learning_rate_scheduler`. Experiment 004 compares these on MNIST.

## Copy-paste example

```python
from growingnn.training.lr_scheduler_action import (
    ActionLearningRateScheduler,
    ScheduleMode,
)
from growingnn.training.lr_scheduler_global import (
    ComposedLearningRateScheduler,
    CosineAnnealingLearningRate,
    build_composed_learning_rate_scheduler,
)

total_epochs = generations * epochs_per_generation

config.lr_scheduler = ComposedLearningRateScheduler(
    global_schedule=CosineAnnealingLearningRate(
        t_max=total_epochs, eta_min=1e-4, initial_lr=0.01
    ),
    recovery=ActionLearningRateScheduler(
        ScheduleMode.WARMUP_LOGISTIC,
        alpha=1.0,
        warmup_iterations=10,
        k=10.0,
    ),
    total_epochs=total_epochs,
    initial_lr=0.01,
)

# Or:
config.lr_scheduler = build_composed_learning_rate_scheduler(
    "cosine",
    total_epochs=total_epochs,
    initial_lr=0.01,
)
```

## Simulation scoring

`run_simulation_scoring_gradient_descent` in `growingnn/simulation/score_functions/simulation_training.py` wraps `gradient_descent` with `freeze_global_learning_rate_progress_if_supported`. That freezes `global_epoch` on the shared composed scheduler so short simulation rollouts do not burn global progress. `structure_changed` still only resets recovery.

## Comparison with the original growingNN paper

The paper focused on action-aware LR after mutations. Composition keeps that idea and adds a standard global decay on top, which the paper did not separate as global × factor.

## Known limitations

Recovery must use `alpha=1.0`. Metric-driven and budget-driven torch schedulers (`ReduceLROnPlateau`, `OneCycleLR`) are out of scope for this adapter set. Epoch 0 overwrites the optimizer LR from the scheduler; the SGD constructor LR is only a throwaway default.
