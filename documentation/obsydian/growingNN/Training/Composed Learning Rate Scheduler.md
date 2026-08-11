`ComposedLearningRateScheduler` in `growingnn/training/lr_scheduler.py` multiplies a global base curve by GrowingNN recovery. It is the scheduler you pass to `RunningConfig.lr_scheduler` when you want both. The [[Training loop]] still calls the same methods: `alpha_scheduler`, `structure_changed`, `reset`.

## What it does

```text
effective_lr = max(MIN_LEARNING_RATE, base_lr(global_epoch) * recovery_factor)
```

Base schedule: absolute LR over the full run (`global_epoch`). Continues across generations when no action runs.

Recovery factor: after `structure_changed()`, starts near `0` and warms to `1`. When idle, factor stays `1`, so training is just the base schedule. Used by [[Learning Rate Scheduler]] recovery modes with `alpha=1.0` as the peak factor, not an absolute LR.

Until the first action, `mark_warmup_schedule_as_fully_complete` primes recovery so early epochs follow the base curve only.

## Base adapters

Pure functions `lr_at(global_epoch, total_epochs)`. No optimizer required.

1. `CosineAnnealingBase(T_max, eta_min=0, initial_lr=0.01)` — cosine decay like `torch.optim.lr_scheduler.CosineAnnealingLR`
2. `StepLRBase(step_size, gamma=0.1, initial_lr=0.01)` — drop by `gamma` every `step_size` epochs
3. `ExponentialLRBase(gamma, initial_lr=0.01)` — `initial_lr * gamma^epoch`
4. `LinearDecayBase(T_max, eta_min=0, initial_lr=0.01)` — linear decay from start to floor
5. `ConstantBase(lr)` — fixed base

`total_epochs` on the composed object is usually `generations * epochs` from `RunningConfig`.

Factory helpers in the same module: `build_base_learning_rate_schedule` and `build_composed_learning_rate_scheduler`. Experiment 004 compares these bases on MNIST.

## Copy-paste example

```python
from growingnn.training.lr_scheduler import (
    LearningRateScheduler,
    ScheduleMode,
    ComposedLearningRateScheduler,
    CosineAnnealingBase,
    build_composed_learning_rate_scheduler,
)

total_epochs = generations * epochs_per_generation

config.lr_scheduler = ComposedLearningRateScheduler(
    base=CosineAnnealingBase(T_max=total_epochs, eta_min=1e-4, initial_lr=0.01),
    recovery=LearningRateScheduler(
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

`run_quiet_simulation_scoring_gradient_descent` in `growingnn/simulation/score_functions/simulation_training.py` wraps quiet `gradient_descent` with `freeze_base_learning_rate_progress_if_supported`. That freezes `global_epoch` on the shared composed scheduler so short simulation rollouts do not burn the global base progress. `structure_changed` still only resets recovery.

## Comparison with the original growingNN paper

The paper focused on action-aware LR after mutations. Composition keeps that idea and adds a standard global decay on top, which the paper did not separate as base × factor.

## Known limitations

Recovery must use `alpha=1.0`. Metric-driven and budget-driven torch schedulers (`ReduceLROnPlateau`, `OneCycleLR`) are out of scope for this adapter set. `gradient_descent` peeks the first LR via `peek_learning_rate_without_advancing` so composed global epoch advances once per real training epoch.
