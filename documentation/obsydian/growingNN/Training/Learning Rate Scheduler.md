When GrowingNN changes the graph, weights need a short recovery. When nothing changes, training should follow a normal global learning rate curve. [[Composed Learning Rate Scheduler]] multiplies those two signals.

`effective_lr = max(MIN_LEARNING_RATE, base_lr(global_epoch) * recovery_factor)`

`MIN_LEARNING_RATE` is `0.001` in `growingnn/training/lr_scheduler.py`.

## Standalone GrowingNN schedules

Pass a `LearningRateScheduler` on `RunningConfig.lr_scheduler`. Modes live in `ScheduleMode`:

- `CONSTANT` — fixed absolute LR (`alpha`)
- `PROGRESSIVE` / `PROGRESSIVE_PARABOLIC` — rise then fall inside one generation window
- `WARMUP_COSINE` / `WARMUP_LOGISTIC` / `WARMUP_EXPONENTIAL` — action-aware warmup via `iterations_since_change`

`train_generations` calls `structure_changed()` only after an architecture action runs. That resets warmup. See [[Training loop]] and Experiment 000.

Example:

```python
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode

config.lr_scheduler = LearningRateScheduler(
    ScheduleMode.WARMUP_LOGISTIC,
    alpha=0.01,
    warmup_iterations=10,
    k=10.0,
)
```

## Why composition

A cosine or step schedule over the whole run is the usual PyTorch mental model. GrowingNN still needs a low LR right after a mutation. Composition keeps the base curve and only multiplies a recovery factor after `structure_changed()`.

Details and copy-paste setup: [[Composed Learning Rate Scheduler]].

## Known limitations

`ReduceLROnPlateau` and `OneCycleLR` are not wrapped yet. They need different trainer hooks than epoch-index base adapters.
