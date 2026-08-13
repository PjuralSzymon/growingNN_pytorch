# Simulation scheduler stagnation modes plan

## Goal

`SimulationScheduler` decides whether [[Simulation]] runs after one generation.

Keep `PROGRESS_CHECK`. With `stagnation_window=1`, it compares the final accuracy of the current generation with the final accuracy of the previous generation. Larger windows compare the current value with the best final value in the recent generation window.

Add two more modes that inspect the final validation accuracies collected across generations:

1. `SLOPE_ESTIMATION`;
2. `MEAN_STANDARD_DEVIATION_STAGNATION`.

The training stopper keeps its current role. It checks target accuracy and parameter count.

This design does not stop a generation early. For example, all 30 epochs run before the scheduler analyses them. It only improves the decision made at the generation boundary.

## Scheduler classes

Place scheduler classes in `growingnn/simulation/simulation_schedulers/`.

Use one abstract base:

```python
class SimulationScheduler(ABC):
    @abstractmethod
    def can_simulate(...) -> bool:
        ...
```

Add one concrete class per mode:

- `AlwaysSimulationScheduler`;
- `ProgressCheckSimulationScheduler`;
- `NeverSimulationScheduler`;
- `SlopeEstimationSimulationScheduler`;
- `MeanStandardDeviationStagnationSimulationScheduler`.

Each class exposes its `SchedulerMode` through the `mode` class attribute. `RunningConfig` accepts the abstract `SimulationScheduler` type.

`MEAN_STANDARD_DEVIATION_STAGNATION` states the calculation and its purpose. The method is a heuristic. It does not calculate a formal confidence interval.

## Parameters

Add:

```python
slope_epsilon: float = 1e-4
standard_deviation_multiplier: float = 1.5
```

Rules:

- `slope_epsilon >= 0`;
- `standard_deviation_multiplier > 0`;
- accuracy uses the range `0.0` to `1.0`;
- non-finite values return `False`.

The parameters must be calibrated from real runs.

## Slope estimation

Use every value in `generation_val_acc`. Each value is the final validation accuracy of one generation.

For points `(x_i, y_i)`:

- `x_i = 0, 1, ..., W - 1`;
- `y_i` is final generation validation accuracy.

Calculate:

```text
a = sum((x_i - mean(x)) * (y_i - mean(y)))
    / sum((x_i - mean(x))^2)
```

The requested flatness rule is:

```text
abs(a) <= slope_epsilon
```

A negative slope with a large magnitude does not satisfy this rule. If declining accuracy should also start simulation, use:

```text
a <= slope_epsilon
```

The first implementation should use the requested absolute rule. The decision can be changed after observing real histories.

Add a pure helper:

```python
def _least_squares_slope(values: Sequence[float]) -> float:
```

It can use Python arithmetic. NumPy is not needed.

## Mean and standard-deviation stagnation

Use the same final validation-accuracy history across generations.

Calculate:

```text
mu = mean(values)
sigma = sample standard deviation(values)
distance = max(values) - mu
```

The base rule is:

```text
distance <= standard_deviation_multiplier * sigma
```

For:

```text
0.910, 0.912, 0.911, 0.913, 0.912, 0.911
```

the values are approximately:

```text
mu = 0.9115
sigma = 0.001049
distance = 0.0015
distance / sigma = 1.43
```

The example triggers when `standard_deviation_multiplier >= 1.43`.

The base rule can also accept a rising line. Add a slope guard:

```text
distance <= standard_deviation_multiplier * sigma
and
abs(least_squares_slope(values)) <= slope_epsilon
```

When `sigma == 0`, all values are equal. The mode returns `True`.

Use `statistics.stdev`. Do not add a dependency.

## Scheduler API

All concrete classes implement the same boundary-level method:

```python
def can_simulate(
    self,
    generation: int,
    generation_val_acc: Sequence[float],
    quiet: bool = False,
) -> bool:
```

Class behavior:

- `NEVER` returns `False`;
- `ALWAYS` returns `True`;
- `PROGRESS_CHECK` keeps comparing final accuracies between generations;
- `SLOPE_ESTIMATION` checks the slope;
- `MEAN_STANDARD_DEVIATION_STAGNATION` checks the mean, deviation, and slope;
- the new modes return `False` when `generation_val_acc` has fewer than `2` values.

All metric-based schedulers use `generation_val_acc`. The scheduler interface remains compatible with the unchanged [[Training loop]].

## Training integration

Do not change `gradient_descent`.

`train_generations` already receives the full history:

```python
traced.gm, history = gradient_descent(...)
```

Keep the existing scheduler call unchanged:

```python
if config.simulation_scheduler.can_simulate(
    generation,
    generation_val_acc,
    quiet=config.quiet,
):
    action, _, _ = config.simulation_alg.get_action(...)
```

The existing stopper check remains before this call. A terminal stopper result prevents simulation.

`train_generations` remains the owner of orchestration described in [[Training loop]]. The scheduler only compares metrics. The simulation algorithm still generates an action, and `action.execute` still applies it to the live model.

## Verification

Add small deterministic checks for:

1. rising, flat, falling, and noisy slope histories;
2. fewer than `2` generation values;
3. constant values with `sigma == 0`;
4. a rising line rejected by `MEAN_STANDARD_DEVIATION_STAGNATION`;
5. invalid parameter values;
6. preserved `ALWAYS`, `PROGRESS_CHECK`, and `NEVER` behavior;
7. current versus previous final accuracy in `PROGRESS_CHECK`;
8. unchanged scheduler call in `train_generations`;
9. stopper priority over simulation.

## Comparison with the original growingNN paper

The original growingNN method alternates weight training and architecture search. These modes preserve that sequence. They only replace the fixed simulation decision with an analysis of weight-training progress.

## Known limitations

1. The complete generation runs before the decision. No epochs are saved.
2. Training accuracy may keep rising during overfitting.
3. Accuracy is discrete and noisy.
4. The thresholds depend on dataset size and generation length.
5. The mean and standard-deviation rule is not a formal confidence interval.
