---
name: lr-scheduler-research
description: Explains and safeguards GrowingNN learning-rate schedules, simulation schedulers, architecture-action resets, slope decisions, and the current MNIST scheduler evidence. Use when changing or analyzing training, learning rates, stagnation detection, MCTS timing, scheduler experiments, or their documentation. Never create unit tests for experiments or for other tests.
---

# Learning-rate and simulation scheduler research

Use this skill for work that touches:

- `growingnn/training/lr_scheduler.py`
- `growingnn/training/trainer.py`
- `growingnn/simulation/simulation_schedulers/`
- scheduler experiment scripts and output
- MCTS timing, stagnation, or architecture-action recovery

Read [CURRENT_STATE.md](CURRENT_STATE.md) before making claims or changes.

## Source order

Use current evidence in this order:

1. product code for actual behavior
2. raw `board/main.json` and `board/metrics/training.json` files
3. experiment scripts for grid configuration
4. the experiment report for the interpreted result

Do not treat the report as more authoritative than newer code or output. If they differ, state the difference and update the report when requested.

## Required reasoning

Keep these systems separate:

1. The learning-rate scheduler controls optimizer step size.
2. The simulation scheduler decides whether MCTS may run.
3. MCTS selects an architecture action.
4. An executed action calls `structure_changed()` and resets action-aware warmup.

Never call a slope threshold an LR threshold.

Never attribute an accuracy change to LR alone when mutation and LR reset occur together. The current experiment confounds them.

Distinguish:

- no-action generation boundaries
- action plus LR-reset boundaries
- post-action warmup
- stable full-rate training
- first actions and later actions
- simulated candidate score and realized post-action gain

## Change checklist

Before editing scheduler behavior:

1. Trace the call order through `train_generations()` and `gradient_descent()`.
2. Check whether scheduler state is copied or advanced inside simulation.
3. Check the exact epoch attached to an action and to its next metric.
4. Preserve the `MIN_LEARNING_RATE` floor unless the task changes it explicitly.
5. Add one small deterministic unit test for every changed product function under `growingnn/`.
6. Use Arrange, Act, and Assert labels in each product unit test.
7. Re-run the focused scheduler tests.
8. Do not create any tests for experiments. No unit tests should be created for an experiment, and no unit tests should be created for other regression, CI, or integration tests. Everything test-related or experiment-related should not have a separate unit test.

For experiments, record:

- all RNG seeds and deterministic settings
- starting weight hash and parameter count
- LR mode, current LR, and warmup counter
- signed slope angle and threshold
- action type, epoch, and generation
- metric before mutation, immediate response, recovery time, peak loss, and fixed-window gain

Do not execute a terminal action without a later training and evaluation window.

## Research direction

Prefer controlled comparisons before larger grids:

1. no action and no reset
2. reset only
3. action only
4. action plus reset

Replay the same checkpoint, action, and epoch across LR schedules. Use fixed-architecture and Never-simulation controls. Test acceptance or rollback for harmful actions.

Treat `3°` logistic as the best tested pair, not as a proven default. It currently has only two seeds.
