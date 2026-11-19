# Experiment 012: GPT-2 memory dataset comparison

## Status

Runner implemented. This page describes the protocol; it does not contain run results until the script is run.

## Goal

Compare four medium-sized teacher-memory datasets to find out which gives a fixed, compressed GPT-2 student the best recovery. Each arm uses 8,192 training target positions and 2,048 held-out positions.

All arms start from the same GPT-2 checkpoint and receive the same `DelNeurons` action (`model.transformer.h.2.mlp.c_fc`, ratio `0.9`). Training settings and evaluation prompts are held fixed. MCTS is disabled so the experiment isolates the dataset recipe.

## Dataset arms

| Arm | Dataset recipe | Training targets | Validation targets |
|---|---|---:|---:|
| Generated prompts | 512 distinct prompts, each continued by the teacher for 16 tokens | 8,192 | 2,048 from 128 separate prompts |
| Real text | WikiText-2 contexts labeled with GPT-2 top-1 predictions | 8,192 | 2,048 |
| Soft teacher targets | Same real-text windows, storing teacher top-8 token probabilities per position | 8,192 | 2,048 |
| Mixed | Half generated continuations and half real-text contexts labeled with GPT-2 | 8,192 | 2,048 balanced across sources |

Soft targets change the information stored per position, not the position count. Report snapshot bytes separately from target-position count.

## Run sequence

1. Freeze GPT-2 as teacher and generate the four datasets with reproducible splits.
2. Validate exact counts and ensure no train/validation overlap.
3. Record teacher and action-only baselines.
4. Create four identical student copies, apply the same shrink action, and verify their parameter counts and forward passes match.
5. Train each student against its dataset for the same three-epoch initial run. Use the current GrowingNN trainer for hard-label arms; the soft-target arm needs an experiment-local top-k distillation loss because the current trainer expects integer labels. Keep optimizer settings and update counts aligned. Save epoch history and checkpoints.
6. Evaluate all arms on the same held-out corpus positions and prompts. Report token loss, teacher agreement, generation examples, parameter count, runtime, and memory snapshot size.
7. Publish a report with per-epoch plots and side-by-side prompt outputs.

## Commands

From the repository root, run a small pipeline check:

    python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py --smoke

Run the medium comparison:

    python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py

The medium run fetches WikiText-2 rows from Hugging Face on its first run and caches them in the experiment output folder. It does not need the datasets package or an AWS download. A local UTF-8 corpus can be used with --text-file PATH. The run defaults to CPU; --device cuda opts into GPU.

The report and snapshots are written under experiments/output/train_llm/exp012_gpt2_memory_dataset_comparison/.

Three epochs is a first feasibility setting. If validation is still improving, follow up with more epochs while keeping the dataset fixed.

## Interpretation limits

WikiText-2 and template-generated prompts do not cover the full language distribution GPT-2 learned. Good training metrics alone do not prove general generation quality. The experiment must show held-out loss/agreement and free-running outputs, and compare them with the action-only baseline.

## Protocol details

See the [full Experiment 012 design](../../../experiments/experiment_012_gpt2_memory_dataset_comparison.md) for controls, output layout, proposed commands, limitations, and extension points.
