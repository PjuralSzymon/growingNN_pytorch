# Experiment 010: GPT-2 compression with GrowingNN search

## Goal

Use the pretrained `openai-community/gpt2` model as a frozen teacher, save a generated next-token memory snapshot, and let the existing GrowingNN trainer and Monte Carlo search try neuron-deletion actions. The score gives accuracy weight `0.1` and model size weight `1.0`. The vocabulary output projection is kept fixed because changing its width changes the tokenizer's output space.

## Setup

Script: `experiments/train_llm_exp010_gpt2_compression.py`

Run the quick feasibility check:

```powershell
python ./experiments/train_llm_exp010_gpt2_compression.py --smoke --local-files-only
```

Run the 10-generation, 10-epoch configuration after the model is cached:

```powershell
python ./experiments/train_llm_exp010_gpt2_compression.py --local-files-only
```

This larger run plans 100 training epochs total. It writes per-epoch losses and agreement to the console, saves `progress.json` and a progress checkpoint after each generation, and compares greedy generation latency and output tokens on the same 10 held-aside prompts before and after training. Its outputs go to `experiments/output/train_llm/exp010_gpt2_compression/g10_e10/`, keeping the earlier feasibility-run report intact. Use `--generations N --epochs M` to make a smaller intermediate run; for example, `--generations 2 --epochs 1`.

The script captures GPT-2 with the project's Hugging Face FX capture adapter. The teacher generates target continuations; each target is paired with a fixed-width context and attention mask, so the existing trainer optimizes next-token cross entropy. Position IDs are reconstructed from the mask so left padding does not shift GPT-2 token positions. Search enables `DelNeurons` at a 10% shrink ratio and disables growth and layer deletion. The size score is calibrated locally so the unmodified model scores `0.5` on the size term. The requested score weights are teacher-token agreement `0.1` and model size `1.0`.

The recorded feasibility verification below used 24 training prompts and eight separate held-out prompts, up to four continuation tokens per prompt, one training epoch per generation, and two generations. The current larger default is 10 generations with 10 epochs each. MCTS can retain a full copied GPT-2 graph for each expanded candidate. Trying all 12 candidates exhausted system memory at candidate five, and CUDA had less than 2 GiB free. The experiment therefore samples three eligible candidates per search round with a fixed seed and records the deferred candidates and their reason; MCTS chooses among the offered actions. Smoke mode samples one candidate. The default device is CPU; pass `--device cuda` to opt into GPU or `--device auto` for memory-aware selection.

## Results

Completed 10-generation, 10-epoch run on CPU with the unquantized GPT-2 checkpoint. The run completed 100 training epochs and nine existing MCTS-selected `DelNeurons` mutations:

| Measure | Result |
| --- | ---: |
| Train / held-out prompts | 24 / 8 |
| Teacher target tokens, train / held-out | 96 / 32 |
| Captured FX nodes | 622 |
| Eligible shrink candidates / offered to MCTS | 12 / 3 |
| Search generations / training epochs | 10 / 100 |
| Candidates offered | `h.2.mlp.c_fc`, `h.3.mlp.c_fc`, `h.10.mlp.c_fc` |
| Selected actions | 9 `DelNeurons` actions across the three offered MLP projections |
| Parameters before | 124,439,808 |
| Parameters after | 120,594,234 |
| Reduction | 3,845,574 (3.09%) |
| Baseline teacher-token accuracy | 1.00 |
| Student training teacher-token accuracy | 0.854 (82 / 96) |
| Student held-out teacher-token accuracy | 0.813 (26 / 32) |
| Held-out greedy token agreement | 0.781 (25 / 32) |
| Held-out exact continuation match | 2 / 8 prompts |
| Training and search runtime | 1,715 seconds (28.6 minutes) |
| Same 10-prompt generation time, before / after | 0.688 / 0.684 seconds |
| Same-prompt generated token agreement | 33 / 40 (82.5%); 3 / 10 exact continuations |
| Full-context 24-token output agreement, 3 prompts | 15 / 72 (20.8%); 0 / 3 exact continuations |
| Initial FX capture vs Hugging Face output | 72 / 72 tokens match |
| Status | Shrunk |

Held-out teacher-token agreement improved from the first measured epoch's `75.0%` to `81.3%`, while the model lost `3.09%` of its parameters. The initial captured graph exactly matched Hugging Face generation on three full-context prompts. After compression, the same 24-token examples matched only `15 / 72` tokens; this points to behavior loss from compression rather than a capture mismatch. The 10-prompt timing check compares four-token outputs. Both timing measurements are single CPU runs without KV caching, so they are not reliable speed claims.

![Training and held-out loss, teacher-token agreement, and parameter count over 100 epochs](/assets/experiments/010-gpt2-training-history.png)

### Readable generation examples

The original captured student matched Hugging Face GPT-2 exactly on all 72 generated tokens for these three prompts. The tables show the student after each search step, with the same 24-token greedy generation and full GPT-2 context. The compressed outputs become repetitive; this is behavior loss from the selected shrink actions.

**Prompt:** Explain a school using a new learning tool in clear steps.

**Original GPT-2:** The new tool is called "Learning Tools for Teachers" and it's a tool that will help teachers learn from

| Student generation | Action applied | Compressed output |
| ---: | --- | --- |
| 0 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The first step is to first think about the first student The first student The first student |
| 1 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The first step The first step The first step The first step |
| 2 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The first student The first student The first student The first studentThe |
| 3 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The first student The first student The first student The first studentThe first studentThe |
| 4 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The first student The first student The first student The first student |
| 5 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The first student The first student The first student The first student |
| 6 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The first student The first student The first student The first student The first |
| 7 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The first student The first student The first student The first student |
| 8 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The first student The first student The first student The first |
| 9 | none (final generation) | The first student The first student The first student The first |

**Prompt:** What could happen next in a story about a garden adapting to less water?

**Original GPT-2:** The answer is that the garden will adapt to less water. The garden will adapt to less water.

| Student generation | Action applied | Compressed output |
| ---: | --- | --- |
| 0 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The answer is that the answer is no. The answer is that the answer is no. |
| 1 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The answer is that we think about it first. The first answer is that we think about it first |
| 2 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The answer is that first answer The answer The answer The answer The answer |
| 3 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The answer is that answer answer The answer The answer The answer The answer |
| 4 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The answer The answer The answer The answer The answer |
| 5 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The answer The answer The answer The answer The answer The answer |
| 6 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The answer The answer The answer The answer The answer The answer |
| 7 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The answer The answer The answer The answer The answer The answer |
| 8 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The answer The answer The answer The answer The answer |
| 9 | none (final generation) | The answer The answer The answer The answer The answer |

**Prompt:** Give a beginner-friendly example related to a spacecraft preparing for a long journey.

**Original GPT-2:** The first step is to get a good understanding of the spacecraft's attitude and attitude control system. The spacecraft will

| Student generation | Action applied | Compressed output |
| ---: | --- | --- |
| 0 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The first spacecraft The first spacecraft The first spacecraft The first spacecraft The first |
| 1 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The first spacecraft The first spacecraft The first spacecraft The first spacecraft |
| 2 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The first spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The |
| 3 | DelNeurons: h.3.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft spacecraft spacecraft |
| 4 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft spacecraft |
| 5 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft spacecraft |
| 6 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft spacecraft |
| 7 | DelNeurons: h.10.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft |
| 8 | DelNeurons: h.2.mlp.c_fc -> 0.9 | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft |
| 9 | none (final generation) | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft The spacecraft spacecraft |

### Training history

| Generation | Train loss (first → last) | Held-out loss (first → last) | Held-out agreement | Parameters at end |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.146 → 0.389 | 0.892 → 0.695 | 81.2% | 124,439,808 |
| 2 | 0.398 → 0.301 | 0.666 → 0.675 | 81.2% | 123,966,412 |
| 3 | 0.407 → 0.330 | 0.908 → 0.943 | 81.2% | 123,493,016 |
| 4 | 0.639 → 0.375 | 1.129 → 1.145 | 78.1% | 123,067,267 |
| 5 | 0.459 → 0.338 | 1.172 → 1.197 | 81.2% | 122,684,554 |
| 6 | 0.553 → 0.408 | 1.242 → 1.364 | 81.2% | 122,258,805 |
| 7 | 0.439 → 0.360 | 1.284 → 1.322 | 81.2% | 121,785,409 |
| 8 | 0.484 → 0.384 | 1.319 → 1.435 | 81.2% | 121,359,660 |
| 9 | 0.419 → 0.402 | 1.389 → 1.420 | 81.2% | 120,976,947 |
| 10 | 0.427 → 0.412 | 1.395 → 1.430 | 81.2% | 120,594,234 |

The configured weights were agreement `0.1` and size `1.0`; the calibrated size score rose from `0.500` to `0.508`. The run selected nine shrink actions, one after each of the first nine training generations. The held-out loss rose from `0.695` at the end of generation 1 to `1.430` at the end, despite token accuracy remaining near `81%`; this is a meaningful quality tradeoff to monitor rather than evidence that all behavior was retained.

Candidate diagnostics recorded 12 attention QKV projections rejected because their outputs are split into head components, 24 projections rejected because a dependent `LayerNorm` has no supported width-resize rule, one vocabulary projection pinned to tokenizer classes, and nine otherwise eligible MLP projections deferred by the per-round graph-copy memory budget. Three eligible candidates were offered to MCTS each round; MCTS selected the action rather than a hand-written picker. Detailed per-target reasons and all 100 epoch values are in `report.json`; a readable copy is in `report.md`.

The earlier two-generation smoke run remains a quick plumbing check: it executed one shrink action, reduced parameters by `0.38%`, and measured `0.41s` before versus `0.37s` after on ten two-token prompts, with matching outputs on all ten. It is not the 10×10 result.

## Artifacts

Each run writes `report.json`, `report.md`, `generation_outputs.json`, `training_history.png`, `progress.json`, `teacher_checkpoint.pt`, `teacher_memory_snapshot.pt`, `student_initial_checkpoint.pt`, `student_best_checkpoint.pt`, and `student_compressed.pt` into its run folder. The completed 10×10 run is in `experiments/output/train_llm/exp010_gpt2_compression/g10_e10/`; the previous completed run was preserved in `g10_e10_before_generation_examples/`. The Markdown report includes the original GPT-2 outputs and all 10 post-search generation outputs for three prompts. These raw files are ignored by Git. The report page is included in the View Local website and appears after running `documentation/website/view_local.bat`.

## Finding

The prior Experiment 010 trained a second model with the same architecture. The replacement uses the existing search and completes a 10×10 run with a 3.09% parameter reduction. It retained `81.3%` held-out next-token agreement, but full-context output agreement was only `15 / 72` tokens for three prompts. The generation-by-generation examples show repetitive text after the first shrink. The high next-token score therefore overstates retained generation quality. Stronger compression needs coordinated Transformer edits, a larger teacher snapshot, and evaluation on longer outputs.

## Next step

Next, improve candidate scoring and safe coordinated width changes so attention and dependent normalization layers can participate, then repeat with a larger, more varied snapshot and stronger independent evaluation. Avoid interpreting unchanged token accuracy as full behavior retention: the held-out loss increased substantially during the 10×10 run.
