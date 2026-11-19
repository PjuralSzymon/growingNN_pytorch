# Experiment 012: GPT-2 memory dataset comparison

## Goal

Find out whether the memory dataset is limiting GPT-2 recovery after a shrink action. Compare four ways to build the teacher memory while keeping the compressed model, shrink action, training setup, token budget, and evaluation prompts the same.

This experiment follows Experiment 011's finding that the first shrink action alone causes a large generation change. It does not use MCTS to choose actions: every arm receives the same fixed action. That isolates the dataset question.

## Starting point and fixed controls

- Teacher: unquantized `openai-community/gpt2`, in evaluation mode.
- Student initialization: a fresh copy of the same teacher checkpoint for each arm.
- Compression: apply `DelNeurons` at ratio `0.9` to `model.transformer.h.2.mlp.c_fc` in each student. Verify the resized graph runs before training. Record the parameter count and the action-only output as the shared baseline.
- Training: same optimizer, learning rate, batch size, epoch count, and seed in all arms. Use the existing GrowingNN next-token training path for hard-label arms. Arm C needs an experiment-local top-k distillation loss/training loop because the current trainer expects one integer class target per example; keep its optimizer and update schedule identical. Do not run MCTS or allow more architecture changes.
- Medium training budget: **8,192 target-token positions per arm**. Validation is a separate, disjoint **2,048 target-token positions per arm**.
- Keep the GPT-2 vocabulary projection fixed. Report source-token accuracy and cross-entropy; also report prompt continuation agreement and readable samples.

Use a fixed context window of 32 tokens for the primary comparison, matching Experiment 010's memory format. Real-text windows can also be evaluated at longer contexts as a separately labeled extension, not mixed into the primary comparison.

## Four dataset arms

| Arm | How examples are made | Training budget | Validation budget | Target format |
|---|---|---:|---:|---|
| A. Generated prompts | Sample 512 distinct prompts from a larger compositional topic/template pool. Have the frozen teacher greedily continue each prompt for 16 tokens. | 512 × 16 = **8,192** | 128 held-out prompts × 16 = **2,048** | One greedy teacher token per position |
| B. Real text | Tokenize WikiText-2 raw text into contexts from disjoint train and validation document splits. Ask GPT-2 for its top-1 next token at each position. | **8,192 positions** | **2,048 positions** | One hard teacher token per position |
| C. Soft teacher targets | Use the same real-text contexts as B. Ask the frozen teacher for its distribution at each target position. Store the teacher's top-8 token IDs and normalized probabilities. | **8,192 positions** | **2,048 positions** | Top-8 teacher distribution; train with a normalized top-k distillation loss, not single-token cross-entropy |
| D. Mixed | Combine equal portions of A-style generated continuations and B-style real-text contexts with teacher top-1 targets. Keep the held-out split balanced and disjoint. | 4,096 prompt + 4,096 real text = **8,192** | 1,024 prompt + 1,024 real text = **2,048** | One hard teacher token per position |

For A and D, expand the prompt pool rather than repeating the same small set of 150 Experiment 010 templates. Save the prompt text, token IDs, and split membership. For B–D, record the corpus name/configuration, dataset version or revision, and document split. Do not let train and validation windows overlap.

The soft-target arm is a different target representation, not simply a larger dataset. Its storage size will be larger because each position has eight token/probability pairs. Report both the number of positions and serialized snapshot bytes so it is not mistaken for eight times as many examples.

## Run sequence

1. Load and freeze the teacher; save model ID, tokenizer identity, random seed, and corpus revision.
2. Build and save the four medium datasets. Check that each has exactly 8,192 training positions and 2,048 held-out positions, with no split overlap. Stop with a clear error if a source cannot be loaded; do not silently fall back to synthetic text.
3. Record the teacher's baseline loss, top-1 agreement, and generation outputs on held-out data/prompts.
4. Create four independent students from the same teacher state. Apply the same fixed shrink action once to each. Verify identical parameter counts and successful forward passes before training.
5. Train each student against its assigned memory. Run arms sequentially so CPU memory use remains bounded. Save a checkpoint and history after each arm, and rewrite the report after every completed arm.
6. Evaluate every student on the same held-out prompt suite and held-out corpus positions, regardless of its training source. For soft-target training, report hard-token accuracy and cross-entropy too, so the four arms share comparable metrics.
7. Generate the same fixed set of 10 prompts before compression, after compression but before training, and after training. Save the prompt, teacher continuation, action-only continuation, and final continuation for each arm.
8. Render a report with dataset counts/bytes, per-epoch histories, parameter counts, held-out metrics, latency, and side-by-side example outputs.

## Primary comparisons

- Does training improve the fixed action-only student's held-out token loss/agreement?
- Which source best restores the original teacher's held-out behavior: generated continuations, real text, or their mixture?
- Does soft top-k distillation preserve behavior better than a hard greedy target at the same number of token positions?
- Do gains on teacher-forced next-token metrics correspond to better free-running prompt generations?
- What is the tradeoff between recovery, training time, and memory snapshot bytes?

Do not conclude that a dataset is sufficient from training loss alone. Held-out loss, held-out agreement, and free-running generations are required. Keep the action-only result as the reference line in every chart.

## Expected output layout

```text
experiments/output/train_llm/exp012_gpt2_memory_dataset_comparison/
  report.md
  report.json
  datasets/
    generated_prompts_train.pt
    generated_prompts_validation.pt
    real_text_train.pt
    real_text_validation.pt
    soft_teacher_train.pt
    soft_teacher_validation.pt
    mixed_train.pt
    mixed_validation.pt
  arms/
    generated_prompts/{student_after_training.pt,history.json}
    real_text/{student_after_training.pt,history.json}
    soft_teacher/{student_after_training.pt,history.json}
    mixed/{student_after_training.pt,history.json}
  charts/
    dataset_snapshot_sizes.png
    training_and_validation.png
```

The Markdown report should start with a short result summary and the four-arm comparison table. Follow it with the parameter/action controls, dataset recipe and exact counts, charts, per-arm prompt tables, limitations, and a recommendation for the next experiment. Keep the serialized datasets and model checkpoints outside the website bundle; publish only the report and compact figures.

## Initial run settings

- Four arms, 8,192 training targets and 2,048 validation targets each.
- Three epochs per arm for the first medium run. Use the same batch size and learning rate across arms; record the exact values in the JSON report.
- CPU is the safe default, with an explicit optional CUDA setting.
- Include a small `--smoke` mode that creates tiny versions of all four datasets and completes one short training pass, to validate dataset construction, soft-target loss, action application, checkpointing, and report generation before the medium run.

Three epochs is a feasibility setting, not evidence of convergence. If validation loss is still consistently improving at epoch three, increase epochs in a follow-up while holding the dataset fixed.

## Limitations and interpretation

- WikiText-2 is a practical first natural-language corpus, not broad coverage of GPT-2's training distribution. Record that limitation in the report.
- Generated prompt continuations can be repetitive and narrow. The expanded prompt pool and a disjoint validation pool reduce, but do not eliminate, this bias.
- Matching the teacher on a finite memory does not establish general language quality. Held-out text and prompt generation are both necessary.
- The chosen deletion is a controlled probe based on Experiment 010/011, not a claim that this is the best GPT-2 compression action.
- A larger dataset may improve recovery, but it cannot guarantee recovery of functions lost by the structural edit.

## Commands

```powershell
python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py --smoke
python .\experiments\run_llm_exp012_gpt2_memory_dataset_comparison.py
```

The normal run uses the four medium arms. The smoke run is only a pipeline check and labels its result as smoke data. WikiText-2 rows are fetched from Hugging Face and cached under the output folder, so neither the datasets package nor an AWS download is required. The runner is implemented in run_llm_exp012_gpt2_memory_dataset_comparison.py.
