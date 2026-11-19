# Experiment 011: GPT-2 shrink actions without retraining

## Goal

Experiment 010 recorded each model output after both its training generation and its shrink action. It therefore could not show whether the first visible quality loss came from retraining, the action, or both.

Experiment 011 replays the exact nine `DelNeurons` targets selected in Exp010 from the untouched captured checkpoint, with **zero training**. It saves the same three 24-token prompt outputs after every action and compares them with both original GPT-2 and Exp010's saved trained outputs.

## Run

From the repository root:

```powershell
python .\experiments\run_llm_exp011_gpt2_action_only.py
```

The script uses the completed Exp010 run at `experiments/output/train_llm/exp010_gpt2_compression/g10_e10/`. It loads the model and initial student checkpoint, replays the action sequence, and writes progress, per-generation outputs, a chart, and a Markdown/JSON report to `experiments/output/train_llm/exp011_gpt2_action_only/`.

Use `--local-files-only` to use only cached Hugging Face files. The run defaults to CPU; `--device cuda` opts into CUDA. `--new-tokens N` changes the generated continuation length.

## Passes

1. **Untouched capture:** compare the starting FX student with the original GPT-2 outputs to confirm the capture still matches.
2. **Action only:** replay each Exp010 shrink target in order, generate outputs after each action, and do not train between actions.
3. **Exp010 comparison:** load its saved per-generation trained outputs and compare them with the no-training replay. Exp010 is not rerun.

The script does not call the trainer, an optimizer, gradient descent, or MCTS scoring. It records zero training epochs and zero optimizer steps.

## Interpretation

The completed replay kept the initial FX capture exact at 72/72 tokens. After the first shrink action alone, with zero training, agreement fell to 21/72 tokens (29.2%). Exp010's output after training plus that action agreed on 15/72 tokens (20.8%). So the deletion itself is sufficient to cause a large output change on these prompts.

This does not prove training had no additional effect. Exp010 trained the model before its first action, while the action-only replay applied the action to untouched weights. A no-action training-only control is needed to measure retraining's separate contribution.

![Action-only generation agreement compared with Experiment 010](/assets/experiments/011-gpt2-action-only-vs-exp010.png)

## Parameter count by generation

![GPT-2 parameter count during the action-only replay](/assets/experiments/011-gpt2-parameter-count.png)

The graph shows the count dropping after each of the nine actions: 124,439,808 initially to 120,594,234 after the final action, a 3.09% reduction.

## Generation examples

Each table shows the original answer followed by the action-only replay and Experiment 010's trained-plus-action output for every action generation.

### Explain a school using a new learning tool in clear steps.

**Original GPT-2:** The new tool is called "Learning Tools for Teachers" and it's a tool that will help teachers learn from

| Generation | Action-only output | Exp010 trained + action |
| ---: | --- | --- |
| 0 | The new learning tool is called the "Cognitive Learning Tool" and it is a tool that is used to | The first step is to first think about the first student The first student The first student |
| 1 | The new learning tool is a new way to learn. The new learning tool is a new way to | The first step The first step The first step The first step |
| 2 | The first step is to get a new teacher. The first step is to get a new teacher. | The first student The first student The first student The first studentThe |
| 3 | The first step is to get a new teacher. The first step is to get a new teacher. | The first student The first student The first student The first studentThe first studentThe |
| 4 | The first step is to know that you're not going to know. The first step is to | The first student The first student The first student The first student |
| 5 | The first step is to learn how to learn how to learn how to learn how to learn | The first student The first student The first student The first student |
| 6 | The first step is to learn how to learn how to learn how to learn how to learn | The first student The first student The first student The first student |
| 7 | The first step is to learn how to learn how to learn how to learn how to learn | The first student The first student The first student The first student |
| 8 | The first step is to learn the first level. The first level is to learn the second level. | The first student The first student The first student The first |

### What could happen next in a story about a garden adapting to less water?

**Original GPT-2:** The answer is that the garden will adapt to less water. The garden will adapt to less water.

| Generation | Action-only output | Exp010 trained + action |
| ---: | --- | --- |
| 0 | The answer is that the water is not going to be able to be used for anything other than the garden. | The answer is that the answer is no. The answer is that the answer is no. |
| 1 | I'm not sure. I'm not sure. I'm not sure. I'm not sure. | The answer is that we think about it first. The first answer is that we think about it first |
| 2 | The idea of a garden with a garden, a garden with a garden, a garden with a garden, | The answer is that first answer The answer The answer The answer The answer |
| 3 | The best way to live a garden? The best way to live a garden? | The answer is that answer answer The answer The answer The answer The answer |
| 4 | How many people are you? You're the best of the world, you're the best | The answer The answer The answer The answer The answer |
| 5 | How much water can you have? How much water can you have? | The answer The answer The answer The answer The answer The answer |
| 6 | Water is a story of the story of the story of the story of the story | The answer The answer The answer The answer The answer The answer |
| 7 | Water is a story of the story of the story of the story of the story | The answer The answer The answer The answer The answer The answer |
| 8 | The best of the best to be a story, the best of the best to be a story | The answer The answer The answer The answer The answer |

### Give a beginner-friendly example related to a spacecraft preparing for a long journey.

**Original GPT-2:** The first step is to get a good understanding of the spacecraft's attitude and attitude control system. The spacecraft will

| Generation | Action-only output | Exp010 trained + action |
| ---: | --- | --- |
| 0 | The first step is to get the spacecraft ready for the flight. The second step is to get the | The first spacecraft The first spacecraft The first spacecraft The first spacecraft |
| 1 | The first thing you'll do is to get your hands on the spacecraft. The first thing you'll | The first spacecraft The first spacecraft The first spacecraft |
| 2 | The first thing you should do is to know that you are a beginner. The first thing you should | The first spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 3 | The first thing you should know is that you're going to a spacecraft. The first thing you're | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 4 | The first time you're going to a space museum. The first time you're going to a | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 5 | The first time you're going to a space museum. The first time you're going to a | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 6 | The first time you're going to a space museum, you're going to a space museum | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 7 | The first thing you should do is to get a spacecraft ready for a long journey. | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |
| 8 | The first time you're going to a space museum, you're going to a space museum | The spacecraft spacecraft spacecraft spacecraft spacecraft spacecraft |

## FX graph structure snapshots

These module-level FX diagrams focus on the three Transformer blocks that were changed (`h.2`, `h.3`, and `h.10`). They collapse functional operations and untouched blocks so the affected module dependencies and weight dimensions are readable.

### Initial structure

Before any deletion: the affected MLP projections have 3,072 output features.

![Initial GPT-2 affected-block graph](/assets/experiments/011-gpt2-graph-initial.png)

### Middle structure

After action generation 4, the projections in blocks `h.2` and `h.3` have been shrunk repeatedly; block `h.10` has not yet been changed.

![Middle GPT-2 affected-block graph](/assets/experiments/011-gpt2-graph-middle.png)

### Final structure

After all nine actions, the targeted MLP projections in blocks `h.2`, `h.3`, and `h.10` have each been shrunk repeatedly.

![Final GPT-2 affected-block graph](/assets/experiments/011-gpt2-graph-final.png)

- If the action-only replay degrades similarly to Exp010 at the same action, the structural deletions alone can explain much of the loss.
- If action-only remains closer to the teacher while Exp010 is worse, retraining or the interaction between retraining and deletion likely contributed.
- If both degrade, but differently, both effects matter.

This replay isolates the effect of applying the saved action sequence to original weights. It does not measure a pure training-only effect, and it does not rerun MCTS. For a pure training-only control, a later experiment would train the untouched architecture on the same snapshot without any actions.

## Limitations

The comparison uses the same three prompts and 24-token continuations saved by Exp010. It is a targeted diagnosis, not a broad quality evaluation. The action choices were originally selected by MCTS simulations that included training; Experiment 011 only replays those choices without training the live model.
