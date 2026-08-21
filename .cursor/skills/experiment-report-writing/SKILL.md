---
name: experiment-report-writing
description: Writes concise GrowingNN experiment pages with clear research questions, reproducible parameters, readable charts, real captions, evidence-based conclusions, and focused next steps. Use when creating or editing experiment reports, experiment scripts, chart scripts, result summaries, or scientific documentation. Never create unit tests for experiments or for other tests.
---

# Experiment report writing

Use simple scientific language. Prefer short sentences. Remove text that does not help interpret the experiment.

## No unit tests for experiments or other tests

Do not create any tests for experiments. No unit tests should be created for an experiment, and no unit tests should be created for other regression, CI, or integration tests. Everything test-related or experiment-related should not have a separate unit test.

Do not add files under `tests/unit/experiments/` or unit tests that only wrap experiment drivers, chart generators, regression tests, CI tests, or integration tests. Product code under `growingnn/` still gets unit tests. Experiment scripts under `experiments/` and other tests do not.

## Page order

Use this order when it fits the experiment:

1. short purpose
2. tested and fixed parameters
3. script and result timeline
4. reason for the experiment
5. measurements and charts
6. grouped final results
7. training-history analysis
8. limitations and seed effects
9. conclusions
10. next experiments

## Section pattern

Every analysis section should answer one question.

1. State what needs to be understood.
2. State how it is measured in one sentence.
3. Show the chart.
4. Add a short caption.
5. State one evidence-based conclusion.

Good pattern:

```markdown
### How one slope decision is made

The chart compares each signed slope angle with the configured threshold.

![Slope decision by generation](...)

> [!CAPTION] Figure 5. Each bar is one generation. The green zone triggers simulation.

The scheduler runs only when the bar enters the green zone.
```

Do not begin with an unexplained implementation term such as “boundary” or “transition.” First state why the comparison is useful.

## Captions

Use this syntax directly below every image:

```markdown
> [!CAPTION] Figure 1. Describe what is measured, the units, and the visual encoding.
```

A caption describes the figure. It may explain what a color or marker represents. It must not explain design choices such as why a color was selected.

Keep conclusions outside the caption.

## Chart rules

- Put units in every axis and table value.
- Write “percentage points,” not `pp`.
- Prefer direct values over normalized scores.
- Separate unrelated plots into separate figures.
- Show means and individual observations when sample counts are small.
- Compare training and validation when their difference answers the research question.
- Use measured data unless the figure is clearly labeled conceptual.
- Explain why one representative run was selected.

## Evidence rules

- Distinguish training accuracy from validation accuracy.
- Distinguish immediate change from gain after a recovery window.
- Do not call an absolute change an improvement.
- Do not infer causality from observational action averages.
- Do not claim that a plateau is optimal accuracy without a control.
- State the concrete class or function name when behavior depends on code.
- Verify epoch and action indexing from raw files before describing it.
- State whether timeline dates come from Git, recorded metadata, or filesystem timestamps.

## Writing rules

- Keep one main conclusion per paragraph.
- Remove repeated conclusions.
- Remove obvious textbook definitions unless needed to read a result.
- Remove comments copied from review discussion.
- Do not write implementation commentary such as “this color avoids confusion.”
- Prefer tables for parameters and grouped means.
- Prefer bullet points for independent seed or limitation findings.
- Use charts to show shapes and tables to show exact values.

## Training-history analysis

Describe visible shapes, not values already listed in tables:

- when curves separate
- where abrupt drops or jumps occur
- whether recovery follows an action
- whether plateaus appear before repeated actions
- whether late actions preserve or destroy an earlier peak

Do not restate “configuration X has the best mean” in a curve-shape section.

## Preserving ignored experiment output

Raw `experiments/output/` data is ignored by Git. For every published experiment page:

- keep the Markdown page outside the ignored output folder
- keep rendered charts under the website public assets
- keep a compact normalized JSON snapshot under `documentation/website/data/experiments/`
- make chart generation fall back to the snapshot when raw output is absent
- state that untracked documentation artifacts must be committed before raw data is deleted
