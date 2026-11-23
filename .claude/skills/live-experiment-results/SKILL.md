---
name: live-experiment-results
description: Every experiment run must publish its results while it runs, so the report shows partial results and anyone can see where the run stands without waiting for the end. Use when writing or changing an experiment script in experiments/, a phase of one, or its report generator in documentation/website/scripts/.
---

# Live experiment results

## The rule

A run that takes hours must never hide its results until the last step. After every finished
unit of work (one arm, one candidate, one generation, one seed, one setting) the run writes what it
knows so far, and the report generator turns that into a partial report. Regenerating the report in
the middle of a run must show every finished unit, clearly marked as partial.

## In the experiment script

1. **One file per finished unit**, written as soon as the unit ends, with everything the report
   needs for it (scores, real accuracy, timing, sizes). This file is also what a resume skips.
2. **A `progress.json` next to them, rewritten after every unit:**
   `status` (`running`, `complete`, `failed`), `done` and `total` units, the unit running now,
   `started_at`, `updated_at`, and elapsed seconds. Set `status: complete` only at the very end.
3. **The summary is rewritten after every unit**, not only at the end: build it from the unit
   files written so far and give it the same `status`, `done` and `total`. Metrics that need more
   data (a correlation over arms, a mean over seeds) are computed from what exists, with the count
   they used, or left `null` when there is too little.
4. **Write atomically** with the script's `write_json` (temporary file and `os.replace`), so a
   reader never sees half a file.
5. **Log one line per unit** (`START ...`, `DONE ...: <key numbers>`), flushed (`python -u`), so
   the log alone tells where the run is.

## In the report generator

1. Read the summary (or, for older runs, the unit files) and render every finished unit, whether
   the run is complete or not.
2. A partial section starts with one line: `Partial: 9 of 16 arms in setting 5 of 5, updated
   <time>.` A complete one has no such line. Conclusions that need all data say so instead of being
   written from partial numbers.
3. "Pending" is only for a phase that has not produced a single unit yet.
4. The generator only reads run files. It never imports a module the running experiment would
   need to match on resume, and editing it never disturbs a running experiment.

## Checking

- A unit test that runs a tiny fake phase, stops after some units, and checks that the summary
  and `progress.json` already hold those units with `status: running`.
- A generator test (or a manual run) on a partial summary that checks the partial line and the
  finished rows appear.

## Running experiments

Do not start experiments yourself; give the user the command, unless they ask you to run it. Do
not poll a running experiment: read its `progress.json` or summary when the user asks.
