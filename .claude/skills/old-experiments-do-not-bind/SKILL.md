---
name: old-experiments-do-not-bind
description: Finished experiments and their reports exist to help design new solutions, not to constrain the code. Use when adding or changing an action, a default in growingnn/core/config.py, a search algorithm, or anything an old experiment script also uses, and you are tempted to keep a default off or an old behaviour only so old experiments rerun the same way.
---

# Old experiments do not bind new code

## The rule

Old experiments and reports are a record of what we learned. Their job is to point at the next
improvement. Rerunning them with identical results is **not** a priority and is never a reason to:

- keep a new action, fix or feature **off by default** ("so old experiments get the same candidates");
- keep a known-bad default, a bug, or an old code path alive for an old script;
- add compatibility flags, version switches or special cases for finished experiments;
- avoid changing `growingnn/` because an old experiment imports it.

New actions and fixes are **on by default**. The library moves forward; old experiments are read,
not preserved.

## What still matters

- **A run that is in progress now:** do not change the code it imports until it finishes, because
  resuming needs the same source (see the source snapshot in its output folder). This is about a
  live run, not about finished ones.
- **Reports keep their numbers:** a report states what happened in its run. Do not rewrite old
  results; if new code would give different numbers, that is expected and needs no note.
- **Old results guide new work:** read them to decide what to build next and compare new runs
  against them.

## When a change alters what old scripts would do

Make the change, and mention it to the user in one line ("old experiment scripts that rely on
defaults now also get X"). Do not ask whether to keep the old behaviour, and do not make it opt-in.
