---
name: commit-plan-writing
description: >-
  Analyzes git status and diffs, then writes a numbered commit plan with
  dependency order, one-line feat/fix/refactor(R5) subjects, git add commands,
  and short per-file roles. Use when the user asks for a commit plan, commit
  summary, what to commit next, how to split commits, or git add + commit
  grouping for current changes. Never create or plan unit tests for
  experiments or for other tests.
---

# Commit plan writing

Write commit plans for the current working tree in the exact style below. Do not create empty commits. Do not commit unless the user explicitly asks.

## When invoked

1. Run readonly git inspection in parallel:
   - `git status --short`
   - `git diff HEAD --stat`
   - `git diff --cached --stat`
   - `git log -8 --oneline`
   - `git ls-files --others --exclude-standard`
2. Read enough of the diffs to group by dependency and meaning (not by accidental file proximity).
3. Prefer writing/updating a local plan file only if the user asks for a file; otherwise answer in chat in the same form.
4. Never stage secrets (`.env`, credentials, keys).
5. Exclude noisy/local-only paths unless the user asks: `COMMIT_PLAN_*.md`, `__pycache__`, `.pytest_cache`, raw `experiments/output/**` boards, generated Angular cache.

## Commit message rules (this repo)

Every subject must be exactly:

```text
<prefix> <imperative description>
```

Prefix is one of:

- `fix(R5)` — bug fix, regression, incorrect behavior
- `feat(R5)` — new behavior, new module, new capability, tests for new capability
- `refactor(R5)` — structure/naming/docs/tooling with no intended behavior change

Use present/imperative: "add", "freeze", "expose", "document", not "added"/"adding".
One line only. Specific enough to understand without opening the diff.

## How to split commits

Order commits so the tree stays coherent after each step:

1. Core abstraction / API first
2. Call-site wiring that depends on that API
3. Product unit tests for `growingnn/` behavior (with the feature, or immediately after)
4. Experiments / drivers (no unit tests for these)
5. Docs / skills last

Do not create any tests for experiments. No unit tests should be created for an experiment, and no unit tests should be created for other regression, CI, or integration tests. Everything test-related or experiment-related should not have a separate unit test. Do not plan `tests/unit/experiments/` files or unit tests that wrap regression, CI, integration, or experiment scripts.

Put mechanical rename-only import retargets in their own `refactor(R5)` commit when they dominate the diff.
Do not mix unrelated features into one commit.
Keep each commit reviewable: one clear why.

## Required output form

Use this structure verbatim (fill in real content):

```markdown
# Commit plan (remaining work)

Branch: <branch-name>

Already committed:
- `<shortsha>` <subject>
- ...

Do **not** commit: <paths to exclude>

---

## Commit 1 — <short label> (do this first)

**Commit:** `<prefix(R5) imperative subject>`

**What:** <1-3 sentences: what changes and why>

```bash
git add <file1> <file2> ...
git commit -m "<prefix(R5) imperative subject>"
```

| File | Role |
|---|---|
| `<path>` | <one-line role> |

---

## Commit 2 — <short label>

**Commit:** `<prefix(R5) imperative subject>`

```bash
git add <files>
git commit -m "<prefix(R5) imperative subject>"
```

---

## Commit N — <short label>

...

---

## Order

1 → 2 → ... → N
```

Notes for filling the template:

- Start with the main abstraction/core change as Commit 1 when one exists.
- Include a File/Role table on the first/core commit; later commits may omit the table if the `git add` list is enough.
- Repeat the exact commit subject in both `**Commit:**` and `git commit -m`.
- Use one-line `git add` commands (space-separated paths). Quote paths that contain spaces.
- End with `## Order` as `1 → 2 → ... → N`.

## Example shape (reference)

```markdown
# Commit plan (remaining work)

Branch: work/R5/adding-composite-lr

Already committed:
- `22af17a` split into lr_scheduler_action / lr_scheduler_global
- `77015d4` retarget imports to ActionLearningRateScheduler

Do **not** commit: COMMIT_PLAN_composed_lr_and_exp004.md

---

## Commit 1 — main abstraction (do this first)

**Commit:** `refactor(R5) make LearningRateScheduler an ABC with Action and Composed subclasses`

**What:** Turn LearningRateScheduler into the shared ABC; move concrete action schedules to ActionLearningRateScheduler; make ComposedLearningRateScheduler inherit the same interface.

```bash
git add growingnn/training/lr_scheduler_action.py growingnn/training/lr_scheduler_global.py tests/unit/training/lr_scheduler_test.py
git commit -m "refactor(R5) make LearningRateScheduler an ABC with Action and Composed subclasses"
```

| File | Role |
|---|---|
| growingnn/training/lr_scheduler_action.py | ABC + ActionLearningRateScheduler |
| growingnn/training/lr_scheduler_global.py | ComposedLearningRateScheduler(LearningRateScheduler) |
| tests/unit/training/lr_scheduler_test.py | inheritance + schedule tests |

---

## Order

1 → 2 → 3
```

## Quality checklist

- [ ] Inspected status + staged + unstaged + recent log
- [ ] Commit 1 is the main structural/API change when present
- [ ] Every subject uses `fix(R5)` / `feat(R5)` / `refactor(R5)`
- [ ] Each commit has copy-paste `git add` + `git commit -m`
- [ ] Exclusions listed under Do not commit
- [ ] Ends with Order `1 → 2 → ...`
