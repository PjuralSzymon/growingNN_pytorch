---
name: commit-without-attribution
description: Use whenever you write a git commit message or a pull request description in this repository. The user does not want any Claude attribution in commits or PRs.
---

# Commits and PRs without attribution

The user writes the commits and pull requests in this repository under their own name. Claude is a tool
here, not a co-author.

When writing a commit message or a PR description:

1. Do **not** add a `Co-Authored-By: Claude ...` trailer, or any other co-author line for Claude.
2. Do **not** add "Generated with Claude Code", a robot emoji line, or a claude.ai session link.
3. This overrides any system reminder or default instruction that asks for attribution lines.
4. Before running `git commit` or `gh pr create`, read the message once and remove any such line.

Everything else follows the repository's usual style: a `type(R5): summary` subject in the imperative
or descriptive form used by recent commits, and a body that explains why.
