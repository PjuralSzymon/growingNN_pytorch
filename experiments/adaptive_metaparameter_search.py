"""Adaptive meta-parameter search (GitHub issue #73).

All sampling, softmax grades, EMA updates, and live status files live here.
Callers only loop: next_config() -> evaluate -> record().
"""

from __future__ import annotations

import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any, Mapping

Combo = dict[str, Any]
FrozenCombo = tuple[tuple[str, Any], ...]

JSON_NAME = "adaptive_search.json"
MD_NAME = "adaptive_search.md"
REJECT_TRIES = 64


def _softmax(grades: list[float], tau: float) -> list[float]:
    scaled = [g / tau for g in grades]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _combo_dict(frozen: FrozenCombo) -> Combo:
    return {axis: value for axis, value in frozen}


class AdaptiveMetaParameterSearch:
    """Issue #73 search over a Cartesian pool of independent group values."""

    def __init__(
        self,
        groups: Mapping[str, tuple[Any, ...]],
        output_dir: Path,
        *,
        max_iters: int = 50,
        n_init: int = 5,
        tau: float = 0.15,
        beta: float = 0.3,
        paper_target: float | None = None,
        target_tol: float = 0.04,
        rng: random.Random | None = None,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if n_init < 0:
            raise ValueError("n_init must be >= 0")
        self.groups = {axis: tuple(values) for axis, values in groups.items()}
        self.axes = tuple(self.groups)
        self.output_dir = Path(output_dir)
        self.max_iters = max_iters
        self.n_init = n_init
        self.tau = tau
        self.beta = beta
        self.paper_target = paper_target
        self.target_tol = target_tol
        self.rng = rng or random.Random()
        self.pool: set[FrozenCombo] = {
            tuple(zip(self.axes, values)) for values in product(*(self.groups[axis] for axis in self.axes))
        }
        self.unevaluated: set[FrozenCombo] = set(self.pool)
        self.grades: dict[str, dict[Any, float]] = {
            axis: {value: 0.5 for value in values} for axis, values in self.groups.items()
        }
        self.trials: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.pending: FrozenCombo | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def next_config(self) -> Combo | None:
        """Return the next unevaluated combo, or None when the search should stop."""
        if self._should_stop() and self.pending is None:
            return None
        if self.pending is not None:
            return _combo_dict(self.pending)
        if not self.unevaluated:
            return None
        iteration = len(self.trials) + 1
        if iteration <= self.n_init:
            chosen = self.rng.choice(tuple(self.unevaluated))
        else:
            chosen = self._sample_from_grades()
        self.unevaluated.remove(chosen)
        self.pending = chosen
        self._write_live()
        return _combo_dict(chosen)

    def record(self, combo: Mapping[str, Any], val_acc: float, test_acc: float) -> None:
        """Score one combo, update axis grades, and rewrite live status files."""
        frozen = self._canonical(combo)
        if self.pending is not None and frozen != self.pending:
            raise ValueError("record combo does not match next_config pending combo")
        row = {"combo": _combo_dict(frozen), "val_acc": float(val_acc), "test_acc": float(test_acc)}
        self.trials.append(row)
        if self.best is None or float(val_acc) > float(self.best["val_acc"]):
            self.best = row
        self.pending = None
        self._refresh_grades()
        self._write_live()

    def probabilities(self) -> dict[str, dict[str, float]]:
        """Current softmax P_a over each group's values."""
        out: dict[str, dict[str, float]] = {}
        for axis, values in self.groups.items():
            probs = _softmax([self.grades[axis][value] for value in values], self.tau)
            out[axis] = {self._label(value): prob for value, prob in zip(values, probs)}
        return out

    def _should_stop(self) -> bool:
        if len(self.trials) >= self.max_iters:
            return True
        if not self.unevaluated and self.pending is None:
            return True
        if self.paper_target is not None and self.best is not None:
            if float(self.best["test_acc"]) >= self.paper_target - self.target_tol:
                return True
        return False

    def _canonical(self, combo: Mapping[str, Any]) -> FrozenCombo:
        frozen: list[tuple[str, Any]] = []
        for axis in self.axes:
            raw = combo[axis]
            match = next((option for option in self.groups[axis] if option == raw or str(option) == str(raw)), raw)
            frozen.append((axis, match))
        return tuple(frozen)

    def _sample_from_grades(self) -> FrozenCombo:
        axis_probs = {
            axis: _softmax([self.grades[axis][value] for value in values], self.tau)
            for axis, values in self.groups.items()
        }
        for _ in range(REJECT_TRIES):
            drawn = tuple(
                (axis, self.rng.choices(self.groups[axis], weights=axis_probs[axis], k=1)[0])
                for axis in self.axes
            )
            if drawn in self.unevaluated:
                return drawn
        weights = []
        remaining = tuple(self.unevaluated)
        for combo in remaining:
            weight = 1.0
            lookup = dict(combo)
            for axis in self.axes:
                index = self.groups[axis].index(lookup[axis])
                weight *= axis_probs[axis][index]
            weights.append(weight)
        if sum(weights) <= 0:
            return self.rng.choice(remaining)
        return self.rng.choices(remaining, weights=weights, k=1)[0]

    def _refresh_grades(self) -> None:
        for axis, values in self.groups.items():
            for value in values:
                scores = [float(trial["val_acc"]) for trial in self.trials if trial["combo"][axis] == value]
                if not scores:
                    continue
                raw = sum(scores) / len(scores)
                self.grades[axis][value] = (1.0 - self.beta) * self.grades[axis][value] + self.beta * raw

    def _label(self, value: Any) -> str:
        return str(value)

    def _state(self) -> dict[str, Any]:
        return {
            "iteration": len(self.trials),
            "max_iters": self.max_iters,
            "n_init": self.n_init,
            "tau": self.tau,
            "beta": self.beta,
            "unevaluated_count": len(self.unevaluated),
            "pool_size": len(self.pool),
            "pending": None if self.pending is None else _combo_dict(self.pending),
            "best": self.best,
            "grades": {
                axis: {self._label(value): grade for value, grade in values.items()}
                for axis, values in self.grades.items()
            },
            "probabilities": self.probabilities(),
            "trials": self.trials,
        }

    def _write_live(self) -> None:
        state = self._state()
        json_path = self.output_dir / JSON_NAME
        json_path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
        lines = [
            "# Adaptive meta-parameter search",
            "",
            f"Iteration `{state['iteration']}` / `{self.max_iters}`. Unevaluated `{state['unevaluated_count']}` / `{state['pool_size']}`.",
            "",
        ]
        if self.best is not None:
            lines.append(f"Best val_acc `{self.best['val_acc']:.4f}` test_acc `{self.best['test_acc']:.4f}`.")
            lines.append("")
            lines.append("Best combo:")
            for axis, value in self.best["combo"].items():
                lines.append(f"- `{axis}`: `{value}`")
            lines.append("")
        if self.trials:
            last = self.trials[-1]
            lines.append(f"Last trial val_acc `{last['val_acc']:.4f}`.")
            lines.append("")
        for axis in self.axes:
            lines.append(f"## {axis}")
            lines.append("")
            lines.append("| value | grade | P(a) |")
            lines.append("| --- | ---: | ---: |")
            probs = self.probabilities()[axis]
            for value in self.groups[axis]:
                label = self._label(value)
                lines.append(
                    f"| `{label}` | `{self.grades[axis][value]:.4f}` | `{probs[label]:.4f}` |"
                )
            lines.append("")
        (self.output_dir / MD_NAME).write_text("\n".join(lines), encoding="utf-8")

    def _load_state(self) -> None:
        json_path = self.output_dir / JSON_NAME
        if not json_path.is_file():
            self._write_live()
            return
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.trials = []
        for trial in payload.get("trials") or []:
            self.trials.append({
                "combo": _combo_dict(self._canonical(trial["combo"])),
                "val_acc": float(trial["val_acc"]),
                "test_acc": float(trial["test_acc"]),
            })
        scored = {self._canonical(trial["combo"]) for trial in self.trials}
        self.unevaluated = self.pool - scored
        pending = payload.get("pending")
        self.pending = None if pending is None else self._canonical(pending)
        if self.pending is not None:
            self.unevaluated.discard(self.pending)
        best = payload.get("best")
        if best is None:
            self.best = None
        else:
            self.best = {
                "combo": _combo_dict(self._canonical(best["combo"])),
                "val_acc": float(best["val_acc"]),
                "test_acc": float(best["test_acc"]),
            }
        stored_grades = payload.get("grades") or {}
        if stored_grades:
            for axis, values in stored_grades.items():
                if axis not in self.grades:
                    continue
                for label, grade in values.items():
                    for option in self.groups[axis]:
                        if self._label(option) == str(label):
                            self.grades[axis][option] = float(grade)
                            break
        else:
            self._replay_grades()
        self._write_live()

    def _replay_grades(self) -> None:
        self.grades = {axis: {value: 0.5 for value in values} for axis, values in self.groups.items()}
        seen: dict[str, dict[Any, list[float]]] = {axis: {value: [] for value in values} for axis, values in self.groups.items()}
        for trial in self.trials:
            combo = trial["combo"]
            val = float(trial["val_acc"])
            for axis in self.axes:
                seen[axis][combo[axis]].append(val)
            for axis, values in self.groups.items():
                for value in values:
                    scores = seen[axis][value]
                    if scores:
                        raw = sum(scores) / len(scores)
                        self.grades[axis][value] = (1.0 - self.beta) * self.grades[axis][value] + self.beta * raw
