"""Read-only candidate score previews for alternative simulation weights."""

from __future__ import annotations

import copy
import math
from typing import Any


def recalculate_simulation(
    simulation: dict[str, Any],
    accuracy_weight: float,
    param_count_weight: float,
) -> dict[str, Any]:
    """Recalculate saved candidate composites without rerunning a simulation."""
    accuracy_weight, param_count_weight = float(accuracy_weight), float(param_count_weight)
    weights = (accuracy_weight, param_count_weight)
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError("Weights must be finite and non-negative")
    divisor = sum(weights)
    if divisor <= 0:
        raise ValueError("At least one weight must be greater than zero")

    candidates = simulation.get("candidateActions") or simulation.get("candidates") or []
    recalculated: list[dict[str, Any]] = []
    unavailable_actions: list[str] = []
    for index, candidate in enumerate(candidates):
        terms = (candidate.get("scoreBreakdown") or {}).get("terms") or {}
        accuracy_raw = (terms.get("accuracy") or {}).get("raw")
        param_count_raw = (terms.get("paramCount") or {}).get("raw")
        action = candidate.get("action")
        name = candidate.get("name") or action or f"Action {index + 1}"
        if (accuracy_weight > 0 and accuracy_raw is None) or (
            param_count_weight > 0 and param_count_raw is None
        ):
            unavailable_actions.append(str(name))
            continue
        try:
            accuracy_value = float(accuracy_raw or 0)
            param_count_value = float(param_count_raw or 0)
        except (TypeError, ValueError):
            unavailable_actions.append(str(name))
            continue
        if not math.isfinite(accuracy_value) or not math.isfinite(param_count_value):
            unavailable_actions.append(str(name))
            continue
        accuracy_contribution = accuracy_weight * accuracy_value
        param_count_contribution = param_count_weight * param_count_value
        composite = (accuracy_contribution + param_count_contribution) / divisor
        recalculated.append(
            {
                "index": index,
                "action": action,
                "name": name,
                "score": composite,
                "scoreBreakdown": {
                    "composite": composite,
                    "terms": {
                        "accuracy": {
                            "weight": accuracy_weight,
                            "raw": accuracy_raw,
                            "weighted": accuracy_contribution,
                        },
                        "paramCount": {
                            "weight": param_count_weight,
                            "raw": param_count_raw,
                            "weighted": param_count_contribution,
                        },
                    },
                    "weights": {
                        "weight_acc": accuracy_weight,
                        "weight_countW": param_count_weight,
                    },
                },
            }
        )

    projected = max(recalculated, key=lambda row: row["score"]) if recalculated else None
    original_action = simulation.get("actionChosen")
    if original_action is None:
        original = next(
            (candidate for candidate in candidates if candidate.get("chosen") or candidate.get("isChosen")),
            None,
        )
        original_action = original.get("action") if original else None
    projected_action = projected.get("action") if projected else None
    return {
        "weights": {
            "accuracy": accuracy_weight,
            "paramCount": param_count_weight,
        },
        "candidates": recalculated,
        "originalAction": original_action,
        "projectedAction": projected_action,
        "projectedName": projected.get("name") if projected else None,
        "sameAction": projected_action is not None and projected_action == original_action,
        "unavailableActions": unavailable_actions,
    }


def apply_recalculated_scores(
    search_tree: dict[str, Any],
    recalculation: dict[str, Any],
) -> dict[str, Any]:
    """Return a tree copy with recalculated scores on top-level action nodes."""
    updated_tree = copy.deepcopy(search_tree)
    scores_by_action = {
        row.get("action"): row["score"]
        for row in recalculation.get("candidates") or []
        if row.get("action") is not None
    }
    for child in updated_tree.get("children") or []:
        score = scores_by_action.get(child.get("action"))
        if score is not None:
            child["finalScore"] = score
            child["compositeScore"] = score
    return updated_tree
