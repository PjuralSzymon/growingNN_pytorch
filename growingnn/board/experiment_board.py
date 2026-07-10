"""Write live experiment artifacts consumed by GrowingNN Board."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch.fx as fx
import torch.nn as nn

import growingnn.core.config as project_config
from growingnn.core.config import RunningConfig
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_SCORE_TERM_LABELS = {
    "weight_acc": "accuracy",
    "weight_loss": "loss",
    "weight_time": "time",
    "weight_countW": "paramCount",
}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2)
    for attempt in range(5):
        try:
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)
            return
        except (PermissionError, OSError):
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


class ExperimentBoard:
    """Single entry point on the growingnn side for board export."""

    def __init__(
        self,
        experiment_dir: Path | str,
        *,
        experiment_id: str | None = None,
        experiment_name: str = "GrowingNN experiment",
        dataset: str = "",
        device: str = "",
    ) -> None:
        self.root = Path(experiment_dir)
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.dataset = dataset
        self.device = device or "cpu"
        self._started_at = time.time()
        self._global_epoch = 0
        self._current_generation = 0
        self._current_epoch_in_generation = 0
        self._status = "running"
        self._training_rows: list[dict[str, Any]] = []
        self._generation_epoch_ranges: list[dict[str, int]] = []
        self._last_simulation: dict[str, Any] | None = None
        self._simulation_actions: dict[int, dict[str, Any]] = {}
        self._config_snapshot: dict[str, Any] = {}
        self.simulation_metrics: dict[str, Any] = {}

        for sub in ("graphs", "generations", "simulations", "metrics"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def bind_config(self, config: RunningConfig) -> None:
        sched = config.simulation_scheduler
        sim_alg = config.simulation_alg
        sim_alg_name = getattr(sim_alg, "__name__", "none") if sim_alg is not None else "none"
        lr_sched = config.lr_scheduler._schedule
        score = config.simulation_score
        self._config_snapshot = {
            "totalGenerations": config.generations,
            "epochsPerGeneration": config.epochs,
            "simulationSetSize": config.simulation_set_size,
            "simulationAlgorithm": sim_alg_name,
            "simulationTimeSec": sched.simulation_time,
            "simulationEpochs": sched.simulation_epochs,
            "simulationSchedulerMode": sched.mode.name,
            "learningRateMode": type(lr_sched).__name__,
            "learningRateAlpha": lr_sched.alpha,
            "scoreWeights": getattr(score, "weights", None) if score is not None else None,
        }

    def on_run_start(self, model: nn.Module | fx.GraphModule, config: RunningConfig) -> None:
        self.bind_config(config)
        self._write_main()
        self.save_graphs(model, generation=0, tag="start")

    def on_generation_start(self, generation: int, model: nn.Module | fx.GraphModule) -> None:
        """Snapshot architecture at the start of a generation (after any prior simulation)."""
        self._current_generation = generation
        self._current_epoch_in_generation = 0
        self.save_graphs(model, generation=generation)
        self._write_main()

    def on_epoch_end(
        self,
        *,
        generation: int,
        epoch_in_generation: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        param_count: int,
    ) -> None:
        self._current_generation = generation
        self._current_epoch_in_generation = epoch_in_generation
        row = {
            "globalEpoch": self._global_epoch,
            "generation": generation,
            "epochInGeneration": epoch_in_generation,
            "trainLoss": train_loss,
            "trainAcc": train_acc,
            "valLoss": val_loss,
            "valAcc": val_acc,
            "lr": lr,
            "paramCount": param_count,
        }
        self._training_rows.append(row)
        self._global_epoch += 1
        self._write_metrics()
        self._write_main()

    def on_generation_end(
        self,
        generation: int,
        model: nn.Module | fx.GraphModule,
        history: dict[str, list[float]],
        param_count: int,
    ) -> None:
        start_global = self._global_epoch - len(history.get("train_loss", []))
        end_global = self._global_epoch - 1
        self._generation_epoch_ranges.append(
            {"generation": generation, "globalEpochStart": max(0, start_global), "globalEpochEnd": max(0, end_global)}
        )
        payload = {
            "generation": generation,
            "lastUpdate": _utc_now(),
            "paramCount": param_count,
            "finalTrainAcc": history["train_acc"][-1] if history["train_acc"] else None,
            "finalValAcc": history["val_acc"][-1] if history["val_acc"] else None,
            "finalTrainLoss": history["train_loss"][-1] if history["train_loss"] else None,
            "finalValLoss": history["val_loss"][-1] if history["val_loss"] else None,
            "graphFull": f"graphs/gen_{generation}_full.pdf",
            "graphSimplified": f"graphs/gen_{generation}_simplified.pdf",
        }
        _safe_write_json(self.root / "generations" / f"generation_{generation}.json", payload)
        self.save_graphs(model, generation=generation)
        self._write_main()

    def on_simulation_finished(
        self,
        generation: int,
        *,
        action: object | None,
        max_depth: int,
        rollouts: int,
        duration_sec: float,
        param_count_before: int,
        param_count_after: int | None = None,
        val_acc_before: float | None = None,
        candidates: list[dict[str, Any]] | None = None,
        search_tree: dict[str, Any] | None = None,
    ) -> None:
        chosen = str(action) if action is not None else None
        chosen_entry = next((c for c in (candidates or []) if c.get("chosen")), None)
        if chosen:
            self._simulation_actions[generation] = {
                "action": chosen,
                "shortLabel": self.action_short_label(chosen),
                "atGlobalEpoch": self._global_epoch,
            }
        if search_tree is None and candidates:
            search_tree = self.search_tree_from_candidates(candidates, rollouts)
        payload = {
            "generation": generation,
            "lastUpdate": _utc_now(),
            "durationSec": round(duration_sec, 3),
            "rollouts": rollouts,
            "maxDepth": max_depth,
            "actionsAnalyzed": len(candidates) if candidates else rollouts,
            "paramCountBefore": param_count_before,
            "paramCountAfter": param_count_after,
            "valAccBefore": val_acc_before,
            "actionChosen": chosen,
            "scoreChosen": chosen_entry.get("ucbScore") if chosen_entry else None,
            "scoreWeights": self._config_snapshot.get("scoreWeights"),
            "startingStructure": {
                "totalParams": param_count_before,
                "accuracy": val_acc_before,
            },
            "candidates": candidates or [],
            "searchTree": search_tree,
        }
        _safe_write_json(self.root / "simulations" / f"simulation_gen_{generation}.json", payload)
        self._last_simulation = {
            "generation": generation,
            "actionsAnalyzed": payload["actionsAnalyzed"],
            "treeDepth": max_depth,
            "executionTimeSec": payload["durationSec"],
            "actionChosen": chosen,
            "actionShortLabel": self.action_short_label(chosen) if chosen else None,
            "scoreChosen": payload["scoreChosen"],
        }
        self._write_main()

    def note_simulation_graph_saved(self, generation: int) -> None:
        """Call after gen_N_simulation PDFs are written so main.json can reference them."""
        sim_path = self.root / "simulations" / f"simulation_gen_{generation}.json"
        if sim_path.is_file():
            data = json.loads(sim_path.read_text(encoding="utf-8"))
            files = dict(data.get("files") or {})
            files.update(
                {
                    "simulationGraphPdf": f"graphs/gen_{generation}_simulation_simplified.pdf",
                    "simulationGraphFull": f"graphs/gen_{generation}_simulation_full.pdf",
                }
            )
            data["files"] = files
            _safe_write_json(sim_path, data)
        self._write_main()

    def on_run_finished(self, status: str = "completed") -> None:
        self._status = status
        self._write_main()

    def save_graphs(
        self,
        model: nn.Module | fx.GraphModule,
        *,
        generation: int,
        tag: str | None = None,
    ) -> None:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        suffix = f"gen_{generation}" if tag is None else tag
        full = self.root / "graphs" / f"{suffix}_full.pdf"
        simple = self.root / "graphs" / f"{suffix}_simplified.pdf"
        full.parent.mkdir(parents=True, exist_ok=True)
        simple.parent.mkdir(parents=True, exist_ok=True)
        draw_torch_fx_graph(gm, str(full.with_suffix("")), fmt="pdf")
        draw_filtered_fx_graph(gm, str(simple.with_suffix("")), fmt="pdf")

    def save_candidate_graph(
        self,
        generation: int,
        index: int,
        model: nn.Module | fx.GraphModule,
    ) -> str:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        rel = f"graphs/candidates/gen_{generation}_cand_{index}_simplified.pdf"
        out = self.root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_filtered_fx_graph(gm, str(out.with_suffix("")), fmt="pdf")
        return rel

    @staticmethod
    def _raw_score_terms(
        *,
        val_acc: float,
        val_loss: float,
        param_count: int,
        train_time_sec: float = 0.0,
    ) -> dict[str, float]:
        return {
            "weight_acc": val_acc,
            "weight_loss": min(1.0 / (max(val_loss, 1e-8) + 1), 1.0),
            "weight_time": 1.0 / (project_config.TIME_EFFICIENCY_WEIGHT * train_time_sec + 1.0),
            "weight_countW": 1.0 / (float(param_count) * project_config.WEIGHT_COUNT_WEIGHT + 1.0),
        }

    def build_score_breakdown(
        self,
        config: RunningConfig,
        *,
        val_acc: float,
        val_loss: float,
        param_count: int,
        train_time_sec: float = 0.0,
    ) -> dict[str, Any] | None:
        """Per-term score breakdown for board export (uses config.simulation_score weights)."""
        score_fn = config.simulation_score
        if score_fn is None:
            return None
        raw = self._raw_score_terms(
            val_acc=val_acc,
            val_loss=val_loss,
            param_count=param_count,
            train_time_sec=train_time_sec,
        )
        terms: dict[str, Any] = {}
        total = 0.0
        for key, weight in score_fn.weights.items():
            if weight <= 0.0:
                continue
            value = raw[key]
            terms[_SCORE_TERM_LABELS[key]] = {
                "weight": weight,
                "raw": round(value, 6),
                "weighted": round(weight * value, 6),
            }
            total += weight * value
        divisor = score_fn.weight_sum()
        composite = total / divisor if divisor else 0.0
        return {
            "composite": round(composite, 6),
            "valAcc": round(val_acc, 6),
            "valLoss": round(val_loss, 6),
            "paramCount": param_count,
            "trainTimeSec": round(train_time_sec, 3),
            "terms": terms,
            "weights": dict(score_fn.weights),
        }

    def score_breakdown_from_metrics(
        self,
        metrics: dict[str, Any],
        config: RunningConfig,
    ) -> dict[str, Any] | None:
        """Build board scoreBreakdown from simulation_metrics filled by SimulationScore.score."""
        score_fn = config.simulation_score
        if not score_fn or not metrics:
            return None
        labels = {
            "weight_acc": "accuracy",
            "weight_loss": "loss",
            "weight_time": "time",
            "weight_countW": "paramCount",
        }
        terms: dict[str, Any] = {}
        for key, label in labels.items():
            raw = metrics.get(f"{key}_score")
            weight = metrics.get(f"{key}_weight")
            if raw is None or weight is None:
                continue
            terms[label] = {
                "weight": weight,
                "raw": raw,
                "weighted": metrics.get(f"{key}_weighted", weight * raw),
            }
        return {
            "composite": metrics.get("composite_score"),
            "terms": terms,
            "weights": dict(score_fn.weights),
        }

    def format_candidate_row(
        self,
        partial: dict[str, Any],
        config: RunningConfig,
        *,
        generation: int,
        index: int,
        model: nn.Module | fx.GraphModule | None = None,
        chosen: bool = False,
    ) -> dict[str, Any]:
        """Enrich a simulation candidate with board fields from rollout metrics."""
        action_str = partial.get("action")
        sim_metrics = partial.get("simulationMetrics") or {}
        breakdown = self.score_breakdown_from_metrics(sim_metrics, config)
        composite = sim_metrics.get("composite_score") or partial.get("compositeScore")
        score = partial.get("score") if partial.get("score") is not None else composite
        val_acc = sim_metrics.get("weight_acc_score")
        val_loss = sim_metrics.get("weight_loss_score")
        param_count = partial.get("paramCount") or partial.get("paramsAfter")
        row: dict[str, Any] = {
            "action": action_str,
            "name": self.action_short_label(str(action_str) if action_str else None),
            "score": score,
            "compositeScore": composite,
            "visits": partial.get("visits", 1),
            "ucbScore": partial.get("ucbScore", score),
            "paramCount": param_count,
            "accuracyAfter": val_acc,
            "valLossAfter": val_loss,
            "scoreBreakdown": breakdown,
            "structure": self.structure_summary(model) if model is not None else partial.get("structure"),
            "chosen": chosen,
        }
        if model is not None:
            row["graphPdf"] = self.save_candidate_graph(generation, index, model)
        return row

    def greedy_candidate_row(
        self,
        action: object,
        model: nn.Module | fx.GraphModule,
        score: float,
        config: RunningConfig,
        index: int,
    ) -> dict[str, Any]:
        return self.format_candidate_row(
            {
                "action": str(action),
                "score": score,
                "visits": 1,
                "ucbScore": score,
                "simulationMetrics": dict(self.simulation_metrics),
                "paramCount": GraphStructureQuery.get_amount_of_parameters(model),
            },
            config,
            generation=self._current_generation,
            index=index,
            model=model,
        )

    def finish_greedy_simulation(
        self,
        *,
        action: object | None,
        rollouts: int,
        duration_sec: float,
        param_count_before: int,
        candidates: list[dict[str, Any]],
    ) -> None:
        chosen = str(action) if action is not None else None
        for row in candidates:
            row["chosen"] = row.get("action") == chosen
        self.on_simulation_finished(
            self._current_generation,
            action=action,
            max_depth=0,
            rollouts=rollouts,
            duration_sec=duration_sec,
            param_count_before=param_count_before,
            candidates=candidates,
            search_tree=self.search_tree_from_candidates(candidates, rollouts),
        )

    @staticmethod
    def action_short_label(action_str: str | None) -> str:
        if not action_str:
            return "—"
        match = re.search(r"\(\s*([^:(]+)", action_str)
        return match.group(1).strip() if match else action_str[:48]

    @staticmethod
    def structure_summary(model: nn.Module | fx.GraphModule) -> dict[str, Any]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        modules = [n.target for n in gm.graph.nodes if n.op == "call_module"]
        hidden = GraphStructureQuery.get_all_hidden_modules(gm)
        return {
            "moduleCount": len(modules),
            "modules": [str(m) for m in modules[:12]],
            "hiddenModuleCount": len(hidden),
            "hiddenModules": hidden[:8],
            "paramCount": GraphStructureQuery.get_amount_of_parameters(gm),
        }

    def _build_generation_timeline(self) -> list[dict[str, Any]]:
        epg = int(self._config_snapshot.get("epochsPerGeneration") or 1)
        total_gen = int(self._config_snapshot.get("totalGenerations") or 1)
        timeline: list[dict[str, Any]] = []
        for g in range(total_gen):
            start = g * epg
            end = start + epg
            gen_epochs = [r for r in self._training_rows if r["generation"] == g]
            epoch_values = [round(r["valAcc"], 4) for r in gen_epochs]
            current_epoch = None
            current_epoch_index = None
            if g == self._current_generation:
                current_epoch = self._current_epoch_in_generation
                if gen_epochs:
                    current_epoch_index = min(self._current_epoch_in_generation, len(epoch_values) - 1)
            action = self._simulation_actions.get(g)
            timeline.append(
                {
                    "generation": g,
                    "startEpoch": start,
                    "endEpoch": end,
                    "currentEpoch": current_epoch,
                    "currentEpochIndex": current_epoch_index,
                    "isCurrent": g == self._current_generation,
                    "epochValues": epoch_values,
                    "actionExecuted": action,
                }
            )
        return timeline

    def _write_metrics(self) -> None:
        _safe_write_json(
            self.root / "metrics" / "training.json",
            {"lastUpdate": _utc_now(), "epochs": self._training_rows},
        )

    def _resolve_architecture_graphs(self) -> tuple[str, str]:
        """Newest architecture PDF on disk (simulation snapshot beats plain gen graph)."""
        g = self._current_generation
        stems: list[str] = []
        for gen in range(g, -1, -1):
            stems.append(f"graphs/gen_{gen}_simulation")
            stems.append(f"graphs/gen_{gen}")
        stems.append("graphs/start")
        for stem in stems:
            simple = self.root / f"{stem}_simplified.pdf"
            if not simple.is_file():
                continue
            full = self.root / f"{stem}_full.pdf"
            return (
                full.relative_to(self.root).as_posix() if full.is_file() else simple.relative_to(self.root).as_posix(),
                simple.relative_to(self.root).as_posix(),
            )
        return "graphs/start_full.pdf", "graphs/start_simplified.pdf"

    def _write_main(self) -> None:
        total_epochs = self._config_snapshot.get("totalGenerations", 0) * self._config_snapshot.get(
            "epochsPerGeneration", 0
        )
        main = {
            "experimentId": self.experiment_id,
            "experimentName": self.experiment_name,
            "lastUpdate": _utc_now(),
            "experimentStartedOn": datetime.fromtimestamp(self._started_at, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "trainingTimeElapsedSec": int(time.time() - self._started_at),
            "status": self._status,
            "dataset": self.dataset,
            "device": self.device,
            "trainingParameters": {
                **self._config_snapshot,
                "currentGeneration": self._current_generation,
                "currentEpoch": self._current_epoch_in_generation,
                "totalEpochs": total_epochs,
                "completedGlobalEpochs": self._global_epoch,
            },
            "generationTimeline": self._build_generation_timeline(),
            "lastSimulation": self._last_simulation,
            "graphs": dict(zip(["latestFull", "latestSimplified"], self._resolve_architecture_graphs())),
        }
        _safe_write_json(self.root / "main.json", main)

    def mcts_candidates_from_root(self, root, config: RunningConfig) -> list[dict[str, Any]]:
        """Build candidate list from MCTS root children (call before root.kill())."""
        best = root.get_best_child()
        best_action = str(best.action) if best is not None and best.action is not None else None
        generation = self._current_generation
        out: list[dict[str, Any]] = []
        for index, child in enumerate(root.child_nodes):
            if child.visit_counter == 0:
                ucb = float("inf")
            elif project_config.MCTS_UCB1_USE_SQRT:
                n = child.visit_counter
                ucb = child.value / n + project_config.MCTS_UCB1_C * math.sqrt(
                    math.log(max(root.visit_counter, 1)) / n
                )
            else:
                ucb = child.value + project_config.MCTS_UCB1_C * (
                    math.log(max(root.visit_counter, 1)) / child.visit_counter
                )
            action_str = str(child.action) if child.action is not None else None
            visits = child.visit_counter
            rollout_score = child.value / visits if visits else child.value
            sim_metrics = dict(getattr(child, "rollout_metrics", None) or {})
            child_gm = child.traced.gm if getattr(child, "traced", None) is not None else child.model
            partial = {
                "action": action_str,
                "score": rollout_score,
                "visits": visits,
                "ucbScore": ucb,
                "simulationMetrics": sim_metrics,
                "paramCount": GraphStructureQuery.get_amount_of_parameters(child_gm) if child_gm else None,
            }
            out.append(
                self.format_candidate_row(
                    partial,
                    config,
                    generation=generation,
                    index=index,
                    model=child_gm,
                    chosen=action_str == best_action,
                )
            )
        out.sort(key=lambda row: row.get("ucbScore") or 0.0, reverse=True)
        return out

    @staticmethod
    def _mcts_ucb_score(parent: Any, child: Any) -> float:
        if child.visit_counter == 0:
            return float("inf")
        if project_config.MCTS_UCB1_USE_SQRT:
            n = child.visit_counter
            return child.value / n + project_config.MCTS_UCB1_C * math.sqrt(
                math.log(max(parent.visit_counter, 1)) / n
            )
        return child.value + project_config.MCTS_UCB1_C * (
            math.log(max(parent.visit_counter, 1)) / child.visit_counter
        )

    def mcts_search_tree_from_root(
        self,
        root: Any,
        config: RunningConfig,
        *,
        chosen_node: Any | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """Serialize visited MCTS nodes with depth and final rollout score (call before root.kill())."""

        def walk(node: Any, node_id: str, depth: int) -> dict[str, Any]:
            visits = int(node.visit_counter)
            mean_score = node.value / visits if visits else float(node.value)
            action_str = str(node.action) if node.action is not None else None
            ucb_score = None
            if node.parent is not None:
                ucb = self._mcts_ucb_score(node.parent, node)
                ucb_score = None if ucb == float("inf") else round(ucb, 6)
            sim_metrics = dict(getattr(node, "rollout_metrics", None) or {})
            composite = sim_metrics.get("composite_score")
            final_score = composite if composite is not None else (mean_score if visits else None)
            analyzed_children = [child for child in node.child_nodes if child.visit_counter > 0]
            child_rows = [
                walk(child, f"{node_id}-{index}", depth + 1)
                for index, child in enumerate(analyzed_children)
            ]
            max_depth_below = depth
            for child_row in child_rows:
                max_depth_below = max(max_depth_below, child_row["maxDepthBelow"])
            row: dict[str, Any] = {
                "id": node_id,
                "action": action_str,
                "name": self.action_short_label(action_str) if action_str else "root",
                "depth": depth,
                "visits": visits,
                "totalValue": round(float(node.value), 6),
                "meanScore": round(float(mean_score), 6) if visits else None,
                "ucbScore": ucb_score,
                "compositeScore": composite,
                "finalScore": round(float(final_score), 6) if final_score is not None else None,
                "maxDepthBelow": max_depth_below,
                "accuracyAfter": sim_metrics.get("weight_acc_score"),
                "chosen": chosen_node is not None and node is chosen_node,
                "children": child_rows,
            }
            return row

        tree = walk(root, "0", 0)
        tree["simMaxDepth"] = max_depth
        return tree

    @staticmethod
    def search_tree_from_candidates(
        candidates: list[dict[str, Any]] | None,
        rollouts: int,
    ) -> dict[str, Any]:
        """Flat search tree for greedy/random runs (one root, one level of tried actions)."""
        children: list[dict[str, Any]] = []
        for index, row in enumerate(candidates or []):
            composite = row.get("compositeScore")
            mean = row.get("score")
            final_score = composite if composite is not None else mean
            children.append(
                {
                    "id": f"0-{index}",
                    "action": row.get("action"),
                    "name": row.get("name") or ExperimentBoard.action_short_label(row.get("action")),
                    "depth": 1,
                    "visits": row.get("visits", 1),
                    "totalValue": mean,
                    "meanScore": mean,
                    "ucbScore": row.get("ucbScore"),
                    "compositeScore": composite,
                    "finalScore": final_score,
                    "maxDepthBelow": 1,
                    "accuracyAfter": row.get("accuracyAfter"),
                    "chosen": bool(row.get("chosen")),
                    "children": [],
                }
            )
        return {
            "id": "0",
            "name": "root",
            "depth": 0,
            "visits": rollouts,
            "totalValue": None,
            "meanScore": None,
            "ucbScore": None,
            "compositeScore": None,
            "finalScore": None,
            "maxDepthBelow": 1 if children else 0,
            "simMaxDepth": 1 if children else 0,
            "accuracyAfter": None,
            "chosen": False,
            "children": children,
        }
