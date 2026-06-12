"""Unit tests for ExperimentBoard file export."""

from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn

from growingnn.board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.l2(self.l1(x))


def test_experiment_board_writes_main_and_metrics(tmp_path: Path):
    """
    ExperimentBoard should create main.json and append epoch rows to metrics/training.json.
    """
    # Arrange
    board = ExperimentBoard(tmp_path, experiment_name="test-run")
    cfg = RunningConfig(
        generations=2,
        epochs=1,
        lr_scheduler=LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        enable_experiment_board=True,
        experiment_board=board,
    )
    model = _TinyNet()

    # Act
    board.on_run_start(model, cfg)
    board.on_epoch_end(
        generation=0,
        epoch_in_generation=0,
        train_loss=1.0,
        train_acc=0.5,
        val_loss=0.9,
        val_acc=0.6,
        lr=0.01,
        param_count=30,
    )
    board.on_run_finished()

    # Assert
    main = json.loads((tmp_path / "main.json").read_text(encoding="utf-8"))
    assert main["experimentName"] == "test-run"
    assert main["status"] == "completed"
    assert main["trainingParameters"]["completedGlobalEpochs"] == 1
    metrics = json.loads((tmp_path / "metrics" / "training.json").read_text(encoding="utf-8"))
    assert len(metrics["epochs"]) == 1
    assert metrics["epochs"][0]["valAcc"] == 0.6


def test_resolve_architecture_graphs_prefers_existing_pdf(tmp_path: Path):
    """
    _resolve_architecture_graphs should not point at gen_N when only gen_{N-1} PDFs exist yet.
    """
    # Arrange
    board = ExperimentBoard(tmp_path)
    board._current_generation = 1
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    (graphs_dir / "gen_0_simulation_simplified.pdf").write_bytes(b"%PDF-1.4")
    (graphs_dir / "gen_0_simulation_full.pdf").write_bytes(b"%PDF-1.4")

    # Act
    full, simple = board._resolve_architecture_graphs()

    # Assert
    assert simple == "graphs/gen_0_simulation_simplified.pdf"
    assert full == "graphs/gen_0_simulation_full.pdf"


def test_build_score_breakdown_uses_config_weights(tmp_path: Path):
    """
    build_score_breakdown should compute composite and per-term values from rollout metrics.
    """
    # Arrange
    from growingnn.simulation.score_functions.simulation_score import SimulationScore

    board = ExperimentBoard(tmp_path)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.5),
    )

    # Act
    breakdown = board.build_score_breakdown(
        cfg,
        val_acc=0.42,
        val_loss=1.2,
        param_count=1000,
        train_time_sec=2.0,
    )

    # Assert
    assert breakdown is not None
    assert breakdown["valAcc"] == 0.42
    assert "accuracy" in breakdown["terms"]
    assert breakdown["terms"]["accuracy"]["weighted"] == 0.42


def test_search_tree_from_candidates_builds_root_and_children(tmp_path: Path):
    """
    search_tree_from_candidates should expose one root node and per-action child scores.
    """
    # Arrange
    board = ExperimentBoard(tmp_path)
    candidates = [
        {
            "action": " ( Add Layer Action: [] ) ",
            "name": "Add Layer Action",
            "visits": 3,
            "score": 0.5,
            "ucbScore": 1.2,
            "compositeScore": 0.48,
            "accuracyAfter": 0.4,
            "chosen": True,
        }
    ]

    # Act
    tree = board.search_tree_from_candidates(candidates, rollouts=5)

    # Assert
    assert tree["name"] == "root"
    assert tree["visits"] == 5
    assert len(tree["children"]) == 1
    child = tree["children"][0]
    assert child["name"] == "Add Layer Action"
    assert child["depth"] == 1
    assert child["finalScore"] == 0.48
    assert child["ucbScore"] == 1.2
    assert child["compositeScore"] == 0.48
    assert child["chosen"] is True


def test_mcts_search_tree_from_root_exports_nested_nodes(tmp_path: Path):
    """
    mcts_search_tree_from_root should serialize visited nodes at every depth with final scores.
    """

    # Arrange
    class _Node:
        def __init__(self, parent=None, action=None, value=0.0, visits=0, children=None, metrics=None):
            self.parent = parent
            self.action = action
            self.value = value
            self.visit_counter = visits
            self.child_nodes = children or []
            self.rollout_metrics = metrics

    grandchild = _Node(
        action=" ( Add Layer Action: [] ) ",
        value=0.7,
        visits=1,
        metrics={"composite_score": 0.7},
    )
    child = _Node(
        action=" ( Delete Neurons Action: [] ) ",
        value=1.6,
        visits=2,
        children=[grandchild],
        metrics={"composite_score": 0.65},
    )
    unvisited = _Node(action=" ( Remove Layer Action: [] ) ", visits=0)
    root = _Node(value=3.0, visits=5, children=[child, unvisited])
    child.parent = root
    grandchild.parent = child
    unvisited.parent = root
    board = ExperimentBoard(tmp_path)

    # Act
    tree = board.mcts_search_tree_from_root(
        root, RunningConfig(generations=1, epochs=1), chosen_node=child, max_depth=2
    )

    # Assert
    assert tree["name"] == "root"
    assert tree["depth"] == 0
    assert tree["simMaxDepth"] == 2
    assert tree["maxDepthBelow"] == 2
    assert len(tree["children"]) == 1
    child_row = tree["children"][0]
    assert child_row["depth"] == 1
    assert child_row["finalScore"] == 0.65
    assert child_row["chosen"] is True
    assert len(child_row["children"]) == 1
    grandchild_row = child_row["children"][0]
    assert grandchild_row["depth"] == 2
    assert grandchild_row["finalScore"] == 0.7
    assert grandchild_row["children"] == []


def test_action_short_label_extracts_action_name():
    """
    action_short_label should return the human-readable action class name from repr strings.
    """
    # Arrange
    raw = " ( Delete Neurons Action: ['hidden', 0.1] ) "

    # Act
    label = ExperimentBoard.action_short_label(raw)

    # Assert
    assert label == "Delete Neurons Action"
