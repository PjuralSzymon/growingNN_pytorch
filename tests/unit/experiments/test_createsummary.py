"""Unit tests for CIFAR-10 grid summary collection from disk."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_PATH = _REPO_ROOT / "experiments" / "createsummary.py"


def _load_createsummary_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("createsummary", _EXPERIMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_hyperparameters() -> dict[str, object]:
    return {
        "generations": 20,
        "epochs": 30,
        "batch_size": 64,
        "lr_alpha": 0.01,
        "simulation_time": 750.0,
        "simulation_epochs": 15,
        "simulation_set_size": 2000,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.15,
        "augmentation_factor": 0.5,
        "model_channels": 32,
        "model_hidden_dim": 256,
    }


def _write_run_history(run_dir: Path, *, train_acc: list[float], val_acc: list[float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "param_count": [100, 120],
        },
        run_dir / "train_cifar10_history.pt",
    )


def _write_board_action(
    run_dir: Path,
    *,
    generation: int,
    action_type: str,
    train_acc_before: float,
    train_acc_after: float,
) -> None:
    board_dir = run_dir / "board"
    simulations_dir = board_dir / "simulations"
    generations_dir = board_dir / "generations"
    metrics_dir = board_dir / "metrics"
    simulations_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    simulation = {
        "generation": generation,
        "actionChosen": f"( {action_type}: ['hidden'] )",
        "candidates": [{"name": action_type, "chosen": True, "action": f"( {action_type}: ['hidden'] )"}],
    }
    (simulations_dir / f"simulation_gen_{generation}.json").write_text(
        json.dumps(simulation),
        encoding="utf-8",
    )
    (generations_dir / f"generation_{generation}.json").write_text(
        json.dumps({"generation": generation, "finalTrainAcc": train_acc_before}),
        encoding="utf-8",
    )
    (metrics_dir / "training.json").write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "generation": generation + 1,
                        "epochInGeneration": 0,
                        "trainAcc": train_acc_after,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_build_hyperparameter_folder_name_round_trips_with_parser():
    """
    build_hyperparameter_folder_name and parse_hyperparameters_from_folder_name
    should recover the same hyperparameter values.
    """
    # Arrange
    module = _load_createsummary_module()
    hyperparameters = _sample_hyperparameters()

    # Act
    folder_name = module.build_hyperparameter_folder_name(hyperparameters)
    parsed = module.parse_hyperparameters_from_folder_name(folder_name)

    # Assert
    assert parsed == hyperparameters


def test_parse_hyperparameters_from_folder_name_supports_legacy_aug_token_without_f():
    """
    parse_hyperparameters_from_folder_name should accept older folder names that used _aug.
    """
    # Arrange
    module = _load_createsummary_module()
    folder_name = "g5_ep30_bs64_lr0.01_simt150.0_sime15_simsz2000_tgt0.9_wacc1.0_wcw0.5_aug0.5_ch32_hd256"

    # Act
    parsed = module.parse_hyperparameters_from_folder_name(folder_name)

    # Assert
    assert parsed is not None
    assert parsed["generations"] == 5
    assert parsed["augmentation_factor"] == 0.5


def test_collect_run_results_loads_history_from_seed_folders(tmp_path):
    """
    collect_run_results should scan hyperparameter_folder_name/seed_N folders and load history.
    """
    # Arrange
    module = _load_createsummary_module()
    hyperparameters = _sample_hyperparameters()
    folder_name = module.build_hyperparameter_folder_name(hyperparameters)
    run_dir = tmp_path / folder_name / "seed_0"
    run_dir.mkdir(parents=True)
    torch.save(
        {"val_acc": [0.5, 0.7], "param_count": [100, 120]},
        run_dir / module.HISTORY_FILENAME,
    )

    # Act
    results = module.collect_run_results(tmp_path)

    # Assert
    assert len(results) == 1
    assert results[0].seed == 0
    assert results[0].best_val_acc == 0.7
    assert results[0].params_before == 100
    assert results[0].params_after == 120
    assert results[0].architecture_changed is True


def test_load_completed_run_returns_none_for_missing_or_incomplete_dirs(tmp_path):
    """
    load_completed_run should return None for a missing dir or one without history.
    """
    # Arrange
    module = _load_createsummary_module()
    hyperparameters = _sample_hyperparameters()
    folder_name = module.build_hyperparameter_folder_name(hyperparameters)
    missing_dir = module.run_dir_for_seed(tmp_path, folder_name, 0)
    incomplete_dir = module.run_dir_for_seed(tmp_path, folder_name, 1)
    incomplete_dir.mkdir(parents=True)

    # Act
    missing_result = module.load_completed_run(
        missing_dir,
        hyperparameters=hyperparameters,
        hyperparameter_folder_name=folder_name,
        seed=0,
    )
    incomplete_result = module.load_completed_run(
        incomplete_dir,
        hyperparameters=hyperparameters,
        hyperparameter_folder_name=folder_name,
        seed=1,
    )

    # Assert
    assert missing_result is None
    assert incomplete_result is None


def test_run_dir_for_seed_builds_expected_path(tmp_path):
    """
    run_dir_for_seed should place runs under runs_root/<config>/seed_<N>.
    """
    # Arrange
    module = _load_createsummary_module()

    # Act
    run_dir = module.run_dir_for_seed(tmp_path, "config_a", 3)

    # Assert
    assert run_dir == tmp_path / "config_a" / "seed_3"


def test_parse_hyperparameters_from_folder_name_accepts_bare_aug_token_without_value():
    """
    parse_hyperparameters_from_folder_name should accept legacy folder names with bare _aug.
    """
    # Arrange
    module = _load_createsummary_module()
    folder_name = "g10_ep30_bs64_lr0.01_simt500.0_sime15_simsz2000_tgt0.9_wacc1.0_wcw0.2_aug_ch32_hd256"

    # Act
    parsed = module.parse_hyperparameters_from_folder_name(folder_name)

    # Assert
    assert parsed is not None
    assert "augmentation_factor" not in parsed


def test_write_grid_summary_skips_runs_missing_optional_param(tmp_path):
    """
    write_grid_summary should analyze augmentation_factor only for runs that recorded it.
    """
    # Arrange
    module = _load_createsummary_module()
    base_hyperparameters = {
        "generations": 20,
        "epochs": 30,
        "batch_size": 64,
        "lr_alpha": 0.01,
        "simulation_time": 750.0,
        "simulation_epochs": 15,
        "simulation_set_size": 2000,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.15,
        "model_channels": 32,
        "model_hidden_dim": 256,
    }
    results = [
        module.RunResult(
            hyperparameters=dict(base_hyperparameters, augmentation_factor=0.0),
            hyperparameter_folder_name="with_augf0",
            seed=0,
            run_dir=tmp_path / "with_augf0",
            best_val_acc=0.60,
            final_val_acc=0.60,
            params_before=100,
            params_after=100,
            architecture_changed=False,
        ),
        module.RunResult(
            hyperparameters=dict(base_hyperparameters, augmentation_factor=0.5),
            hyperparameter_folder_name="with_augf0.5",
            seed=0,
            run_dir=tmp_path / "with_augf0.5",
            best_val_acc=0.70,
            final_val_acc=0.70,
            params_before=100,
            params_after=100,
            architecture_changed=False,
        ),
        module.RunResult(
            hyperparameters=dict(base_hyperparameters),
            hyperparameter_folder_name="legacy_aug",
            seed=0,
            run_dir=tmp_path / "legacy_aug",
            best_val_acc=0.80,
            final_val_acc=0.80,
            params_before=100,
            params_after=100,
            architecture_changed=False,
        ),
    ]
    summary_path = tmp_path / "summary.txt"

    # Act
    module.write_grid_summary(results, summary_path, allowed_output_root=tmp_path)
    text = summary_path.read_text(encoding="utf-8")

    # Assert
    assert "augmentation_factor:" in text
    assert "  0.0: mean=0.6000 (n=1)" in text
    assert "  0.5: mean=0.7000 (n=1)" in text
    assert "augmentation_factor: spread=0.1000" in text


def test_write_grid_summary_only_reports_varying_parameters(tmp_path):
    """
    write_grid_summary should include sensitivity only for hyperparameters that vary across runs.
    """
    # Arrange
    module = _load_createsummary_module()
    base_hyperparameters = {
        "generations": 20,
        "epochs": 30,
        "batch_size": 64,
        "lr_alpha": 0.01,
        "simulation_time": 750.0,
        "simulation_epochs": 15,
        "simulation_set_size": 2000,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.15,
        "model_channels": 32,
        "model_hidden_dim": 256,
    }
    results = []
    for aug, acc in ((0.0, 0.60), (0.5, 0.70)):
        hyperparameters = dict(base_hyperparameters, augmentation_factor=aug)
        results.append(
            module.RunResult(
                hyperparameters=hyperparameters,
                hyperparameter_folder_name=module.build_hyperparameter_folder_name(hyperparameters),
                seed=0,
                run_dir=tmp_path / f"seed_{aug}",
                best_val_acc=acc,
                final_val_acc=acc,
                params_before=100,
                params_after=100,
                architecture_changed=False,
            )
        )
    summary_path = tmp_path / "summary.txt"

    # Act
    module.write_grid_summary(results, summary_path, allowed_output_root=tmp_path)
    text = summary_path.read_text(encoding="utf-8")

    # Assert
    assert "augmentation_factor:" in text
    assert "batch_size:" not in text
    assert "Suggested tuning priority" in text
    assert "augmentation_factor: spread=0.1000" in text


def test_write_grid_summary_rejects_path_outside_allowed_root(tmp_path):
    """
    write_grid_summary should reject summary paths outside the allowed output root.
    """
    # Arrange
    module = _load_createsummary_module()
    hyperparameters = _sample_hyperparameters()
    results = [
        module.RunResult(
            hyperparameters=hyperparameters,
            hyperparameter_folder_name=module.build_hyperparameter_folder_name(hyperparameters),
            seed=0,
            run_dir=tmp_path / "run",
            best_val_acc=0.7,
            final_val_acc=0.7,
            params_before=100,
            params_after=100,
            architecture_changed=False,
        )
    ]
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_path = tmp_path / "outside" / "summary.txt"

    # Act / Assert
    with pytest.raises(ValueError, match="inside"):
        module.write_grid_summary(results, outside_path, allowed_output_root=allowed_root)


def test_try_claim_run_grants_exclusive_access(tmp_path):
    """
    try_claim_run should allow only one live claim per run directory.
    """
    # Arrange
    module = _load_createsummary_module()
    run_dir = tmp_path / "seed_0"

    # Act / Assert
    assert module.try_claim_run(run_dir) is True
    assert module.try_claim_run(run_dir) is False
    module.release_run_claim(run_dir)
    assert module.try_claim_run(run_dir) is True


def test_try_claim_run_reclaims_stale_lock(tmp_path):
    """
    try_claim_run should reclaim a lock left by a non-running PID.
    """
    # Arrange
    module = _load_createsummary_module()
    run_dir = tmp_path / "seed_1"
    run_dir.mkdir(parents=True)
    (run_dir / module.RUN_LOCK_FILENAME).write_text("999999999\n", encoding="utf-8")

    # Act / Assert
    assert module.try_claim_run(run_dir) is True


def test_load_board_action_executions_reads_simulation_and_generation_metrics(tmp_path):
    """
    load_board_action_executions should parse chosen actions and train-acc deltas from board files.
    """
    # Arrange
    module = _load_createsummary_module()
    run_dir = tmp_path / "seed_0"
    _write_board_action(
        run_dir,
        generation=0,
        action_type="Add Seq Dropout Layer Action",
        train_acc_before=0.20,
        train_acc_after=0.25,
    )

    # Act
    executions = module.load_board_action_executions(run_dir)

    # Assert
    assert len(executions) == 1
    assert executions[0].action_type == "Add Seq Dropout Layer Action"
    assert executions[0].train_acc_before == 0.20
    assert executions[0].train_acc_after == 0.25
    assert executions[0].train_acc_delta == pytest.approx(0.05)


def test_normalize_action_type_merges_seq_linear_into_seq_layer():
    """
    normalize_action_type should treat Add Seq Linear Layer Action as Add Seq Layer Action.
    """
    # Arrange
    module = _load_createsummary_module()

    # Act
    normalized = module.normalize_action_type("Add Seq Linear Layer Action")

    # Assert
    assert normalized == "Add Seq Layer Action"


def test_write_grid_summary_includes_action_analysis_tables(tmp_path):
    """
    write_grid_summary should append action usage and accuracy tables when board artifacts exist.
    """
    # Arrange
    module = _load_createsummary_module()
    hyperparameters = _sample_hyperparameters()
    folder_name = module.build_hyperparameter_folder_name(hyperparameters)
    run_a = tmp_path / folder_name / "seed_0"
    run_b = tmp_path / folder_name / "seed_1"
    _write_run_history(run_a, train_acc=[0.4, 0.6], val_acc=[0.3, 0.5])
    _write_run_history(run_b, train_acc=[0.5, 0.7], val_acc=[0.4, 0.6])
    _write_board_action(
        run_a,
        generation=0,
        action_type="Add Seq Dropout Layer Action",
        train_acc_before=0.20,
        train_acc_after=0.25,
    )
    _write_board_action(
        run_b,
        generation=0,
        action_type="Add Seq Dropout Layer Action",
        train_acc_before=0.30,
        train_acc_after=0.40,
    )
    _write_board_action(
        run_b,
        generation=1,
        action_type="Add Seq Conv Layer Action",
        train_acc_before=0.50,
        train_acc_after=0.55,
    )
    _write_board_action(
        run_b,
        generation=2,
        action_type="Add Seq Linear Layer Action",
        train_acc_before=0.55,
        train_acc_after=0.56,
    )
    results = [
        module.RunResult(
            hyperparameters=hyperparameters,
            hyperparameter_folder_name=folder_name,
            seed=0,
            run_dir=run_a,
            best_val_acc=0.5,
            final_val_acc=0.5,
            params_before=100,
            params_after=120,
            architecture_changed=True,
        ),
        module.RunResult(
            hyperparameters=hyperparameters,
            hyperparameter_folder_name=folder_name,
            seed=1,
            run_dir=run_b,
            best_val_acc=0.6,
            final_val_acc=0.6,
            params_before=100,
            params_after=120,
            architecture_changed=True,
        ),
    ]
    summary_path = tmp_path / "summary.txt"

    # Act
    module.write_grid_summary(results, summary_path, allowed_output_root=tmp_path)
    text = summary_path.read_text(encoding="utf-8")

    # Assert
    assert "Configs ranked by mean best validation accuracy:" in text
    assert "Action analysis (from board/simulations):" in text
    assert "1. Action usage count:" in text
    assert "Add Seq Dropout Layer Action" in text
    assert "Add Seq Conv Layer Action" in text
    assert "Add Seq Linear Layer Action" not in text
    assert "2. Mean best train accuracy by action type (runs that used the action):" in text
    assert "3. Mean best test accuracy by action type (runs that used the action):" in text
    assert "4. Mean train accuracy change after action execution:" in text
    train_section = text.split("2. Mean best train accuracy by action type (runs that used the action):")[1]
    train_section = train_section.split("3. Mean best test accuracy by action type (runs that used the action):")[0]
    train_lines = [line for line in train_section.splitlines() if line.strip() and not line.startswith("-")]
    train_accs = [float(line.split()[-2]) for line in train_lines[1:]]
    assert train_accs == sorted(train_accs, reverse=True)
