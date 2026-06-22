"""Unit tests for CIFAR-10 grid summary collection from disk."""

import importlib.util
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
