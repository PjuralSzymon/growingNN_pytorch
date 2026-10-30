"""Unit tests for the Experiment 008 generation-only recipe."""

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

import growingnn.core.config as project_config
from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from experiments.train_cifar10 import _BatchLoader, _CudaCIFAR
from experiments.train_cifar10_exp008_generation_only import (
    CHANNELS,
    EPOCHS,
    GENERATIONS,
    HIDDEN,
    PHASE3_SIMULATION,
    PHASE4_ROLLOUT_DEPTH,
    PHASE4_SIMULATIONS,
    PHASE5_CLEAN_PREFIX,
    PHASE5_ROLLOUT_DEPTH,
    PHASE5_SIMULATIONS,
    PHASE6_EPOCHS,
    PHASE6_ROLLOUT_DEPTH,
    PHASE6_SIMULATIONS,
    PHASE6_SIZES,
    PHASE7_CLEAN_PREFIX,
    PHASE7_ROLLOUT_DEPTH,
    PHASE7_SIMULATION,
    PHASE7_TRAIN_EPOCHS,
    PHASE8_CLEAN_PREFIX,
    PHASE8_GENERATIONS,
    PHASE8_ROLLOUT_DEPTH,
    PHASE8_SIMULATION,
    PHASE10_ARMS,
    PHASE10_CLEAN_PREFIX,
    PHASE10_EPOCHS,
    PHASE10_GENERATIONS,
    PHASE10_ROLLOUT_DEPTH,
    PHASE10_SEEDS,
    PHASE10_SIMULATION,
    PHASE11_ARMS,
    PHASE11_CLEAN_PREFIX,
    PHASE11_EPOCHS,
    PHASE11_GENERATIONS,
    PHASE11_ROLLOUT_DEPTH,
    PHASE11_SEEDS,
    PHASE11_SHORT_GENERATIONS,
    PHASE11_SIMULATION,
    SEEDS,
    CosineDecayScheduler,
    apply_action_group,
    apply_phase10_switches,
    apply_phase11_arm,
    apply_simulation_budget,
    history_from_generations,
    parse_args,
    phase11_settings,
    prefix_clean_loader,
    rollout_depth,
    with_cutout,
)


def test_parse_args_defaults_to_the_three_experiment_seeds():
    """
    Experiment 008 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args([])

    # Assert
    assert args.seeds == SEEDS
    assert args.seeds == (110, 111, 112)
    assert args.phase == 1


def test_parse_args_rejects_a_seed_outside_the_experiment():
    """
    A seed outside 110, 111, and 112 should be rejected.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--seeds", "1"])


def test_parse_args_phase_two_defaults_to_the_three_action_groups():
    """
    Phase 2 should run the all, grow, and seq_res groups on the experiment seeds.
    """

    # Arrange / Act
    args = parse_args(["--phase", "2"])

    # Assert
    assert args.phase == 2
    assert args.groups == ("all", "grow", "seq_res")
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_three_defaults_to_the_experiment_seeds():
    """
    Phase 3 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "3"])

    # Assert
    assert args.phase == 3
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_three_rejects_an_action_group_list():
    """
    Phase 3 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "3", "--groups", "grow"])


def test_parse_args_phase_four_defaults_to_the_experiment_seeds():
    """
    Phase 4 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "4"])

    # Assert
    assert args.phase == 4
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_four_rejects_an_action_group_list():
    """
    Phase 4 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "4", "--groups", "grow"])


def test_phase_four_compares_one_two_and_four_epochs_on_the_full_grade():
    """
    Phase 4 should score every action for 1, 2, and 4 epochs on the 45,000-image training set.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    for simulation in PHASE4_SIMULATIONS:
        apply_simulation_budget(cfg, simulation)

    # Assert
    assert PHASE4_SIMULATIONS == ((300.0, 1, 45000), (300.0, 2, 45000), (300.0, 4, 45000))
    assert PHASE4_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert cfg.simulation_set_size == 45000
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 is True


def test_parse_args_phase_five_defaults_to_the_experiment_seeds():
    """
    Phase 5 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "5"])

    # Assert
    assert args.phase == 5
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_five_rejects_an_action_group_list():
    """
    Phase 5 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "5", "--groups", "grow"])


def test_phase_five_compares_epochs_one_through_seven_on_the_clean_prefix():
    """
    Phase 5 should score every action for 1, 2, 4, 5, 6, and 7 epochs on the first 5,000 clean images.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    for simulation in PHASE5_SIMULATIONS:
        apply_simulation_budget(cfg, simulation)

    # Assert
    assert PHASE5_CLEAN_PREFIX == 5000
    assert PHASE5_SIMULATIONS == (
        (300.0, 1, 5000),
        (300.0, 2, 5000),
        (300.0, 4, 5000),
        (300.0, 5, 5000),
        (300.0, 6, 5000),
        (300.0, 7, 5000),
    )
    assert PHASE5_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 7
    assert cfg.simulation_set_size == 5000


def test_parse_args_phase_six_defaults_to_the_experiment_seeds():
    """
    Phase 6 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "6"])

    # Assert
    assert args.phase == 6
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_six_rejects_an_action_group_list():
    """
    Phase 6 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "6", "--groups", "grow"])


def test_phase_six_halves_the_clean_prefix_six_times_at_four_epochs():
    """
    Phase 6 should score every action for 4 epochs on clean prefixes 5000, 2500, 1250, 625, 312, 156, and 78.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    for simulation in PHASE6_SIMULATIONS:
        apply_simulation_budget(cfg, simulation)

    # Assert
    assert PHASE6_EPOCHS == 4
    assert PHASE6_SIZES == (5000, 2500, 1250, 625, 312, 156, 78)
    assert tuple(size // 2 for size in PHASE6_SIZES[:-1]) == PHASE6_SIZES[1:]
    assert PHASE6_SIMULATIONS == tuple((300.0, 4, size) for size in PHASE6_SIZES)
    assert PHASE6_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert cfg.simulation_set_size == 78


def test_parse_args_phase_seven_defaults_to_the_experiment_seeds():
    """
    Phase 7 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "7"])

    # Assert
    assert args.phase == 7
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_seven_rejects_an_action_group_list():
    """
    Phase 7 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "7", "--groups", "grow"])


def test_phase_seven_lengthens_real_training_at_the_1250_image_grade():
    """
    Phase 7 should train 20 and 30 epochs per generation on the 1,250-image, 4-epoch clean grade.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    apply_simulation_budget(cfg, PHASE7_SIMULATION)

    # Assert
    assert PHASE7_TRAIN_EPOCHS == (20, 30)
    assert PHASE7_CLEAN_PREFIX == 1250
    assert PHASE7_SIMULATION == (300.0, 4, 1250)
    assert PHASE7_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert cfg.simulation_set_size == 1250


def test_parse_args_phase_eight_defaults_to_the_experiment_seeds():
    """
    Phase 8 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "8"])

    # Assert
    assert args.phase == 8
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_eight_rejects_an_action_group_list():
    """
    Phase 8 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "8", "--groups", "grow"])


def test_phase_eight_compares_generation_counts_at_ten_real_epochs():
    """
    Phase 8 should compare 10, 15, and 20 generations on the 1,250-image, 4-epoch clean grade.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    apply_simulation_budget(cfg, PHASE8_SIMULATION)

    # Assert
    assert PHASE8_GENERATIONS == (10, 15, 20)
    assert PHASE8_CLEAN_PREFIX == 1250
    assert PHASE8_SIMULATION == (300.0, 4, 1250)
    assert PHASE8_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert cfg.simulation_set_size == 1250


def test_parse_args_phase_ten_defaults_to_five_seeds():
    """
    Phase 10 should train seeds 110, 111, 112, 113, and 114 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "10"])

    # Assert
    assert args.phase == 10
    assert args.seeds == PHASE10_SEEDS
    assert args.seeds == (110, 111, 112, 113, 114)


def test_parse_args_phase_ten_rejects_an_action_group_list():
    """
    Phase 10 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "10", "--groups", "grow"])


def test_parse_args_phase_eight_rejects_a_phase_ten_seed():
    """
    Phases before phase 10 should keep the original seed list of 110, 111, and 112.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "8", "--seeds", "113"])


def test_phase_ten_keeps_every_action_on_the_locked_grade():
    """
    Phase 10 should keep all actions and the 15 by 10 run on the 1,250-image, 4-epoch grade.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    apply_simulation_budget(cfg, PHASE10_SIMULATION)

    # Assert
    assert PHASE10_ARMS == ("base", "lr", "conv", "lr_conv")
    assert PHASE10_GENERATIONS == 15
    assert PHASE10_EPOCHS == 10
    assert PHASE10_CLEAN_PREFIX == 1250
    assert PHASE10_SIMULATION == (300.0, 4, 1250)
    assert PHASE10_ROLLOUT_DEPTH == 0
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert cfg.simulation_set_size == 1250
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is True
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_05 is True


def test_phase_ten_arms_turn_the_learning_rate_and_conv_placement_on_or_off():
    """
    Each phase 10 arm should set the new cosine and the before-pool conv insert as its row requires.
    """

    # Arrange
    expected = {
        "base": (False, False),
        "lr": (True, False),
        "conv": (False, True),
        "lr_conv": (True, True),
    }

    # Act / Assert
    for arm, (new_lr, new_conv) in expected.items():
        cfg = RunningConfig(generations=1, epochs=1)
        apply_action_group(cfg, "all")
        apply_phase10_switches(cfg, arm, PHASE10_GENERATIONS, PHASE10_EPOCHS)
        assert isinstance(cfg.lr_scheduler, CosineDecayScheduler) is new_lr
        assert cfg.seq_conv_insert_before_global_pool is new_conv
        assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
        assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
        if new_lr:
            before = cfg.lr_scheduler.global_epoch
            cfg.lr_scheduler.structure_changed()
            assert cfg.lr_scheduler.global_epoch == before
            assert cfg.lr_scheduler.alpha_scheduler(0, 1) == pytest.approx(0.1)
        else:
            assert cfg.lr_scheduler.initial_lr == pytest.approx(0.01)
            assert cfg.lr_scheduler.total_epochs == PHASE10_GENERATIONS * PHASE10_EPOCHS
            cfg.lr_scheduler.structure_changed()
            assert cfg.lr_scheduler.recovery._schedule.iterations_since_change == 0


def test_parse_args_phase_eleven_defaults_to_two_seeds():
    """
    Phase 11 should train seeds 110 and 111 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "11"])

    # Assert
    assert args.phase == 11
    assert args.seeds == PHASE11_SEEDS
    assert args.seeds == (110, 111)


def test_parse_args_phase_eleven_rejects_an_action_group_list():
    """
    Phase 11 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "11", "--groups", "grow"])


def test_parse_args_phase_eleven_rejects_a_seed_outside_its_pair():
    """
    Phase 11 should keep the seed list at 110 and 111.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "11", "--seeds", "112"])


def test_phase_eleven_arms_each_change_one_base_setting():
    """
    Each phase 11 arm should differ from the phase 10 base recipe in only its own setting.
    """

    # Arrange
    base = {
        "starter": "added",
        "channels": 128,
        "hidden": 256,
        "cutout": True,
        "label_smoothing": 0.1,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0,
        "search": "always",
        "generations": 15,
    }
    changed = {
        "minimal": {"starter": "original", "channels": 32, "hidden": 64},
        "slope": {"search": "slope3"},
        "no_cutout": {"cutout": False},
        "no_smooth": {"label_smoothing": 0.0},
        "no_clip": {"max_grad_norm": None},
        "old_decay": {"weight_decay": 0.0005},
        "short": {"generations": 8},
    }

    # Act / Assert
    assert PHASE11_ARMS == tuple(changed)
    for arm, delta in changed.items():
        settings = phase11_settings(arm)
        expect = dict(base)
        expect.update(delta)
        assert settings == expect


def test_phase_eleven_slope_arm_keeps_the_base_budget_and_all_actions():
    """
    The slope arm should use the 3 degree gate on the 1,250-image, 4-epoch grade with every action on.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")
    apply_simulation_budget(cfg, PHASE11_SIMULATION)

    # Act
    apply_phase11_arm(cfg, "slope", PHASE11_GENERATIONS, PHASE11_EPOCHS)

    # Assert
    assert PHASE11_CLEAN_PREFIX == 1250
    assert PHASE11_SIMULATION == (300.0, 4, 1250)
    assert PHASE11_ROLLOUT_DEPTH == 0
    assert PHASE11_SHORT_GENERATIONS == 8
    assert cfg.simulation_scheduler.simulation_time == 300.0
    assert cfg.simulation_scheduler.simulation_epochs == 4
    assert type(cfg.simulation_scheduler).__name__ == "SlopeEstimationSimulationScheduler"
    assert cfg.seq_conv_insert_before_global_pool is False
    assert cfg.lr_scheduler.initial_lr == pytest.approx(0.01)
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.weight_decay == pytest.approx(0.0001)
    assert cfg.max_grad_norm == 1.0


def test_prefix_clean_loader_keeps_the_first_cuda_images_without_augmentation():
    """
    prefix_clean_loader should keep the first CUDA images and turn augmentation off.
    """

    # Arrange
    images = torch.arange(8 * 3 * 4 * 4, dtype=torch.uint8).reshape(8, 3, 4, 4)
    labels = torch.arange(8)
    loader = _BatchLoader(_CudaCIFAR(images, labels, augment=True), batch_size=2, shuffle=False)

    # Act
    prefix = prefix_clean_loader(loader, 5)

    # Assert
    assert len(prefix.dataset) == 5
    assert prefix.dataset.augment is False
    assert torch.equal(prefix.dataset.labels, labels[:5])


def test_prefix_clean_loader_keeps_the_first_subset_images():
    """
    prefix_clean_loader should keep the first images of a plain dataset.
    """

    # Arrange
    images = torch.zeros(8, 1)
    labels = torch.arange(8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    # Act
    prefix = prefix_clean_loader(loader, 5)
    taken = torch.stack([item[1] for item in prefix.dataset])

    # Assert
    assert len(prefix.dataset) == 5
    assert torch.equal(taken, labels[:5])


def test_rollout_depth_uses_zero_extra_actions_and_restores_two():
    """
    rollout_depth should set the Monte Carlo extra-action count for the block and restore it after.
    """

    # Arrange
    project_config.MCTS_ROLLOUT_DEPTH = 2

    # Act
    with rollout_depth(PHASE4_ROLLOUT_DEPTH):
        inside = project_config.MCTS_ROLLOUT_DEPTH

    # Assert
    assert inside == 0
    assert project_config.MCTS_ROLLOUT_DEPTH == 2


def test_phase_three_budget_is_one_hundred_seconds_on_the_all_group():
    """
    The phase 3 budget should be 100 seconds, 8 epochs, and 2000 images, with every action still enabled.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    apply_simulation_budget(cfg, PHASE3_SIMULATION)

    # Assert
    assert PHASE3_SIMULATION == (100.0, 8, 2000)
    assert isinstance(cfg.simulation_scheduler, AlwaysSimulationScheduler)
    assert cfg.simulation_scheduler.simulation_time == 100.0
    assert cfg.simulation_scheduler.simulation_epochs == 8
    assert cfg.simulation_set_size == 2000
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True


def test_grow_group_keeps_add_neurons_and_turns_delete_off():
    """
    The grow group should allow layer adds and neuron adds, and disallow deletes.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)

    # Act
    apply_action_group(cfg, "grow")

    # Assert
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is True
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is False
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01 is False


def test_seq_res_group_turns_add_neurons_off():
    """
    The seq_res group should allow sequential and residual adds, and disallow neuron adds.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)

    # Act
    apply_action_group(cfg, "seq_res")

    # Assert
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is False
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_05 is False


def test_all_group_enables_delete_and_dropout():
    """
    The all group should leave every structure-action flag on.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "seq_conv")

    # Act
    apply_action_group(cfg, "all")

    # Assert
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_09 is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_20 is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True


def test_recipe_is_the_twelve_by_ten_cosine_generation_loop():
    """
    The fixed recipe should be 12 generations of 10 epochs on a 128-channel starter.
    """

    # Arrange
    scheduler = CosineDecayScheduler(GENERATIONS * EPOCHS)

    # Act
    learning_rate = scheduler.alpha_scheduler(0, 1)

    # Assert
    assert GENERATIONS == 12
    assert EPOCHS == 10
    assert CHANNELS == 128
    assert HIDDEN == 256
    assert scheduler.total_epochs == 120
    assert learning_rate == pytest.approx(0.1)


def test_history_marks_official_test_at_generation_ends_without_finish_chunks():
    """
    Saved history should place each official test at the end of its generation and omit finish chunks.
    """

    # Arrange
    summary = {
        "train_acc": [0.1, 0.2, 0.3, 0.4],
        "train_loss": [1.0, 0.9, 0.8, 0.7],
        "val_acc": [0.1, 0.2, 0.3, 0.4],
        "val_loss": [1.0, 0.9, 0.8, 0.7],
        "lr": [0.1, 0.1, 0.09, 0.09],
        "param_count": [10, 10, 20, 20],
        "test_acc": [0.15, 0.35],
        "test_loss": [1.1, 0.8],
        "action_generation": [0],
        "action_name": ["AddSeqConvLayer"],
        "action_val_acc_before": [0.2],
        "action_val_acc_after": [0.2],
    }

    # Act
    history = history_from_generations(summary, seed=110, epochs_per_generation=2)

    # Assert
    assert history["test_epoch"] == [2, 4]
    assert history["test_acc"] == [0.15, 0.35]
    assert "best_chunk" not in history
    assert "best_finish_test_acc" not in history


def test_with_cutout_keeps_the_training_set_length():
    """
    Cutout should wrap the training loader without dropping images.
    """

    # Arrange
    images = torch.zeros(6, 3, 32, 32)
    targets = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=True)

    # Act
    wrapped = with_cutout(loader)

    # Assert
    assert len(wrapped.dataset) == 6


def test_with_cutout_keeps_the_cuda_batch_iterator():
    """
    Cutout should erase inside a CUDA batch loader instead of loading one image at a time.
    """

    # Arrange
    class _Images:
        images = torch.zeros(4, 3, 8, 8)

        def __len__(self) -> int:
            return 4

    class _FastLoader:
        def __init__(self) -> None:
            self.dataset = _Images()
            self.batch_size = 2
            self._shuffle = True

        def __iter__(self):
            yield torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long)

        def __len__(self) -> int:
            return 2

    loader = _FastLoader()

    # Act
    wrapped = with_cutout(loader)
    images, targets = next(iter(wrapped))

    # Assert
    assert wrapped.dataset is loader.dataset
    assert images.shape == (2, 3, 8, 8)
    assert targets.shape == (2,)
