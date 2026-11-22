from types import SimpleNamespace

from experiments.train_cifar10_exp007_test90_package import StepDecayScheduler
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler
from growingnn.simulation.score_functions import simulation_training
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.lr_scheduler_global import ComposedLearningRateScheduler, CosineAnnealingLearningRate

MAIN_SCHEDULES = {
    'cosine (experiments 010-015)': lambda: CosineDecayScheduler(total_epochs=200),
    'step (experiment 007)': lambda: StepDecayScheduler(at_01=100, at_001=150),
    'composed with warmup recovery': lambda: ComposedLearningRateScheduler(
        global_schedule=CosineAnnealingLearningRate(t_max=200, initial_lr=0.1),
        recovery=ActionLearningRateScheduler(ScheduleMode.WARMUP_COSINE, alpha=1.0, warmup_iterations=5),
        total_epochs=200),
    'action warmup': lambda: ActionLearningRateScheduler(ScheduleMode.WARMUP_COSINE, alpha=0.1,
                                                         warmup_iterations=5),
}


def _record_learning_rates(monkeypatch):
    def fake_gradient_descent(model, epochs, train_loader, val_loader, criterion, lr_scheduler, **kwargs):
        return model, {'lr': [lr_scheduler.alpha_scheduler(i, epochs) for i in range(epochs)]}

    monkeypatch.setattr(simulation_training, 'gradient_descent', fake_gradient_descent)


def _score(main_scheduler):
    config = SimpleNamespace(
        lr_scheduler=main_scheduler, simulation_scheduler=SimpleNamespace(simulation_epochs=5),
        sim_train_loader=None, sim_val_loader=None, criterion=None, momentum=0.9, weight_decay=0.0,
        device='cpu', max_grad_norm=None)
    return simulation_training.run_simulation_scoring_gradient_descent(object(), config)[1]['lr']


def _main_run(make_schedule, with_scoring):
    """Main-run learning rates over 10 generations of 20 epochs, optionally with scoring runs."""
    schedule, rates = make_schedule(), []
    for _ in range(10):
        rates += [schedule.alpha_scheduler(epoch, 20) for epoch in range(20)]
        if with_scoring:
            for _ in range(3):
                _score(schedule)
        schedule.structure_changed()  # the trainer calls this after each applied action
    return rates


def test_scoring_starts_the_main_schedule_from_the_beginning(monkeypatch):
    _record_learning_rates(monkeypatch)
    for name, make in MAIN_SCHEDULES.items():
        main = make()
        for _ in range(180):  # where the main run is when generation 8's search starts
            main.alpha_scheduler(0, 20)
        fresh = make()

        assert _score(main) == _score(main) == [fresh.alpha_scheduler(i, 5) for i in range(5)], name


def test_scoring_runs_never_change_the_main_runs_learning_rates(monkeypatch):
    _record_learning_rates(monkeypatch)
    for name, make in MAIN_SCHEDULES.items():
        assert _main_run(make, with_scoring=True) == _main_run(make, with_scoring=False), name
