import random

from experiments.action_family_order import interleave_action_families


class A:
    pass


class B:
    pass


class C:
    pass


def test_initial_round_has_each_family_and_keeps_all_actions():
    actions = [A() for _ in range(7)] + [B() for _ in range(2)] + [C()]
    ordered = interleave_action_families(actions, random.Random(42))
    assert len(ordered) == len(actions)
    assert {id(a) for a in ordered} == {id(a) for a in actions}
    assert {type(a) for a in ordered[:3]} == {A, B, C}
    assert {type(a) for a in ordered[3:5]} == {A, B}


def test_seeded_order_is_reproducible_and_handles_empty_input():
    actions = [A(), A(), B(), C()]
    assert interleave_action_families(actions, random.Random(9)) == interleave_action_families(
        actions, random.Random(9))
    assert interleave_action_families([], random.Random(9)) == []
