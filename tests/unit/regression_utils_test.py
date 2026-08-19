"""Unit tests for regression action-count table formatting."""

from tests.regression.regression_utils import action_count_table_lines


def test_action_count_table_lines_includes_known_zeros_and_total():
    """
    action_count_table_lines should list known types at zero and sum the filled counts.
    """

    # Arrange / Act
    rows = action_count_table_lines({"AddSeqLinearLayer": 14, "DelLayer": 7})

    # Assert
    body = {r.split("|")[0].strip(): int(r.split("|")[1]) for r in rows if " | " in r and not r.startswith("action")}
    assert body["AddSeqLinearLayer"] == 14
    assert body["DelLayer"] == 7
    assert body["AddResConvLayer"] == 0
    assert body["total"] == 21
