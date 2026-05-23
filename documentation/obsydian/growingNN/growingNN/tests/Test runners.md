This page lists top-level test driver scripts under `tests/`.

### `tests/run_all_test.py`

Runs unit tests with pytest on `tests/unit`. Runs each `tests/regression/*.py` file as a subprocess with `MPLBACKEND=Agg` and flags so plots do not block. Optionally runs `tests/integration` if the folder exists. Skips helper-only files such as `regression_utils.py` (see `REGRESSION_SKIP` near line 29). Parses pytest output for failed test names when pytest is run with failure-report flags inside that script.

Usage from repo root: `python tests/run_all_test.py`.

### `tests/unit/run_all_unit_tests.py`

Sets `growingnn.core.config.ENABLE_LOGGING` to `False` then calls `pytest.main(["."])` with current working directory expected to be the unit test tree. Used when you want a quick local run with logging off.

### Related

[[Unit tests index]], [[ResNet18 regression script]].
