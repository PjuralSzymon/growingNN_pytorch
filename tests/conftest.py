import pytest

# This is a pytest hook that adds the description of the test to the report when the test fails.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call" or not rep.failed:
        return
    marker = item.get_closest_marker("description")
    if not marker or not marker.args:
        return
    desc = marker.args[0]
    extra = f"\n\n[Test description]\n{desc}\n"
    try:
        if rep.longrepr is not None:
            rep.longrepr = str(rep.longrepr) + extra
    except Exception:
        pass
