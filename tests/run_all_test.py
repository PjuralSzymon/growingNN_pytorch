"""
Run every test category under ``tests/`` and print how many passed vs failed.

- **unit**: ``pytest tests/unit`` (single run; summary line shows pass/fail counts).
- **regression**: each ``tests/regression/*.py`` harness (except helpers) as a subprocess
  with ``MPLBACKEND=Agg`` and ``--save-output false`` so plots never block and PDFs are not kept.
- **integration**: ``pytest tests/integration`` if that directory exists; otherwise skipped.

Usage (from repo root)::

    python tests/run_all_test.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS = REPO_ROOT / "tests"
UNIT_DIR = TESTS / "unit"
REGRESSION_DIR = TESTS / "regression"
INTEGRATION_DIR = TESTS / "integration"

# Not standalone harnesses (import-only helpers).
REGRESSION_SKIP = frozenset({"regression_utils.py"})


def _agg_env() -> dict[str, str]:
    """Non-interactive matplotlib; no GUI ``plt.show`` blocking."""
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
    }


def run_pytest(target: Path, label: str) -> tuple[int, str, list[str]]:
    """Run pytest on ``target`` (file or dir).

    Returns ``(exit_code, summary_line, failed_test_ids)``.

    ``failed_test_ids`` is parsed from pytest's short summary section
    (lines starting with ``FAILED``/``ERROR``) so callers can list the
    individual failing tests by name.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(target.relative_to(REPO_ROOT)),
        "-q",
        "--tb=line",
        "-rfE",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_agg_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )

    stdout_lines = proc.stdout.splitlines() if proc.stdout else []

    failed_tests: list[str] = []
    for raw in stdout_lines:
        line = raw.strip()
        m = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", line)
        if m:
            failed_tests.append(m.group(1))

    tail = ""
    for raw in reversed(stdout_lines):
        candidate = raw.strip()
        if re.search(r"\b(passed|failed|error|skipped|deselected|no tests ran)\b", candidate, re.I):
            tail = candidate
            break
    if not tail and stdout_lines:
        tail = stdout_lines[-1].strip()
    if proc.returncode != 0 and proc.stderr:
        tail = (tail + "\n" + proc.stderr.strip()).strip()

    return proc.returncode, tail or f"exit {proc.returncode}", failed_tests


def discover_regression_scripts() -> list[Path]:
    scripts: list[Path] = []
    if not REGRESSION_DIR.is_dir():
        return scripts
    for p in sorted(REGRESSION_DIR.glob("*.py")):
        if p.name in REGRESSION_SKIP:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if 'if __name__ == "__main__"' not in text and "if __name__ == '__main__'" not in text:
            continue
        scripts.append(p)
    return scripts


def run_regression_script(path: Path) -> tuple[int, str]:
    cmd = [sys.executable, str(path), "--save-output", "false"]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_agg_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    msg = ""
    if proc.stdout:
        msg += proc.stdout[-4000:]
    if proc.stderr:
        msg += "\n" + proc.stderr[-2000:]
    return proc.returncode, msg.strip()


def parse_pytest_summary_line(line: str) -> str:
    """Keep pytest's own short summary if it looks like a result line."""
    line = line.strip()
    if re.search(r"\b(passed|failed|error|skipped|deselected)\b", line, re.I):
        return line
    return line


def main() -> int:
    print(f"Repo root: {REPO_ROOT}\n")

    overall_fail = 0

    # --- Unit ---
    print("=== UNIT (pytest tests/unit) ===")
    if not UNIT_DIR.is_dir():
        print("  [skip] tests/unit not found")
    else:
        code, tail, failed = run_pytest(UNIT_DIR, "unit")
        print(" ", parse_pytest_summary_line(tail))
        if code != 0:
            overall_fail += 1
            print(f"  [FAIL] pytest exit code {code}")
            if failed:
                print("  Failed tests:")
                for node_id in failed:
                    print(f"    - {node_id}")

    # --- Integration ---
    print("\n=== INTEGRATION (pytest tests/integration) ===")
    if not INTEGRATION_DIR.is_dir():
        print("  [skip] tests/integration not present")
    else:
        code, tail, failed = run_pytest(INTEGRATION_DIR, "integration")
        print(" ", parse_pytest_summary_line(tail))
        if code != 0:
            overall_fail += 1
            print(f"  [FAIL] pytest exit code {code}")
            if failed:
                print("  Failed tests:")
                for node_id in failed:
                    print(f"    - {node_id}")

    # --- Regression (each script is its own process) ---
    print("\n=== REGRESSION (subprocess per script, MPLBACKEND=Agg, --save-output false) ===")
    scripts = discover_regression_scripts()
    if not scripts:
        print("  [skip] no runnable regression scripts found")
    else:
        passed = failed = 0
        for script in scripts:
            rel = script.relative_to(REPO_ROOT)
            code, blob = run_regression_script(script)
            ok = code == 0
            if ok:
                passed += 1
                print(f"  PASS  {rel}")
            else:
                failed += 1
                overall_fail += 1
                print(f"  FAIL  {rel}  (exit {code})")
                if blob:
                    for ln in blob.splitlines()[-15:]:
                        print(f"        {ln}")
        print(f"\n  Regression summary: {passed} passed, {failed} failed (of {len(scripts)} scripts)")

    print("\n=== DONE ===")
    if overall_fail:
        print(f"Overall: {overall_fail} group(s) / script(s) failed (see above).")
        return 1
    print("Overall: all executed groups passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
