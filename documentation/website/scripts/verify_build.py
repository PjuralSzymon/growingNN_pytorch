"""Validate the prerendered Angular documentation output."""

from __future__ import annotations

import re
from pathlib import Path


SITE = Path(__file__).parents[1]
DEFAULT_OUTPUT = SITE / "app" / "dist" / "growingnn-docs" / "browser"


def missing_internal_targets(output: Path) -> list[str]:
    """Return absolute links that have no file or prerendered route target."""
    links = {
        match
        for html_file in output.rglob("*.html")
        if html_file.name != "knowledge-graph.html"
        for match in re.findall(r'(?:href|src)="(/[^"#?]*)', html_file.read_text(encoding="utf-8"))
    }
    return sorted(
        link
        for link in links
        if not (output / link.lstrip("/")).is_file()
        and not (output / link.lstrip("/") / "index.html").is_file()
    )


def verify(output: Path = DEFAULT_OUTPUT, expected_route_count: int | None = None) -> None:
    """Raise an error when required Angular artifacts or routes are missing."""
    if expected_route_count is None:
        from generate_content import load_pages

        expected_route_count = len(load_pages()) + 3
    pages = list(output.rglob("index.html"))
    if len(pages) != expected_route_count:
        raise RuntimeError(f"Expected {expected_route_count} prerendered routes, found {len(pages)}")
    if not (output / "assets" / "knowledge-graph.html").is_file():
        raise RuntimeError("The generated PyVis graph is missing")
    missing = missing_internal_targets(output)
    if missing:
        raise RuntimeError(f"Missing internal targets: {missing}")
    print(f"Verified {len(pages)} prerendered routes with no missing internal targets")


if __name__ == "__main__":
    verify()
