"""Start GrowingNN Board. Works from repo root or from growingnn_board/."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    import uvicorn

    from growingnn_board.config import settings

    uvicorn.run(
        "growingnn_board.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
