"""GrowingNN Board server settings."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    experiments_root: Path = Path(os.environ.get("GROWINGNN_EXPERIMENTS_ROOT", "experiments/output"))
    poll_interval_sec: int = int(os.environ.get("GROWINGNN_BOARD_POLL_SEC", "5"))
    host: str = os.environ.get("GROWINGNN_BOARD_HOST", "127.0.0.1")
    port: int = int(os.environ.get("GROWINGNN_BOARD_PORT", "8765"))


settings = Settings()
