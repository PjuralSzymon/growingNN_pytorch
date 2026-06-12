"""GrowingNN Board FastAPI application."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from growingnn_board.api import get_cache, router
from growingnn_board.config import settings

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class DevStaticFiles(StaticFiles):
    """Serve static assets without long-lived browser cache during development."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".html", ".css")):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


app = FastAPI(title="GrowingNN Board", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)

if _STATIC_DIR.is_dir():
    app.mount("/static", DevStaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index_page():
    response = FileResponse(_STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _poll_loop() -> None:
    cache = get_cache()
    while True:
        if cache.path is not None:
            cache.load(cache.path)
        time.sleep(settings.poll_interval_sec)


@app.on_event("startup")
def _start_watcher() -> None:
    thread = threading.Thread(target=_poll_loop, daemon=True)
    thread.start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("growingnn_board.app:app", host=settings.host, port=settings.port, reload=False)
