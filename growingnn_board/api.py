"""FastAPI routes for GrowingNN Board."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from growingnn_board.cache import ExperimentCache
from growingnn_board.config import settings
from growingnn_board.file_reader import directory_status

router = APIRouter(prefix="/api")
_cache = ExperimentCache()


def get_cache() -> ExperimentCache:
    return _cache


@router.get("/experiments/recent")
def recent_experiments():
    root = settings.experiments_root
    if not root.is_dir():
        return {"experiments": []}
    rows = []
    seen: set[str] = set()
    for main in root.rglob("main.json"):
        path = main.parent
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        from growingnn_board.file_reader import read_main

        parsed = read_main(main)
        if parsed is None:
            continue
        rows.append(
            {
                "path": key,
                "lastUpdate": parsed.lastUpdate,
                "status": directory_status(parsed.lastUpdate),
                "experimentName": parsed.experimentName,
            }
        )
    rows.sort(key=lambda r: r["lastUpdate"], reverse=True)
    return {"experiments": rows[:20]}


@router.post("/experiment/load")
def load_experiment(path: str):
    experiment_path = Path(path)
    if not experiment_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    _cache.load(experiment_path)
    if _cache.main is None:
        raise HTTPException(status_code=400, detail="Not a valid GrowingNN experiment directory")
    return {"ok": True, "warnings": _cache.warnings}


@router.get("/experiment/main")
def experiment_main():
    if _cache.main is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")
    return _cache.main


@router.get("/experiment/training")
def experiment_training():
    if _cache.training is None:
        raise HTTPException(status_code=404, detail="Training metrics not available")
    return _cache.training


@router.get("/generations")
def list_generations():
    return {"generations": sorted(_cache.generations.keys())}


@router.get("/generation/{generation_number}")
def get_generation(generation_number: int):
    data = _cache.generations.get(generation_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return data


@router.get("/simulation/{generation_number}")
def get_simulation(generation_number: int):
    data = _cache.simulations.get(generation_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return data


@router.get("/files/pdf")
def get_pdf(path: str = Query(..., description="Absolute or experiment-relative PDF path")):
    if _cache.path is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")
    experiment_root = _cache.path.resolve()
    file_path = Path(path)
    if not file_path.is_file():
        file_path = experiment_root / path
    file_path = file_path.resolve()
    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        file_path.relative_to(experiment_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="PDF path outside experiment directory") from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")
