"""FastAPI routes for GrowingNN Board."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from growingnn_board.cache import ExperimentCache
from growingnn_board.config import settings
from growingnn_board.file_reader import directory_status, resolve_experiment_directory
from growingnn_board.score_recalculation import apply_recalculated_scores, recalculate_simulation
from growingnn_board.search_tree_viz import render_search_tree_html, resolve_search_tree

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
    experiment_path = resolve_experiment_directory(path, root=settings.experiments_root)
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


@router.get("/simulations")
def list_simulations():
    return {"generations": sorted(_cache.simulations.keys())}


@router.get("/simulation/{generation_number}")
def get_simulation(generation_number: int):
    data = _cache.simulations.get(generation_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return data


@router.get("/simulation/{generation_number}/recalculate")
def recalculate_simulation_scores(
    generation_number: int,
    accuracy_weight: float = Query(..., ge=0),
    param_count_weight: float = Query(..., ge=0),
):
    data = _cache.simulations.get(generation_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    try:
        return recalculate_simulation(data, accuracy_weight, param_count_weight)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/simulation/{generation_number}/search-tree", response_class=HTMLResponse)
def get_simulation_search_tree(
    generation_number: int,
    accuracy_weight: float | None = Query(None, ge=0),
    param_count_weight: float | None = Query(None, ge=0),
):
    if _cache.path is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")
    data = _cache.simulations.get(generation_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    tree = resolve_search_tree(data)
    if tree is None:
        raise HTTPException(status_code=404, detail="Search tree data not available")
    if (accuracy_weight is None) != (param_count_weight is None):
        raise HTTPException(status_code=422, detail="Both preview weights are required")
    if accuracy_weight is not None and param_count_weight is not None:
        try:
            recalculation = recalculate_simulation(data, accuracy_weight, param_count_weight)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        tree = apply_recalculated_scores(tree, recalculation)
    try:
        html = render_search_tree_html(
            tree,
            rollouts=data.get("rollouts"),
            max_depth=data.get("maxDepth"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search tree render failed: {exc}") from exc
    return HTMLResponse(html)


@router.get("/files/html")
def get_html(path: str = Query(..., description="Experiment-relative HTML path")):
    if _cache.path is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")
    experiment_root = _cache.path.resolve()
    file_path = Path(path)
    if not file_path.is_file():
        file_path = experiment_root / path
    file_path = file_path.resolve()
    if file_path.suffix.lower() not in {".html", ".htm"}:
        raise HTTPException(status_code=404, detail="HTML not found")
    try:
        file_path.relative_to(experiment_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="HTML path outside experiment directory") from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="HTML not found")
    return FileResponse(file_path, media_type="text/html")


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
