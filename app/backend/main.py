from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
FRONTEND_DIR = ROOT / "app" / "frontend"
JOBS_FILE = ROOT / "app" / "jobs" / "jobs.json"


class JobRequest(BaseModel):
    mode: str = Field(..., description="pipeline_runner mode")
    args: dict[str, Any] = Field(default_factory=dict)


ALLOWED_MODES = {
    "gcn_infer",
    "pretrained",
    "evaluate",
    "train_gcn",
    "train_gcn_stage_a",
    "train_gcn_stage_b",
    "train_ocr",
    "test_gcn",
    "preprocess_gcn_dataset",
    "convert_hf_cord_to_csv",
    "convert_hf_to_gcn_csv",
    "prepare_ocr_labeling",
    "train_gcn_full",
}


def _load_jobs() -> list[dict[str, Any]]:
    if not JOBS_FILE.exists():
        return []
    return json.loads(JOBS_FILE.read_text(encoding="utf-8"))


def _save_jobs(jobs: list[dict[str, Any]]) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_cmd(mode: str, args: dict[str, Any]) -> list[str]:
    cmd = ["python", "pipeline_runner.py", mode]
    for key, value in args.items():
        if value is None or value == "":
            continue
        flag = "--" + key.replace("_", "-")
        cmd.append(flag)
        cmd.append(str(value))
    return cmd


def _run_job(job_id: str, cmd: list[str]) -> None:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "running"
            j["started_at"] = datetime.utcnow().isoformat()
    _save_jobs(jobs)

    proc = subprocess.run(
        cmd,
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "success" if proc.returncode == 0 else "failed"
            j["return_code"] = proc.returncode
            j["stdout"] = proc.stdout[-20000:]
            j["stderr"] = proc.stderr[-20000:]
            j["finished_at"] = datetime.utcnow().isoformat()
    _save_jobs(jobs)


app = FastAPI(title="Invoice OCR GCN Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/frontend/dashboard.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/features")
def features() -> dict[str, Any]:
    return {
        "labels": ["OTHER", "DATE", "TAX_CODE", "TOTAL_AMOUNT", "PRODUCT_NAME", "UNIT_PRICE"],
        "modes": sorted(ALLOWED_MODES),
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return _load_jobs()[::-1]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    for j in _load_jobs():
        if j["id"] == job_id:
            return j
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict[str, Any]:
    if req.mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

    job_id = str(uuid.uuid4())
    cmd = _build_cmd(req.mode, req.args)
    job = {
        "id": job_id,
        "mode": req.mode,
        "args": req.args,
        "cmd": cmd,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }

    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)

    t = threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True)
    t.start()

    return job
