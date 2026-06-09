from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
import csv
import re
import sys
import random
import unicodedata
from urllib import request as urllib_request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pipeline.core.gcn_classifier import build_features
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.schema import OCRNode

FRONTEND_DIR = ROOT / "app" / "frontend" / "dist"
FRONTEND_FALLBACK_DIR = ROOT / "app" / "frontend"
JOBS_FILE = ROOT / "app" / "jobs" / "jobs.json"
STAGE_B_RAW_DIR = SRC_DIR / "data" / "stage_b_raw_images"
ROOT_OUTPUTS_CHECKPOINTS_DIR = ROOT / "outputs" / "checkpoints"
PYTHON_EXE = Path(sys.executable).resolve()
CONDA_ENV_DIR = PYTHON_EXE.parent


def _load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()

INVOICE_LABELS = [
    "MERCHANT_NAME",
    "MERCHANT_ADDRESS",
    "MERCHANT_PHONE",
    "TAX_CODE",
    "INVOICE_ID",
    "DATE",
    "TIME",
    "CASHIER",
    "ITEM_NAME",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_AMOUNT",
    "SUBTOTAL",
    "SERVICE_FEE",
    "DISCOUNT",
    "TAX_AMOUNT",
    "TOTAL_AMOUNT",
    "PAYMENT_METHOD",
    "OTHER",
]


class JobRequest(BaseModel):
    mode: str = Field(..., description="pipeline_runner mode")
    args: dict[str, Any] = Field(default_factory=dict)


class GcnInferRequest(BaseModel):
    image: str
    lang: str = "en"
    ocr_engine: str = "paddle"
    checkpoint: str | None = None
    ocr_debug_image: str = "outputs/ocr_boxes.jpg"
    output_json: str = "outputs/ocr_result.json"


class PretrainedRequest(BaseModel):
    image: str
    project_dir: str = "."
    lang: str = "en"
    ocr_engine: str = "paddle"
    ocr_debug_image: str = "outputs/ocr_boxes_pretrained.jpg"
    output_json: str = "outputs/pretrained_invoice_result.json"


class EvaluateRequest(BaseModel):
    pred_json: str
    gt_json: str
    output_eval: str = "outputs/eval_report.json"


class TestGcnRequest(BaseModel):
    dataset_json: str
    checkpoint: str
    output_eval: str = "outputs/gcn_eval_report.json"


class PreprocessDatasetRequest(BaseModel):
    input_csv: str
    output_json: str
    doc_id_col: str = "doc_id"
    text_col: str = "text"
    label_col: str = "label"
    score_col: str = "score"
    x1_col: str = "x1"
    y1_col: str = "y1"
    x2_col: str = "x2"
    y2_col: str = "y2"
    same_line_ratio: float = 1.2
    near_threshold: float = 250.0
    min_nodes_per_graph: int = 1


class SplitDatasetRequest(BaseModel):
    input_json: str
    output_train_json: str = "data/stage_b_train.json"
    output_val_json: str = "data/stage_b_val.json"
    output_test_json: str = "data/stage_b_test.json"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    shuffle: int = 1


class ValidateLabelCsvRequest(BaseModel):
    input_csv: str
    label_col: str = "label"


class LabelUpdateItem(BaseModel):
    row_number: int
    label: str


class ApplyLabelUpdatesRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    updates: list[LabelUpdateItem]


class AutoSuggestLabelsRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    text_col: str = "text"
    only_empty: int = 1
    strategy: str = "llm"  # llm | rule
    llm_model: str = "gpt-4.1-mini"
    batch_size: int = 30


class AutoSuggestStartRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    text_col: str = "text"
    doc_id_col: str = "doc_id"
    only_empty: int = 1
    llm_model: str = "gpt-4.1-mini"
    batch_docs: int = 10
    llm_text_batch_size: int = 10
    require_llm: int = 1


class LabelingSampleRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    text_col: str = "text"
    limit: int = 100
    page: int = 1


class LabelingByDocRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    text_col: str = "text"
    doc_id_col: str = "doc_id"
    page: int = 1
    page_size: int = 50


class LabelingGraphInspectRequest(BaseModel):
    input_csv: str
    doc_id: str
    label_col: str = "label"
    text_col: str = "text"
    doc_id_col: str = "doc_id"
    score_col: str = "score"
    x1_col: str = "x1"
    y1_col: str = "y1"
    x2_col: str = "x2"
    y2_col: str = "y2"
    same_line_ratio: float = 1.2
    near_threshold: float = 250.0


class SingleImagePreviewRequest(BaseModel):
    image: str
    lang: str = "vi"
    ocr_engine: str = "paddle"
    det_db_thresh: float = 0.25
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.25
    drop_score: float = 0.45
    use_dilation: int = 0
    det_limit_side_len: int = 1600
    upscale_factor: float = 1.6
    with_ai: int = 0
    llm_model: str = "gpt-4.1-mini"
    output_dir: str = "data/single_image_check"
    save_debug_image: int = 1
    same_line_ratio: float = 1.2
    near_threshold: float = 250.0
    llm_text_batch_size: int = 10


class PrepareOcrLabelingRequest(BaseModel):
    input_dir: str
    output_dir: str = "data/labeling_stage_b"
    lang: str = "en"
    ocr_engine: str = "paddle"
    det_db_thresh: float = 0.25
    det_db_box_thresh: float = 0.58
    det_db_unclip_ratio: float = 1.25
    drop_score: float = 0.45
    use_dilation: int = 0
    det_limit_side_len: int = 1536
    upscale_factor: float = 1.6
    save_debug_images: int = 1
    copy_images: int = 1


class TrainGcnStageARequest(BaseModel):
    dataset_json: str
    checkpoint: str = "outputs/checkpoints/gcn_stage_a.pt"
    epochs: int = 30
    lr: float = 1e-3
    init_checkpoint: str | None = None
    val_dataset_json: str | None = None
    early_stop_patience: int = 0


class TrainGcnStageBRequest(BaseModel):
    dataset_json: str
    base_checkpoint: str
    checkpoint: str = "outputs/checkpoints/gcn_stage_b.pt"
    epochs: int = 20
    lr: float = 5e-4
    val_dataset_json: str | None = None
    early_stop_patience: int = 0


class TrainGcnFullRequest(BaseModel):
    stage_a_csv: str | None = None
    stage_b_csv: str | None = None
    stage_a_json: str = "data/stage_a_dataset.json"
    stage_b_json: str = "data/stage_b_vi_dataset.json"
    stage_a_ckpt: str = "outputs/checkpoints/gcn_stage_a.pt"
    stage_b_ckpt: str = "outputs/checkpoints/gcn_stage_b.pt"
    stage_a_epochs: int = 30
    stage_b_epochs: int = 20
    stage_a_lr: float = 1e-3
    stage_b_lr: float = 5e-4
    init_checkpoint: str | None = None
    eval_json: str | None = None
    output_eval: str = "outputs/gcn_eval_report.json"
    doc_id_col: str = "doc_id"
    text_col: str = "text"
    label_col: str = "label"
    score_col: str = "score"
    x1_col: str = "x1"
    y1_col: str = "y1"
    x2_col: str = "x2"
    y2_col: str = "y2"
    same_line_ratio: float = 1.2
    near_threshold: float = 250.0
    min_nodes_per_graph: int = 1


class TrainOcrRequest(BaseModel):
    command: str
    workdir: str | None = None


class ConvertCordRequest(BaseModel):
    dataset_id: str = "naver-clova-ix/cord-v2"
    split: str = "train"
    output_csv: str | None = None
    limit: int | None = None
    streaming: int = 1


class ConvertGenericRequest(BaseModel):
    dataset_id: str
    split: str = "train"
    output_csv: str | None = None
    doc_id_field: str = "id"
    text_field: str = "text"
    label_field: str = "label"
    bbox_field: str = "bbox"
    score_field: str | None = None
    label_map: str | None = None
    limit: int | None = None
    streaming: int = 1


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


def _write_csv_rows(csv_path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def _resolve_project_path(path_str: str) -> Path:
    normalized = str(path_str or "").replace("\\", "/").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Missing path")

    candidates: list[Path] = []
    rel_path = Path(normalized)
    candidates.append((SRC_DIR / rel_path).resolve())
    candidates.append((ROOT / rel_path).resolve())

    src_root = str(SRC_DIR.resolve())
    project_root = str(ROOT.resolve())
    for candidate in candidates:
        candidate_str = str(candidate)
        if not (candidate_str.startswith(src_root) or candidate_str.startswith(project_root)):
            continue
        if candidate.exists():
            return candidate

    fallback = candidates[0]
    fallback_str = str(fallback)
    if fallback_str.startswith(src_root) or fallback_str.startswith(project_root):
        return fallback
    raise HTTPException(status_code=400, detail="Invalid path")


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    path_parts: list[str] = []

    for candidate in (
        CONDA_ENV_DIR / "Library" / "bin",
        CONDA_ENV_DIR / "DLLs",
        CONDA_ENV_DIR / "Scripts",
        CONDA_ENV_DIR,
    ):
        if candidate.exists():
            path_parts.append(str(candidate))

    existing_path = env.get("PATH", "")
    if existing_path:
        path_parts.append(existing_path)
    env["PATH"] = os.pathsep.join(path_parts)
    env["PYTHONHOME"] = ""
    env["PYTHONPATH"] = str(SRC_DIR)
    return env


def _project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(SRC_DIR.resolve())).replace("\\", "/")
    except ValueError:
        try:
            return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(resolved).replace("\\", "/")


def _image_files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ]


def _stage_b_image_index(base_dir: Path | None = None) -> list[Path]:
    STAGE_B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    indexed: list[Path] = []
    if base_dir is not None:
        indexed.extend(_image_files_under(base_dir / "images"))
    indexed.extend(_image_files_under(STAGE_B_RAW_DIR))
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in indexed:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _find_image_for_doc(doc_id: str, image_index: list[Path]) -> str | None:
    did = doc_id.strip().lower()
    exact = next((p for p in image_index if p.stem.lower() == did), None)
    if exact:
        return str(exact.relative_to(SRC_DIR))
    fuzzy = next((p for p in image_index if did in p.stem.lower()), None)
    if fuzzy:
        return str(fuzzy.relative_to(SRC_DIR))
    return None


def _load_doc_quads_from_ocr_json(input_csv_path: Path, doc_id: str) -> list[list[list[float]]]:
    try:
        ocr_json_path = input_csv_path.parent / "ocr_json" / f"{doc_id}.json"
        if not ocr_json_path.exists():
            return []
        payload = json.loads(ocr_json_path.read_text(encoding="utf-8"))
        quads: list[list[list[float]]] = []
        for node in payload.get("nodes", []):
            quad = node.get("quad") or []
            if isinstance(quad, list) and len(quad) >= 4:
                quads.append([[float(pt[0]), float(pt[1])] for pt in quad[:4]])
            else:
                quads.append([])
        return quads
    except Exception:
        return []


def _build_cmd(mode: str, args: dict[str, Any]) -> list[str]:
    cmd = [str(PYTHON_EXE), "pipeline_runner.py", mode]
    for key, value in args.items():
        if value is None or value == "":
            continue
        flag = "--" + key.replace("_", "-")
        cmd.append(flag)
        value_str = str(value).replace("\\", "/")
        if value_str.startswith("outputs/"):
            value_str = os.path.relpath(str((ROOT / value_str).resolve()), str(SRC_DIR)).replace("\\", "/")
        cmd.append(value_str)
    return cmd


def _run_job(job_id: str, cmd: list[str]) -> None:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "running"
            j["started_at"] = datetime.utcnow().isoformat()
    _save_jobs(jobs)

    proc = subprocess.Popen(
        cmd,
        cwd=str(SRC_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_build_subprocess_env(),
    )

    log_lines: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))
            if len(log_lines) > 2000:
                log_lines = log_lines[-2000:]
            line_text = line.rstrip("\n")
            m = re.search(r"OCR\s+\[(\d+)/(\d+)\]:\s*(.+)$", line_text)
            jobs = _load_jobs()
            for j in jobs:
                if j["id"] == job_id:
                    j["stdout"] = "\n".join(log_lines[-400:])
                    if m:
                        cur = int(m.group(1))
                        total = int(m.group(2))
                        name = m.group(3).strip()
                        pct = int((cur * 100) / total) if total > 0 else 0
                        j["progress"] = {
                            "current": cur,
                            "total": total,
                            "percent": pct,
                            "current_file": name,
                        }
            _save_jobs(jobs)

    proc.wait()

    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "success" if (proc.returncode or 0) == 0 else "failed"
            j["return_code"] = proc.returncode
            j["stdout"] = "\n".join(log_lines[-1000:])
            j["stderr"] = ""
            if j["status"] == "success" and j.get("progress", {}).get("total", 0) > 0:
                j["progress"]["current"] = j["progress"]["total"]
                j["progress"]["percent"] = 100
            j["finished_at"] = datetime.utcnow().isoformat()
    _save_jobs(jobs)


def _enqueue_job(mode: str, args: dict[str, Any]) -> dict[str, Any]:
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    job_id = str(uuid.uuid4())
    cmd = _build_cmd(mode, args)
    job = {
        "id": job_id,
        "mode": mode,
        "args": args,
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


def _enqueue_background_job(
    mode: str,
    args: dict[str, Any],
    target,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "mode": mode,
        "args": args,
        "cmd": [mode],
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    t = threading.Thread(target=target, args=(job_id, args), daemon=True)
    t.start()
    return job


app = FastAPI(title="Invoice OCR GCN Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
elif FRONTEND_FALLBACK_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_FALLBACK_DIR)), name="frontend")


@app.get("/")
def root() -> RedirectResponse:
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/frontend/")
    return RedirectResponse(url="/frontend/dashboard.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/features")
def features() -> dict[str, Any]:
    return {
        "labels": INVOICE_LABELS,
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
    return _enqueue_job(mode=req.mode, args=req.args)


@app.post("/api/pipeline/gcn-infer")
def gcn_infer(req: GcnInferRequest) -> dict[str, Any]:
    return _enqueue_job("gcn_infer", req.model_dump())


@app.post("/api/pipeline/pretrained")
def pretrained(req: PretrainedRequest) -> dict[str, Any]:
    return _enqueue_job("pretrained", req.model_dump())


@app.post("/api/pipeline/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    return _enqueue_job("evaluate", req.model_dump())


@app.post("/api/pipeline/test-gcn")
def test_gcn(req: TestGcnRequest) -> dict[str, Any]:
    return _enqueue_job("test_gcn", req.model_dump())


@app.post("/api/pipeline/preprocess-gcn-dataset")
def preprocess_dataset(req: PreprocessDatasetRequest) -> dict[str, Any]:
    return _enqueue_job("preprocess_gcn_dataset", req.model_dump())


@app.post("/api/pipeline/split-gcn-dataset")
def split_gcn_dataset(req: SplitDatasetRequest) -> dict[str, Any]:
    input_path = SRC_DIR / req.input_json
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset JSON not found: {req.input_json}")

    total_ratio = req.train_ratio + req.val_ratio + req.test_ratio
    if total_ratio <= 0:
        raise HTTPException(status_code=400, detail="Train/val/test ratios must sum to > 0")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    samples = list(data.get("samples", []))
    if not samples:
        raise HTTPException(status_code=400, detail=f"Empty dataset: {req.input_json}")

    if bool(req.shuffle):
        rnd = random.Random(req.seed)
        rnd.shuffle(samples)

    train_ratio = req.train_ratio / total_ratio
    val_ratio = req.val_ratio / total_ratio
    test_ratio = req.test_ratio / total_ratio

    total = len(samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    if total >= 3:
        if n_train <= 0:
            n_train = 1
        if req.val_ratio > 0 and n_val <= 0:
            n_val = 1
        n_test = total - n_train - n_val
        if req.test_ratio > 0 and n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        while n_train + n_val + n_test > total:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    meta = dict(data.get("meta", {}))
    meta["split_from"] = req.input_json
    meta["split_seed"] = req.seed
    meta["split_ratios"] = {
        "train": req.train_ratio,
        "val": req.val_ratio,
        "test": req.test_ratio,
    }

    def write_split(path_str: str, split_name: str, split_samples: list[dict[str, Any]]) -> str:
        out_path = SRC_DIR / path_str
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "meta": {**meta, "split": split_name, "num_graphs": len(split_samples)},
            "samples": split_samples,
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path.relative_to(SRC_DIR))

    train_out = write_split(req.output_train_json, "train", train_samples)
    val_out = write_split(req.output_val_json, "validation", val_samples)
    test_out = write_split(req.output_test_json, "test", test_samples)

    return {
        "status": "ok",
        "input_json": req.input_json,
        "total_graphs": total,
        "outputs": {
            "train": {"path": train_out, "graphs": len(train_samples)},
            "validation": {"path": val_out, "graphs": len(val_samples)},
            "test": {"path": test_out, "graphs": len(test_samples)},
        },
    }


@app.post("/api/pipeline/validate-label-csv")
def validate_label_csv(req: ValidateLabelCsvRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    total_rows = 0
    empty_label_rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if req.label_col not in (reader.fieldnames or []):
            raise HTTPException(status_code=400, detail=f"Missing label column: {req.label_col}")
        for idx, row in enumerate(reader, start=2):
            total_rows += 1
            if str(row.get(req.label_col, "")).strip() == "":
                empty_label_rows.append(idx)

    return {
        "ok": len(empty_label_rows) == 0,
        "input_csv": req.input_csv,
        "label_col": req.label_col,
        "total_rows": total_rows,
        "empty_label_count": len(empty_label_rows),
        "empty_label_sample_rows": empty_label_rows[:20],
    }


@app.get("/api/pipeline/labeling-preview")
def labeling_preview(input_csv: str, label_col: str = "label", limit: int = 30) -> dict[str, Any]:
    csv_path = SRC_DIR / input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {input_csv}")

    empty_rows = []
    total_rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if label_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing label column: {label_col}")
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            if str(row.get(label_col, "")).strip() == "":
                item = {"row_number": row_number}
                for k in ("doc_id", "text", "x1", "y1", "x2", "y2", label_col):
                    if k in row:
                        item[k] = row.get(k, "")
                empty_rows.append(item)
                if len(empty_rows) >= max(1, min(limit, 200)):
                    break

    return {
        "input_csv": input_csv,
        "label_col": label_col,
        "total_rows": total_rows,
        "empty_label_preview": empty_rows,
        "allowed_labels": INVOICE_LABELS,
    }


@app.post("/api/pipeline/labeling-apply")
def labeling_apply(req: ApplyLabelUpdatesRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    updates_map = {u.row_number: u.label.strip() for u in req.updates if u.label.strip()}
    if not updates_map:
        return {"updated": 0, "input_csv": req.input_csv}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if req.label_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing label column: {req.label_col}")
        rows = list(reader)

    updated = 0
    for idx, row in enumerate(rows, start=2):
        if idx in updates_map:
            row[req.label_col] = updates_map[idx]
            updated += 1

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {"updated": updated, "input_csv": req.input_csv}


def _suggest_label_from_text(text: str) -> str:
    t = text.strip()
    low = t.lower()

    if re.search(r"\b(cash|ti[eê]n m[aă]t|momo|zalopay|gopay|gojek|card|visa|mastercard|atm)\b", low):
        return "PAYMENT_METHOD"
    if re.search(r"\b(thu[eê] ng[aâ]n|cashier|qu[aà]y|counter)\b", low):
        return "CASHIER"
    if re.search(r"\b(mst|m[aã]\s*s[oố]\s*thu[eế]|tax code)\b", low):
        return "TAX_CODE"
    if re.search(r"\b(sdt|đi[eệ]n tho[aạ]i|phone|tel|hotline)\b", low) or re.fullmatch(r"[+()0-9.\-\s]{8,20}", t):
        return "MERCHANT_PHONE"
    if re.search(r"\b(đ[ịi]a ch[ỉi]|address)\b", low):
        return "MERCHANT_ADDRESS"
    if re.search(r"\b(h[oó]a đ[oơ]n|invoice|bill\s*no|m[aã]\s*hd|s[oố]\s*hd|ref)\b", low):
        return "INVOICE_ID"
    if re.search(r"\b(ng[aà]y|date)\b", low) or re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", low):
        return "DATE"
    if re.search(r"\b(gi[oờ]|time)\b", low) or re.search(r"\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b", low):
        return "TIME"
    if re.search(r"\b(t[aạ]m t[ií]nh|subtotal)\b", low):
        return "SUBTOTAL"
    if re.search(r"\b(ph[ií]\s*d[iị]ch v[uụ]|service fee)\b", low):
        return "SERVICE_FEE"
    if re.search(r"\b(gi[aả]m gi[aá]|discount)\b", low):
        return "DISCOUNT"
    if re.search(r"\b(vat|thu[eế])\b", low):
        return "TAX_AMOUNT"
    if re.search(r"\b(t[oổ]ng c[oộ]ng|th[aà]nh ti[eề]n|total)\b", low):
        return "TOTAL_AMOUNT"
    if re.search(r"\b(s[lố]?\s*l[uượ]ng|qty|quantity|x\d+)\b", low):
        return "ITEM_QTY"
    if re.search(r"\b(đ[oơ]n gi[aá]|unit price|price|đ\/|vnd)\b", low):
        return "ITEM_UNIT_PRICE"
    if re.search(r"\b(amount|ti[eề]n)\b", low):
        return "ITEM_AMOUNT"
    if len(t) > 2 and any(c.isalpha() for c in t):
        return "ITEM_NAME"
    return "OTHER"


def _extract_json_array(raw_text: str) -> list[str]:
    txt = raw_text.strip()
    start = txt.find("[")
    end = txt.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output is not a JSON array")
    arr = json.loads(txt[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("LLM output must be list")
    out = []
    for item in arr:
        label = str(item).strip().upper()
        if label not in INVOICE_LABELS:
            label = "OTHER"
        out.append(label)
    return out


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text or "") if unicodedata.category(ch) != "Mn"
    )


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(text).lower()).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _money_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(re.fullmatch(r"[-+]?\d[\d.,:/-]*", t)) and any(ch.isdigit() for ch in t)


def _date_like(text: str) -> bool:
    return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", text or ""))


def _time_like(text: str) -> bool:
    return bool(re.search(r"\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b", text or ""))


def _phone_like(text: str) -> bool:
    t = (text or "").strip()
    digits = re.sub(r"\D", "", t)
    return 8 <= len(digits) <= 15


def _tax_code_like(text: str) -> bool:
    digits = re.sub(r"\D", "", text or "")
    return 8 <= len(digits) <= 14


def _build_line_groups(nodes: list[dict[str, Any]]) -> list[list[int]]:
    if not nodes:
        return []
    sorted_ids = sorted(range(len(nodes)), key=lambda i: (nodes[i]["cy"], nodes[i]["cx"]))
    heights = sorted(max(1.0, n["h"]) for n in nodes)
    median_h = heights[len(heights) // 2] if heights else 20.0
    threshold = max(10.0, median_h * 0.65)
    groups: list[list[int]] = []
    current: list[int] = []
    current_cy = None
    for idx in sorted_ids:
        cy = nodes[idx]["cy"]
        if current_cy is None or abs(cy - current_cy) <= threshold:
            current.append(idx)
            current_cy = cy if current_cy is None else (current_cy * (len(current) - 1) + cy) / len(current)
        else:
            groups.append(sorted(current, key=lambda i: nodes[i]["cx"]))
            current = [idx]
            current_cy = cy
    if current:
        groups.append(sorted(current, key=lambda i: nodes[i]["cx"]))
    return groups


def _doc_context_from_rows(
    rows: list[dict[str, Any]],
    row_indices: list[int],
    *,
    text_col: str,
    label_col: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    max_x = 1.0
    max_y = 1.0
    for ridx in row_indices:
        row = rows[ridx]
        x1 = _safe_float(row.get("x1"))
        y1 = _safe_float(row.get("y1"))
        x2 = _safe_float(row.get("x2"))
        y2 = _safe_float(row.get("y2"))
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        nodes.append(
            {
                "csv_index": ridx,
                "row_number": ridx + 2,
                "text": str(row.get(text_col, "") or "").strip(),
                "label": str(row.get(label_col, "") or "").strip().upper(),
                "score": _safe_float(row.get("score"), 1.0),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )

    for node in nodes:
        node["nx1"] = round(node["x1"] / max_x, 4)
        node["ny1"] = round(node["y1"] / max_y, 4)
        node["nx2"] = round(node["x2"] / max_x, 4)
        node["ny2"] = round(node["y2"] / max_y, 4)
        node["ncx"] = round(node["cx"] / max_x, 4)
        node["ncy"] = round(node["cy"] / max_y, 4)
        node["nw"] = round(node["w"] / max_x, 4)
        node["nh"] = round(node["h"] / max_y, 4)
        node["norm_text"] = _norm_text(node["text"])
        node["is_money_like"] = _money_like(node["text"])
        node["is_date_like"] = _date_like(node["text"])
        node["is_time_like"] = _time_like(node["text"])
        node["is_phone_like"] = _phone_like(node["text"])
        node["is_tax_code_like"] = _tax_code_like(node["text"])

    line_groups = _build_line_groups(nodes)
    summary_words = [
        "tong cong",
        "tong tien",
        "thanh tien",
        "tam tinh",
        "subtotal",
        "vat",
        "thue",
        "giam gia",
        "discount",
        "phi dich vu",
        "service fee",
        "tien khach tra",
        "tien mat",
        "thoi lai",
        "can thanh toan",
        "thanh toan",
    ]
    item_header_words = ["ten hang", "mat hang", "sl", "so luong", "dvt", "don gia", "thanh tien", "qty", "amount"]

    summary_line_ids: set[int] = set()
    item_header_line_id: int | None = None
    for line_id, group in enumerate(line_groups):
        line_text = " | ".join(nodes[i]["norm_text"] for i in group if nodes[i]["norm_text"])
        if any(word in line_text for word in summary_words):
            summary_line_ids.add(line_id)
        if item_header_line_id is None and sum(1 for word in item_header_words if word in line_text) >= 2:
            item_header_line_id = line_id

    first_summary_line = min(summary_line_ids) if summary_line_ids else max(0, int(len(line_groups) * 0.7))
    header_boundary = item_header_line_id if item_header_line_id is not None else max(1, int(len(line_groups) * 0.25))

    line_meta: list[dict[str, Any]] = []
    for line_id, group in enumerate(line_groups):
        texts = [nodes[i]["text"] for i in group]
        line_text = " | ".join(texts)
        if line_id < header_boundary:
            region = "header"
        elif line_id >= first_summary_line:
            region = "summary"
        else:
            region = "items"
        line_meta.append({"line_id": line_id, "texts": texts, "text": line_text, "region": region})
        for pos, idx in enumerate(group):
            node = nodes[idx]
            node["line_id"] = line_id
            node["line_region"] = region
            node["line_pos"] = pos
            node["same_line_texts"] = texts[:]
            node["left_text"] = texts[pos - 1] if pos > 0 else ""
            node["right_text"] = texts[pos + 1] if pos + 1 < len(texts) else ""

    for node in nodes:
        same_column = []
        for other in nodes:
            if other is node:
                continue
            if abs(other["ncx"] - node["ncx"]) <= 0.08:
                same_column.append(other)
        above = [x for x in same_column if x["ncy"] < node["ncy"]]
        below = [x for x in same_column if x["ncy"] > node["ncy"]]
        above.sort(key=lambda x: node["ncy"] - x["ncy"])
        below.sort(key=lambda x: x["ncy"] - node["ncy"])
        node["top_text"] = above[0]["text"] if above else ""
        node["bottom_text"] = below[0]["text"] if below else ""

        column_hint = ""
        if item_header_line_id is not None and node["line_region"] == "items":
            headers = line_groups[item_header_line_id]
            best = None
            best_dist = 999.0
            for hidx in headers:
                header_node = nodes[hidx]
                dist = abs(header_node["ncx"] - node["ncx"])
                if dist < best_dist:
                    best_dist = dist
                    best = header_node["norm_text"]
            column_hint = best or ""
        node["column_hint"] = column_hint

        if node["line_region"] == "summary":
            node["row_role_hint"] = "summary_area"
        elif node["line_region"] == "header":
            node["row_role_hint"] = "header_area"
        elif column_hint:
            node["row_role_hint"] = f"item_row:{column_hint}"
        else:
            node["row_role_hint"] = "item_row"

    doc_summary = {
        "line_count": len(line_groups),
        "header_lines": [x["text"] for x in line_meta[: min(6, len(line_meta))]],
        "summary_lines": [x["text"] for x in line_meta[first_summary_line : first_summary_line + 6]],
        "item_header_line": line_meta[item_header_line_id]["text"] if item_header_line_id is not None else "",
        "detected_regions": {
            "header_end_line": header_boundary,
            "summary_start_line": first_summary_line,
        },
    }
    return nodes, doc_summary


def _heuristic_label_for_node(node: dict[str, Any]) -> tuple[str | None, float, str]:
    text = node["text"]
    norm = node["norm_text"]
    left = _norm_text(node.get("left_text", ""))
    right = _norm_text(node.get("right_text", ""))
    same_line = " | ".join(_norm_text(x) for x in node.get("same_line_texts", []))
    around = " | ".join(filter(None, [left, right, _norm_text(node.get("top_text", "")), _norm_text(node.get("bottom_text", "")), same_line]))
    region = node.get("line_region", "")
    column_hint = node.get("column_hint", "")

    if not text.strip():
        return "OTHER", 1.0, "empty_text"
    if node["is_date_like"]:
        return "DATE", 0.98, "date_pattern"
    if node["is_time_like"]:
        return "TIME", 0.98, "time_pattern"
    if re.search(r"\b(cash|momo|zalopay|gopay|gojek|visa|mastercard|atm|card|chuyen khoan|tien mat)\b", norm):
        return "PAYMENT_METHOD", 0.98, "payment_keyword"
    if re.search(r"\b(cashier|thu ngan|quay|counter)\b", norm):
        return "CASHIER", 0.95, "cashier_keyword"
    if re.search(r"\b(mst|ma so thue|tax code)\b", norm) or (
        node["is_tax_code_like"] and ("thue" in around or "mst" in around)
    ):
        return "TAX_CODE", 0.95, "tax_context"
    if re.search(r"\b(dien thoai|phone|tel|hotline|sdt)\b", norm) or (
        node["is_phone_like"] and any(x in around for x in ["dien thoai", "phone", "tel", "hotline", "sdt"])
    ):
        return "MERCHANT_PHONE", 0.94, "phone_context"
    if re.search(r"\b(dia chi|address)\b", norm):
        return "MERCHANT_ADDRESS", 0.92, "address_keyword"
    if re.search(r"\b(hoa don|invoice|bill no|so hd|ma hd|ref)\b", norm):
        return "INVOICE_ID", 0.92, "invoice_keyword"

    if node["is_money_like"]:
        if any(x in around for x in ["giam gia", "discount"]):
            return "DISCOUNT", 0.96, "discount_context"
        if any(x in around for x in ["vat", "thue"]):
            return "TAX_AMOUNT", 0.96, "tax_amount_context"
        if any(x in around for x in ["phi dich vu", "service fee"]):
            return "SERVICE_FEE", 0.96, "service_fee_context"
        if any(x in around for x in ["tam tinh", "subtotal"]):
            return "SUBTOTAL", 0.96, "subtotal_context"
        if any(x in around for x in ["tong cong", "tong tien", "can thanh toan", "thanh toan", "tong thanh toan"]):
            return "TOTAL_AMOUNT", 0.97, "total_context"
        if region == "items":
            if any(x in column_hint for x in ["sl", "so luong", "qty"]):
                return "ITEM_QTY", 0.93, "item_qty_column"
            if any(x in column_hint for x in ["don gia", "price"]):
                return "ITEM_UNIT_PRICE", 0.93, "item_unit_price_column"
            if any(x in column_hint for x in ["thanh tien", "amount"]):
                return "ITEM_AMOUNT", 0.93, "item_amount_column"
        if region == "summary":
            return "TOTAL_AMOUNT", 0.65, "summary_money_fallback"

    if region == "header":
        if len(norm) >= 6 and any(ch.isalpha() for ch in norm):
            if any(x in norm for x in ["vin", "mart", "co.op", "minimart", "store", "shop", "coffee", "tra sua", "quan"]):
                return "MERCHANT_NAME", 0.85, "merchant_name_like"
            if "dia chi" in around:
                return "MERCHANT_ADDRESS", 0.8, "header_address_context"
    if region == "items":
        if any(x in column_hint for x in ["ten hang", "mat hang", "item", "hang"]):
            if any(ch.isalpha() for ch in norm):
                return "ITEM_NAME", 0.88, "item_name_column"
        if any(ch.isalpha() for ch in norm) and not node["is_money_like"]:
            return "ITEM_NAME", 0.62, "item_alpha_fallback"

    return None, 0.0, "needs_llm"


def _build_doc_llm_payload(nodes: list[dict[str, Any]], target_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    compact_nodes = [
        {
            "row_number": n["row_number"],
            "text": n["text"],
            "line_id": n.get("line_id"),
            "region": n.get("line_region"),
            "column_hint": n.get("column_hint", ""),
            "bbox_norm": [n["nx1"], n["ny1"], n["nx2"], n["ny2"]],
            "current_label": n.get("label", ""),
            "heuristic_label": n.get("heuristic_label"),
            "heuristic_confidence": n.get("heuristic_confidence", 0.0),
            "heuristic_reason": n.get("heuristic_reason", ""),
        }
        for n in nodes
    ]
    detailed_targets = [
        {
            "row_number": n["row_number"],
            "text": n["text"],
            "bbox_norm": [n["nx1"], n["ny1"], n["nx2"], n["ny2"]],
            "line_id": n.get("line_id"),
            "line_region": n.get("line_region"),
            "row_role_hint": n.get("row_role_hint"),
            "column_hint": n.get("column_hint", ""),
            "left_text": n.get("left_text", ""),
            "right_text": n.get("right_text", ""),
            "top_text": n.get("top_text", ""),
            "bottom_text": n.get("bottom_text", ""),
            "same_line_texts": n.get("same_line_texts", []),
            "is_money_like": n.get("is_money_like"),
            "is_date_like": n.get("is_date_like"),
            "is_time_like": n.get("is_time_like"),
            "is_phone_like": n.get("is_phone_like"),
            "is_tax_code_like": n.get("is_tax_code_like"),
            "ocr_score": n.get("score", 1.0),
            "heuristic_label": n.get("heuristic_label"),
            "heuristic_confidence": n.get("heuristic_confidence", 0.0),
            "heuristic_reason": n.get("heuristic_reason", ""),
        }
        for n in target_nodes
    ]
    seeded_nodes = [
        {
            "row_number": n["row_number"],
            "text": n["text"],
            "heuristic_label": n.get("heuristic_label"),
            "heuristic_confidence": n.get("heuristic_confidence", 0.0),
            "region": n.get("line_region"),
            "column_hint": n.get("column_hint", ""),
        }
        for n in nodes
        if n.get("heuristic_label")
    ]
    return {
        "document_nodes": compact_nodes,
        "seeded_nodes": seeded_nodes[:80],
        "target_nodes": detailed_targets,
    }


def _llm_suggest_labels_for_doc(
    *,
    doc_id: str,
    doc_summary: dict[str, Any],
    nodes: list[dict[str, Any]],
    target_nodes: list[dict[str, Any]],
    labels: list[str],
    model: str,
) -> dict[int, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    prompt = {
        "task": "Classify OCR nodes from one Vietnamese invoice/receipt using document structure, neighboring texts, line grouping, and node position.",
        "labels": labels,
        "rules": [
            "Return exactly one label per target node.",
            "Use node position, same-line texts, left/right/top/bottom neighbors, and document region.",
            "You will also receive heuristic labels already assigned to many nodes in the same document.",
            "Treat high-confidence heuristic labels on nearby nodes as strong context for classifying target nodes.",
            "If a target node itself has a heuristic_label with high confidence, keep it unless surrounding document structure strongly contradicts it.",
            "A numeric text alone is ambiguous. Do not classify money values from text alone.",
            "If a money value is in item rows near quantity/unit price/amount columns, prefer ITEM_QTY / ITEM_UNIT_PRICE / ITEM_AMOUNT.",
            "If a money value is in summary area near Tong cong / Tam tinh / VAT / Giam gia / Thanh toan, prefer the corresponding summary label.",
            "Merchant info usually appears near the top; payment method and final totals usually appear near the bottom.",
            "If still uncertain, use OTHER.",
            "Do not invent new labels.",
            "If uncertain, use OTHER.",
        ],
        "few_shot_examples": [
            {
                "node_text": "1500000",
                "same_line_texts": ["Tổng cộng", "1500000"],
                "region": "summary",
                "expected_label": "TOTAL_AMOUNT",
            },
            {
                "node_text": "1500000",
                "same_line_texts": ["Coca Cola", "2", "750000", "1500000"],
                "region": "items",
                "column_hint": "thành tiền",
                "expected_label": "ITEM_AMOUNT",
            },
            {
                "node_text": "750000",
                "same_line_texts": ["Coca Cola", "2", "750000", "1500000"],
                "region": "items",
                "column_hint": "đơn giá",
                "expected_label": "ITEM_UNIT_PRICE",
            },
            {
                "node_text": "7/06/2026",
                "same_line_texts": ["Ngày", "7/06/2026"],
                "region": "header",
                "expected_label": "DATE",
            },
            {
                "node_text": "0988123456",
                "same_line_texts": ["Điện thoại", "0988123456"],
                "region": "header",
                "expected_label": "MERCHANT_PHONE",
            },
            {
                "description": "Use neighboring seeded nodes as context",
                "seeded_nodes": [
                    {"text": "SL", "heuristic_label": "OTHER"},
                    {"text": "Đơn giá", "heuristic_label": "OTHER"},
                    {"text": "Thành tiền", "heuristic_label": "OTHER"},
                    {"text": "Coca Cola", "heuristic_label": "ITEM_NAME"},
                    {"text": "2", "heuristic_label": "ITEM_QTY"},
                ],
                "target_node": {
                    "node_text": "1500000",
                    "region": "items",
                    "column_hint": "thành tiền",
                },
                "expected_label": "ITEM_AMOUNT",
            },
        ],
        "doc_id": doc_id,
        "document_summary": doc_summary,
        "context": _build_doc_llm_payload(nodes, target_nodes),
    }

    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Vietnamese invoice OCR node labeler. "
                    "You classify OCR nodes inside a document, not isolated strings."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            {
                "role": "user",
                "content": (
                    "Return JSON object only with this schema: "
                    "{\"labels\": [{\"row_number\": 12, \"label\": \"TOTAL_AMOUNT\"}, ...]}"
                ),
            },
        ],
    }

    req = urllib_request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    obj = json.loads(content)
    arr = obj.get("labels", [])
    if not isinstance(arr, list):
        raise ValueError("LLM output must contain labels list")
    out: dict[int, str] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            row_number = int(item.get("row_number"))
        except Exception:
            continue
        label = str(item.get("label", "")).strip().upper()
        if label not in labels:
            label = "OTHER"
        out[row_number] = label
    return out


def _ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=400,
            detail="Missing OPENAI_API_KEY. Add it to DATN/.env or current environment, then restart backend.",
        )


def _suggest_labels_for_document(
    rows: list[dict[str, Any]],
    row_indices: list[int],
    *,
    text_col: str,
    label_col: str,
    llm_model: str,
    use_llm: bool,
    max_targets_per_request: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    nodes, doc_summary = _doc_context_from_rows(rows, row_indices, text_col=text_col, label_col=label_col)
    row_to_node = {node["row_number"]: node for node in nodes}
    assigned: dict[int, str] = {}
    llm_targets: list[dict[str, Any]] = []
    heuristic_hits = 0

    for node in nodes:
        label, confidence, reason = _heuristic_label_for_node(node)
        node["heuristic_label"] = label
        node["heuristic_confidence"] = confidence
        node["heuristic_reason"] = reason
        if label and confidence >= 0.9:
            assigned[node["row_number"]] = label
            heuristic_hits += 1
        else:
            llm_targets.append(node)

    strategy_used = "heuristic"
    llm_batches = 0
    llm_errors: list[str] = []
    if llm_targets and use_llm:
        strategy_used = "llm+heuristic"
        for s in range(0, len(llm_targets), max_targets_per_request):
            chunk = llm_targets[s : s + max_targets_per_request]
            try:
                llm_map = _llm_suggest_labels_for_doc(
                    doc_id=str(rows[row_indices[0]].get("doc_id", "UNKNOWN_DOC")),
                    doc_summary=doc_summary,
                    nodes=nodes,
                    target_nodes=chunk,
                    labels=INVOICE_LABELS,
                    model=llm_model,
                )
                llm_batches += 1
                for node in chunk:
                    assigned[node["row_number"]] = llm_map.get(
                        node["row_number"],
                        _suggest_label_from_text(node["text"]),
                    )
            except Exception as exc:
                llm_errors.append(str(exc))
                strategy_used = "heuristic+rule_fallback"
                for node in chunk:
                    fallback = node.get("heuristic_label") or _suggest_label_from_text(node["text"])
                    assigned[node["row_number"]] = fallback
    else:
        for node in llm_targets:
            assigned[node["row_number"]] = node.get("heuristic_label") or _suggest_label_from_text(node["text"])

    for row_number, label in list(assigned.items()):
        if label not in INVOICE_LABELS:
            assigned[row_number] = "OTHER"

    stats = {
        "heuristic_hits": heuristic_hits,
        "llm_targets": len(llm_targets),
        "llm_batches": llm_batches,
        "strategy_used": strategy_used,
        "llm_errors": llm_errors,
        "doc_summary": doc_summary,
        "target_preview": [
            {
                "row_number": node["row_number"],
                "text": node["text"],
                "region": node.get("line_region"),
                "column_hint": node.get("column_hint", ""),
                "heuristic_label": node.get("heuristic_label"),
                "heuristic_confidence": node.get("heuristic_confidence"),
            }
            for node in llm_targets[:8]
        ],
        "assigned_preview": [
            {
                "row_number": node["row_number"],
                "text": node["text"],
                "label": assigned.get(node["row_number"], "OTHER"),
                "line_region": node.get("line_region"),
                "column_hint": node.get("column_hint", ""),
            }
            for node in nodes[:10]
        ],
    }
    return assigned, stats


def _run_labeling_auto_job(job_id: str, args: dict[str, Any]) -> None:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "running"
            j["started_at"] = datetime.utcnow().isoformat()
            j["stdout"] = "Khoi dong AI goi y nhan..."
    _save_jobs(jobs)

    csv_path = SRC_DIR / args["input_csv"]
    label_col = args.get("label_col", "label")
    text_col = args.get("text_col", "text")
    doc_id_col = args.get("doc_id_col", "doc_id")
    # Do not overwrite labels that already exist. This keeps reruns incremental.
    only_empty = True
    llm_model = args.get("llm_model", "gpt-4.1-mini")
    batch_docs = max(1, min(int(args.get("batch_docs", 10)), 50))
    llm_text_batch_size = max(1, min(int(args.get("llm_text_batch_size", 10)), 50))
    require_llm = int(args.get("require_llm", 1)) == 1

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            for col in (label_col, text_col, doc_id_col):
                if col not in fields:
                    raise ValueError(f"Missing column: {col}")
            rows = list(reader)

        grouped: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            old_label = str(row.get(label_col, "") or "").strip()
            if only_empty and old_label:
                continue
            doc_id = str(row.get(doc_id_col, "") or "").strip() or "UNKNOWN_DOC"
            grouped.setdefault(doc_id, []).append(i)

        doc_ids = list(grouped.keys())
        total_docs = len(doc_ids)
        logs: list[str] = [f"Tong so anh can goi y: {total_docs}"]
        done_docs = 0
        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["progress"] = {
                    "current_docs": 0,
                    "total_docs": total_docs,
                    "percent": 0,
                    "current_batch_docs": 0,
                }
                j["stdout"] = "\n".join(logs)
        _save_jobs(jobs)

        for s in range(0, total_docs, batch_docs):
            batch_doc_ids = doc_ids[s : s + batch_docs]
            strategy_used = "heuristic"
            batch_row_total = 0
            for did in batch_doc_ids:
                doc_row_indices = grouped[did]
                batch_row_total += len(doc_row_indices)
                assigned, stats = _suggest_labels_for_document(
                    rows,
                    doc_row_indices,
                    text_col=text_col,
                    label_col=label_col,
                    llm_model=llm_model,
                    use_llm=require_llm,
                    max_targets_per_request=llm_text_batch_size,
                )
                strategy_used = stats["strategy_used"]
                for ridx in doc_row_indices:
                    row_number = ridx + 2
                    if row_number in assigned:
                        rows[ridx][label_col] = assigned[row_number]
                logs.append(
                    f"doc={did} rows={len(doc_row_indices)} heuristic={stats['heuristic_hits']} "
                    f"llm_targets={stats['llm_targets']} llm_batches={stats['llm_batches']} mode={stats['strategy_used']}"
                )
                if stats["llm_errors"]:
                    logs.append(f"doc={did} llm_error={stats['llm_errors'][-1]}")

            done_docs += len(batch_doc_ids)
            pct = max(1, round((done_docs * 100) / total_docs)) if total_docs > 0 else 100
            _write_csv_rows(csv_path, fields, rows)
            logs.append(
                f"[{done_docs}/{total_docs}] batch_docs={len(batch_doc_ids)} rows={batch_row_total} mode={strategy_used} saved=ok"
            )

            jobs = _load_jobs()
            for j in jobs:
                if j["id"] == job_id:
                    j["progress"] = {
                        "current_docs": done_docs,
                        "total_docs": total_docs,
                        "percent": pct,
                        "current_batch_docs": len(batch_doc_ids),
                    }
                    j["stdout"] = "\n".join(logs[-120:])
            _save_jobs(jobs)

        _write_csv_rows(csv_path, fields, rows)

        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["status"] = "success"
                j["return_code"] = 0
                j["stdout"] = "\n".join(logs[-200:] + ["Hoan tat AI goi y nhan."])
                j["stderr"] = ""
                j["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs(jobs)
    except Exception as exc:
        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["status"] = "failed"
                j["return_code"] = 1
                j["stderr"] = str(exc)
                j["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs(jobs)


@app.post("/api/pipeline/labeling-auto-suggest")
def labeling_auto_suggest(req: AutoSuggestLabelsRequest) -> dict[str, Any]:
    if req.strategy == "llm":
        _ensure_openai_api_key()

    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if req.label_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing label column: {req.label_col}")
        if req.text_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing text column: {req.text_col}")
        rows = list(reader)

    targets: list[int] = []
    effective_only_empty = True
    for i, row in enumerate(rows):
        old_label = str(row.get(req.label_col, "")).strip()
        if effective_only_empty and old_label:
            continue
        targets.append(i)

    strategy_used = req.strategy
    grouped: dict[str, list[int]] = {}
    for idx in targets:
        doc_id = str(rows[idx].get("doc_id", "") or "").strip() or "UNKNOWN_DOC"
        grouped.setdefault(doc_id, []).append(idx)

    meta: list[dict[str, Any]] = []
    if targets:
        for did, doc_row_indices in grouped.items():
            assigned, stats = _suggest_labels_for_document(
                rows,
                doc_row_indices,
                text_col=req.text_col,
                label_col=req.label_col,
                llm_model=req.llm_model,
                use_llm=req.strategy == "llm",
                max_targets_per_request=max(1, min(req.batch_size, 50)),
            )
            strategy_used = stats["strategy_used"]
            for ridx in doc_row_indices:
                row_number = ridx + 2
                if row_number in assigned:
                    rows[ridx][req.label_col] = assigned[row_number]
            meta.append(
                {
                    "doc_id": did,
                    "row_count": len(doc_row_indices),
                    "heuristic_hits": stats["heuristic_hits"],
                    "llm_targets": stats["llm_targets"],
                    "llm_batches": stats["llm_batches"],
                    "strategy_used": stats["strategy_used"],
                    "llm_errors": stats["llm_errors"],
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input_csv": req.input_csv,
        "suggested_rows": len(targets),
        "strategy_requested": req.strategy,
        "strategy_used": strategy_used,
        "llm_model": req.llm_model if "llm" in strategy_used else None,
        "label_col": req.label_col,
        "only_empty_enforced": True,
        "labels": INVOICE_LABELS,
        "doc_stats": meta[:30],
    }


@app.post("/api/pipeline/labeling-auto-suggest-start")
def labeling_auto_suggest_start(req: AutoSuggestStartRequest) -> dict[str, Any]:
    if int(req.require_llm) == 1:
        _ensure_openai_api_key()

    return _enqueue_background_job(
        mode="labeling_auto_suggest",
        args=req.model_dump(),
        target=_run_labeling_auto_job,
    )


@app.post("/api/pipeline/labeling-sample")
def labeling_sample(req: LabelingSampleRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if req.label_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing label column: {req.label_col}")
        if req.text_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing text column: {req.text_col}")

        rows = []
        total_rows = 0
        empty_count = 0
        limit = max(1, min(req.limit, 500))
        page = max(1, req.page)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            cur_label = str(row.get(req.label_col, "") or "").strip()
            if not cur_label:
                empty_count += 1
            row_pos = total_rows - 1
            if start_idx <= row_pos < end_idx:
                rows.append(
                    {
                        "row_number": row_number,
                        "doc_id": row.get("doc_id", ""),
                        "text": row.get(req.text_col, ""),
                        "label": cur_label,
                    }
                )

    return {
        "input_csv": req.input_csv,
        "total_rows": total_rows,
        "page": page,
        "page_size": limit,
        "total_pages": max(1, (total_rows + limit - 1) // limit),
        "empty_label_count": empty_count,
        "rows": rows,
        "allowed_labels": INVOICE_LABELS,
    }


@app.post("/api/pipeline/labeling-by-doc")
def labeling_by_doc(req: LabelingByDocRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    grouped: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (req.doc_id_col, req.text_col, req.label_col):
            if col not in fields:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")
        for row in reader:
            doc_id = str(row.get(req.doc_id_col, "") or "").strip() or "UNKNOWN_DOC"
            text = str(row.get(req.text_col, "") or "").strip()
            label = str(row.get(req.label_col, "") or "").strip().upper() or "UNLABELED"
            slot = grouped.setdefault(
                doc_id,
                {"doc_id": doc_id, "total_nodes": 0, "empty_labels": 0, "labels": {}, "samples": []},
            )
            slot["total_nodes"] += 1
            if label == "UNLABELED":
                slot["empty_labels"] += 1
            slot["labels"][label] = slot["labels"].get(label, 0) + 1
            if len(slot["samples"]) < 8:
                slot["samples"].append({"text": text, "label": label})

    image_index = _stage_b_image_index(csv_path.parent)
    all_docs = list(grouped.values())
    page_size = max(1, min(req.page_size, 200))
    page = max(1, req.page)
    total_docs = len(all_docs)
    total_pages = max(1, (total_docs + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = start + page_size
    docs = all_docs[start:end]
    for d in docs:
        d["preview_path"] = _find_image_for_doc(d["doc_id"], image_index)

    return {
        "input_csv": req.input_csv,
        "total_docs": total_docs,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "docs": docs,
    }


@app.post("/api/pipeline/labeling-graph-inspect")
def labeling_graph_inspect(req: LabelingGraphInspectRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    doc_rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (
            req.doc_id_col,
            req.text_col,
            req.label_col,
            req.score_col,
            req.x1_col,
            req.y1_col,
            req.x2_col,
            req.y2_col,
        ):
            if col not in fields:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")
        for row_number, row in enumerate(reader, start=2):
            doc_id = str(row.get(req.doc_id_col, "") or "").strip()
            if doc_id == req.doc_id:
                row["_row_number"] = row_number
                doc_rows.append(row)

    if not doc_rows:
        raise HTTPException(status_code=404, detail=f"Doc not found: {req.doc_id}")

    doc_quads = _load_doc_quads_from_ocr_json(csv_path, req.doc_id)
    nodes: list[OCRNode] = []
    labels: list[str] = []
    row_numbers: list[int] = []
    for row_idx, row in enumerate(doc_rows):
        text = str(row.get(req.text_col, "") or "").strip()
        if not text:
            continue
        x1 = float(row.get(req.x1_col, 0) or 0)
        y1 = float(row.get(req.y1_col, 0) or 0)
        x2 = float(row.get(req.x2_col, 0) or 0)
        y2 = float(row.get(req.y2_col, 0) or 0)
        score = float(row.get(req.score_col, 1.0) or 1.0)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        nodes.append(
            OCRNode(
                text=text,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
                w=w,
                h=h,
                quad=tuple((float(pt[0]), float(pt[1])) for pt in (doc_quads[row_idx] if row_idx < len(doc_quads) else [])[:4]) or None,
            )
        )
        labels.append(str(row.get(req.label_col, "") or "").strip())
        row_numbers.append(int(row["_row_number"]))

    features = build_features(nodes).tolist() if nodes else []
    edges = build_graph_edges(nodes, same_line_ratio=req.same_line_ratio, near_threshold=req.near_threshold)
    edge_index = [[], []]
    adjacency = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for src, dst, dist in edges:
        edge_index[0].append(src)
        edge_index[1].append(dst)
        adjacency[src][dst] = 1

    image_index = _stage_b_image_index(csv_path.parent)
    preview_path = _find_image_for_doc(req.doc_id, image_index)
    feature_names = ["text_len", "has_digit", "has_money_token", "cx_norm", "cy_norm", "w_norm", "h_norm", "ocr_score"]
    node_items = []
    for i, node in enumerate(nodes):
        node_items.append(
            {
                "node_index": i,
                "row_number": row_numbers[i],
                "text": node.text,
                "label": labels[i],
                "score": node.score,
                "bbox": [node.x1, node.y1, node.x2, node.y2],
                "quad": [[px, py] for px, py in (node.quad or ())],
                "features": features[i],
            }
        )

    return {
        "doc_id": req.doc_id,
        "input_csv": req.input_csv,
        "preview_path": preview_path,
        "feature_names": feature_names,
        "graph": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "edge_index": edge_index,
            "adjacency_matrix": adjacency,
        },
        "nodes": node_items,
    }


@app.post("/api/pipeline/single-image-preview")
def single_image_preview(req: SingleImagePreviewRequest) -> dict[str, Any]:
    from pipeline.services.ocr_service import OCRService

    image_path = _resolve_project_path(req.image)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image}")

    output_dir = _resolve_project_path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_service = OCRService()
    ocr_overrides = {
        "det_db_thresh": req.det_db_thresh,
        "det_db_box_thresh": req.det_db_box_thresh,
        "det_db_unclip_ratio": req.det_db_unclip_ratio,
        "drop_score": req.drop_score,
        "use_dilation": req.use_dilation,
        "det_limit_side_len": req.det_limit_side_len,
        "upscale_factor": req.upscale_factor,
    }
    processed_image = ocr_service.prepare_image(str(image_path), overrides=ocr_overrides)
    ocr_nodes = ocr_service.run(
        str(image_path),
        lang=req.lang or "vi",
        engine=req.ocr_engine or "paddle",
        overrides=ocr_overrides,
    )

    doc_id = image_path.stem
    rows: list[dict[str, Any]] = []
    row_indices: list[int] = []
    for idx, node in enumerate(ocr_nodes):
        rows.append(
            {
                "doc_id": doc_id,
                "text": node.text,
                "label": "",
                "x1": f"{node.x1:.2f}",
                "y1": f"{node.y1:.2f}",
                "x2": f"{node.x2:.2f}",
                "y2": f"{node.y2:.2f}",
                "score": f"{node.score:.4f}",
            }
        )
        row_indices.append(idx)

    use_ai = bool(req.with_ai)
    assigned, ai_stats = _suggest_labels_for_document(
        rows,
        row_indices,
        text_col="text",
        label_col="label",
        llm_model=req.llm_model,
        use_llm=use_ai,
        max_targets_per_request=max(1, min(int(req.llm_text_batch_size), 50)),
    )
    context_nodes, doc_summary = _doc_context_from_rows(rows, row_indices, text_col="text", label_col="label")
    context_by_row = {node["row_number"]: node for node in context_nodes}
    for node in context_nodes:
        heuristic_label, heuristic_confidence, heuristic_reason = _heuristic_label_for_node(node)
        node["heuristic_label"] = heuristic_label
        node["heuristic_confidence"] = heuristic_confidence
        node["heuristic_reason"] = heuristic_reason

    feature_names = ["text_len", "has_digit", "has_money_token", "cx_norm", "cy_norm", "w_norm", "h_norm", "ocr_score"]
    features = build_features(ocr_nodes).tolist() if ocr_nodes else []
    edges = build_graph_edges(ocr_nodes, same_line_ratio=req.same_line_ratio, near_threshold=req.near_threshold)
    edge_index = [[], []]
    adjacency = [[0 for _ in range(len(ocr_nodes))] for _ in range(len(ocr_nodes))]
    for src, dst, _dist in edges:
        edge_index[0].append(src)
        edge_index[1].append(dst)
        adjacency[src][dst] = 1

    preview_image_path = output_dir / f"{doc_id}_normalized.jpg"
    if processed_image is not None:
        import cv2

        cv2.imwrite(str(preview_image_path), processed_image)
        preview_image_rel = _project_relative_path(preview_image_path)
    else:
        preview_image_rel = _project_relative_path(image_path)

    if bool(req.save_debug_image):
        debug_image_path = output_dir / f"{doc_id}_ocr_boxes.jpg"
        if processed_image is not None:
            ocr_service.save_debug_image_from_array(processed_image, ocr_nodes, str(debug_image_path))
        else:
            ocr_service.save_debug_image(str(image_path), ocr_nodes, str(debug_image_path))
        debug_image_rel = _project_relative_path(debug_image_path)
    else:
        debug_image_rel = ""

    csv_preview_path = output_dir / f"{doc_id}_nodes_to_label.csv"
    csv_fields = ["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"]
    with csv_preview_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            row_number = idx + 2
            writer.writerow(
                {
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "label": assigned.get(row_number, "") if use_ai else "",
                    "x1": row["x1"],
                    "y1": row["y1"],
                    "x2": row["x2"],
                    "y2": row["y2"],
                    "score": row["score"],
                }
            )

    node_items: list[dict[str, Any]] = []
    label_summary: dict[str, int] = {}
    extracted_fields: dict[str, list[str]] = {}
    for idx, node in enumerate(ocr_nodes):
        row_number = idx + 2
        ctx = context_by_row.get(row_number, {})
        final_label = assigned.get(row_number, "") if use_ai else ""
        suggested_label = assigned.get(row_number, "OTHER")
        if final_label:
            label_summary[final_label] = label_summary.get(final_label, 0) + 1
            extracted_fields.setdefault(final_label, []).append(node.text)
        node_items.append(
            {
                "node_index": idx,
                "row_number": row_number,
                "text": node.text,
                "label": final_label,
                "picked_label": final_label,
                "suggested_label": suggested_label,
                "heuristic_label": ctx.get("heuristic_label"),
                "heuristic_confidence": ctx.get("heuristic_confidence"),
                "heuristic_reason": ctx.get("heuristic_reason"),
                "line_id": ctx.get("line_id"),
                "line_region": ctx.get("line_region"),
                "column_hint": ctx.get("column_hint"),
                "row_role_hint": ctx.get("row_role_hint"),
                "score": node.score,
                "bbox": [node.x1, node.y1, node.x2, node.y2],
                "quad": [[px, py] for px, py in (node.quad or ())],
                "features": features[idx] if idx < len(features) else [],
            }
        )

    graph_preview_payload = {
        "doc_id": doc_id,
        "image_path": _project_relative_path(image_path),
        "ocr_boxes_image": debug_image_rel,
        "doc_summary": doc_summary,
        "feature_names": feature_names,
        "graph": {
            "num_nodes": len(ocr_nodes),
            "num_edges": len(edges),
            "edge_index": edge_index,
            "adjacency_matrix": adjacency,
        },
        "label_summary": label_summary,
        "extracted_fields": extracted_fields,
        "nodes": node_items,
        "ai_stats": ai_stats,
    }
    graph_preview_path = output_dir / f"{doc_id}_graph_preview.json"
    graph_preview_path.write_text(json.dumps(graph_preview_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "mode": "ocr_ai_preview" if use_ai else "ocr_preview",
        "doc_id": doc_id,
        "image_path": _project_relative_path(image_path),
        "preview_path": preview_image_rel,
        "ocr_boxes_image": debug_image_rel,
        "ocr_config": ocr_overrides,
        "artifacts": {
            "train_b_csv_path": _project_relative_path(csv_preview_path),
            "graph_json_path": _project_relative_path(graph_preview_path),
            "output_dir": _project_relative_path(output_dir),
        },
        "doc_summary": doc_summary,
        "feature_names": feature_names,
        "graph": {
            "num_nodes": len(ocr_nodes),
            "num_edges": len(edges),
            "edge_index": edge_index,
            "adjacency_matrix": adjacency,
        },
        "label_summary": label_summary,
        "extracted_fields": extracted_fields,
        "ai_stats": ai_stats,
        "nodes": node_items,
    }


@app.post("/api/pipeline/prepare-ocr-labeling")
def prepare_ocr_labeling(req: PrepareOcrLabelingRequest) -> dict[str, Any]:
    return _enqueue_job("prepare_ocr_labeling", req.model_dump())


@app.post("/api/pipeline/train-gcn-stage-a")
def train_stage_a(req: TrainGcnStageARequest) -> dict[str, Any]:
    return _enqueue_job("train_gcn_stage_a", req.model_dump())


@app.post("/api/pipeline/train-gcn-stage-b")
def train_stage_b(req: TrainGcnStageBRequest) -> dict[str, Any]:
    return _enqueue_job("train_gcn_stage_b", req.model_dump())


@app.post("/api/pipeline/train-gcn-full")
def train_full(req: TrainGcnFullRequest) -> dict[str, Any]:
    return _enqueue_job("train_gcn_full", req.model_dump())


@app.post("/api/pipeline/train-ocr")
def train_ocr(req: TrainOcrRequest) -> dict[str, Any]:
    return _enqueue_job("train_ocr", req.model_dump())


@app.post("/api/pipeline/convert-hf-cord-to-csv")
def convert_hf_cord(req: ConvertCordRequest) -> dict[str, Any]:
    return _enqueue_job("convert_hf_cord_to_csv", req.model_dump())


@app.post("/api/pipeline/convert-hf-to-gcn-csv")
def convert_hf_generic(req: ConvertGenericRequest) -> dict[str, Any]:
    return _enqueue_job("convert_hf_to_gcn_csv", req.model_dump())


@app.post("/api/files/upload-images")
async def upload_images(
    files: list[UploadFile] = File(...),
    subdir: str | None = Form(default=None),
) -> dict[str, Any]:
    target_dir = STAGE_B_RAW_DIR / (subdir or datetime.utcnow().strftime("batch_%Y%m%d_%H%M%S"))
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        dst = target_dir / Path(f.filename).name
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dst.relative_to(SRC_DIR)))

    return {
        "saved_count": len(saved),
        "input_dir": str(target_dir.relative_to(SRC_DIR)),
        "files": saved,
    }


@app.get("/api/files/stage-b-raw-images")
def list_stage_b_raw_images() -> dict[str, Any]:
    STAGE_B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in STAGE_B_RAW_DIR.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(SRC_DIR)))
    return {"input_dir": str(STAGE_B_RAW_DIR.relative_to(SRC_DIR)), "count": len(files), "files": files}


@app.get("/api/files/list")
def list_files(dir: str) -> dict[str, Any]:
    target = (SRC_DIR / dir).resolve()
    if not str(target).startswith(str(SRC_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid directory")
    if not target.exists() or not target.is_dir():
        return {"dir": dir, "count": 0, "files": []}
    files = []
    for p in target.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(SRC_DIR)))
    return {"dir": dir, "count": len(files), "files": files}


@app.get("/api/files/checkpoints")
def list_checkpoint_files() -> dict[str, Any]:
    files: list[str] = []
    seen: set[str] = set()

    for base_dir in (ROOT_OUTPUTS_CHECKPOINTS_DIR, SRC_DIR / "outputs" / "checkpoints"):
        if not base_dir.exists() or not base_dir.is_dir():
            continue
        for p in sorted(base_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".pt", ".pth", ".bin"}:
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            if rel not in seen:
                seen.add(rel)
                files.append(rel)

    return {"count": len(files), "files": files}


@app.post("/api/files/clear-stage-b-raw-images")
def clear_stage_b_raw_images() -> dict[str, Any]:
    STAGE_B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    removed_files = 0
    removed_dirs = 0

    for p in sorted(STAGE_B_RAW_DIR.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
            removed_files += 1
        elif p.is_dir():
            try:
                p.rmdir()
                removed_dirs += 1
            except OSError:
                pass

    return {
        "status": "ok",
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "target_dir": str(STAGE_B_RAW_DIR.relative_to(SRC_DIR)),
    }


@app.get("/api/files/image")
def file_image(path: str) -> FileResponse:
    target = _resolve_project_path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(target))


@app.get("/api/files/json")
def file_json(path: str) -> dict[str, Any]:
    target = _resolve_project_path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="JSON file not found")
    if target.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Only .json files are supported")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON content: {exc}") from exc
