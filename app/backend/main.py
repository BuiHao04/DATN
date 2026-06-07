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
    checkpoint: str | None = None
    ocr_debug_image: str = "outputs/ocr_boxes.jpg"
    output_json: str = "outputs/ocr_result.json"


class PretrainedRequest(BaseModel):
    image: str
    project_dir: str = "."
    lang: str = "en"
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


class PrepareOcrLabelingRequest(BaseModel):
    input_dir: str
    output_dir: str = "data/labeling_stage_b"
    lang: str = "en"
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


def _stage_b_image_index() -> list[Path]:
    STAGE_B_RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [
        p
        for p in STAGE_B_RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ]


def _find_image_for_doc(doc_id: str, image_index: list[Path]) -> str | None:
    did = doc_id.strip().lower()
    exact = next((p for p in image_index if p.stem.lower() == did), None)
    if exact:
        return str(exact.relative_to(SRC_DIR))
    fuzzy = next((p for p in image_index if did in p.stem.lower()), None)
    if fuzzy:
        return str(fuzzy.relative_to(SRC_DIR))
    return None


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


def _llm_suggest_labels(texts: list[str], labels: list[str], model: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    prompt = {
        "task": "Classify each OCR text snippet into one label.",
        "labels": labels,
        "rules": [
            "Return exactly one label per input text.",
            "Output JSON array only, no markdown, no explanation.",
            "If uncertain, use OTHER.",
        ],
        "inputs": texts,
    }

    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are an invoice entity labeler."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            {
                "role": "user",
                "content": "Return JSON object: {\"labels\": [\"...\", ...]} with same length as inputs.",
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
        arr = _extract_json_array(content)
    out = [str(x).strip().upper() if str(x).strip().upper() in labels else "OTHER" for x in arr]
    if len(out) < len(texts):
        # Some models occasionally return fewer labels than requested.
        # Keep the valid prefix and backfill the remainder with deterministic rules.
        out.extend(_suggest_label_from_text(t) for t in texts[len(out) :])
    elif len(out) > len(texts):
        out = out[: len(texts)]
    return out


def _ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=400,
            detail="Missing OPENAI_API_KEY. Add it to DATN/.env or current environment, then restart backend.",
        )


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
    llm_text_batch_size = max(1, min(int(args.get("llm_text_batch_size", 30)), 100))
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
            batch_row_indices: list[int] = []
            batch_texts: list[str] = []
            for did in batch_doc_ids:
                for ridx in grouped[did]:
                    batch_row_indices.append(ridx)
                    batch_texts.append(str(rows[ridx].get(text_col, "") or ""))

            labels_out: list[str] = []
            strategy_used = "llm"
            for t in range(0, len(batch_texts), llm_text_batch_size):
                try:
                    labels_out.extend(
                        _llm_suggest_labels(
                            batch_texts[t : t + llm_text_batch_size],
                            INVOICE_LABELS,
                            llm_model,
                        )
                    )
                except Exception as exc:
                    strategy_used = "rule"
                    fallback_batch = [_suggest_label_from_text(x) for x in batch_texts[t : t + llm_text_batch_size]]
                    labels_out.extend(fallback_batch)
                    logs.append(f"Batch {s//batch_docs + 1}: fallback rule ({exc})")

            for pos, ridx in enumerate(batch_row_indices):
                rows[ridx][label_col] = labels_out[pos]

            done_docs += len(batch_doc_ids)
            pct = max(1, round((done_docs * 100) / total_docs)) if total_docs > 0 else 100
            _write_csv_rows(csv_path, fields, rows)
            logs.append(
                f"[{done_docs}/{total_docs}] batch_docs={len(batch_doc_ids)} rows={len(batch_row_indices)} mode={strategy_used} saved=ok"
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
    texts: list[str] = []
    effective_only_empty = True
    for i, row in enumerate(rows):
        old_label = str(row.get(req.label_col, "")).strip()
        if effective_only_empty and old_label:
            continue
        targets.append(i)
        texts.append(str(row.get(req.text_col, "") or ""))

    strategy_used = req.strategy
    labels_out: list[str] = []
    if texts:
        if req.strategy == "llm":
            try:
                bsz = max(1, min(req.batch_size, 100))
                for s in range(0, len(texts), bsz):
                    batch = texts[s : s + bsz]
                    labels_out.extend(_llm_suggest_labels(batch, INVOICE_LABELS, req.llm_model))
            except Exception:
                strategy_used = "rule"
                labels_out = [_suggest_label_from_text(t) for t in texts]
        else:
            labels_out = [_suggest_label_from_text(t) for t in texts]

    for pos, idx in enumerate(targets):
        rows[idx][req.label_col] = labels_out[pos]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input_csv": req.input_csv,
        "suggested_rows": len(targets),
        "strategy_requested": req.strategy,
        "strategy_used": strategy_used,
        "llm_model": req.llm_model if strategy_used == "llm" else None,
        "label_col": req.label_col,
        "only_empty_enforced": True,
        "labels": INVOICE_LABELS,
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

    image_index = _stage_b_image_index()
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

    nodes: list[OCRNode] = []
    labels: list[str] = []
    row_numbers: list[int] = []
    for row in doc_rows:
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

    image_index = _stage_b_image_index()
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
