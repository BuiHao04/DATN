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
import time
import tempfile
import io
from urllib import request as urllib_request
from urllib import error as urllib_error
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from pipeline.core.gcn_classifier import build_features
from pipeline.core.graph_builder import build_graph_edges
from pipeline.core.schema import LABEL_MAP, OCRNode

FRONTEND_DIR = ROOT / "app" / "frontend" / "dist"
FRONTEND_FALLBACK_DIR = ROOT / "app" / "frontend"
JOBS_FILE = ROOT / "app" / "jobs" / "jobs.json"
STAGE_B_RAW_DIR = SRC_DIR / "data" / "stage_b_raw_images"
ROOT_OUTPUTS_CHECKPOINTS_DIR = ROOT / "outputs" / "checkpoints"
PYTHON_EXE = Path(sys.executable).resolve()
CONDA_ENV_DIR = PYTHON_EXE.parent
JOB_STOP_DIR = ROOT / "app" / "jobs" / "stop_flags"
ACTIVE_JOB_PROCS: dict[str, subprocess.Popen] = {}
JOBS_LOCK = threading.Lock()


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
JOB_STOP_DIR.mkdir(parents=True, exist_ok=True)

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
    reuse_ocr_json: int = 0
    det_db_thresh: float = 0.20
    det_db_box_thresh: float = 0.45
    det_db_unclip_ratio: float = 1.80
    drop_score: float = 0.25
    use_dilation: int = 0
    det_limit_side_len: int = 1920
    upscale_factor: float = 1.0


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
    label: str = ""
    text: str | None = None


class ApplyLabelUpdatesRequest(BaseModel):
    input_csv: str
    label_col: str = "label"
    text_col: str = "text"
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
    only_empty: int = 0


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


class LabelingSuggestDocRequest(BaseModel):
    input_csv: str
    doc_id: str
    label_col: str = "label"
    text_col: str = "text"
    doc_id_col: str = "doc_id"
    llm_model: str = "gpt-4.1-mini"
    only_empty: int = 1
    require_llm: int = 1
    batch_size: int = 12


class ExportTrainSubsetRequest(BaseModel):
    input_csv: str = "data/labeling_top1000_ppocrv6/nodes_to_label.csv"
    output_dir: str = "data/train_stage_b"
    limit: int = 1000  # stop after this many fully-labeled images
    label_col: str = "label"
    doc_id_col: str = "doc_id"
    copy_images: int = 1
    copy_ocr_json: int = 1


class SingleImagePreviewRequest(BaseModel):
    image: str
    lang: str = "vi"
    ocr_engine: str = "paddle"
    det_db_thresh: float = 0.20
    det_db_box_thresh: float = 0.45
    det_db_unclip_ratio: float = 1.80
    drop_score: float = 0.25
    use_dilation: int = 0
    det_limit_side_len: int = 1920
    upscale_factor: float = 1.0
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
    lang: str = "vi"
    ocr_engine: str = "paddle"
    det_db_thresh: float = 0.20
    det_db_box_thresh: float = 0.45
    det_db_unclip_ratio: float = 1.80
    drop_score: float = 0.25
    use_dilation: int = 0
    det_limit_side_len: int = 1920
    upscale_factor: float = 1.0
    save_debug_images: int = 1
    copy_images: int = 1
    save_every_images: int = 10
    overwrite_existing: int = 1


def _job_stop_flag_path(job_id: str) -> Path:
    return JOB_STOP_DIR / f"{job_id}.stop"


def _request_job_stop(job_id: str) -> dict[str, Any]:
    jobs = _load_jobs()
    target = None
    for j in jobs:
        if j["id"] == job_id:
            target = j
            break
    if not target:
        raise HTTPException(status_code=404, detail="Job not found")

    if target.get("status") in {"success", "failed", "stopped"}:
        return {
            "ok": True,
            "job_id": job_id,
            "status": target.get("status"),
            "message": "Job đã kết thúc trước đó.",
        }

    stop_flag = _job_stop_flag_path(job_id)
    stop_flag.write_text(str(time.time()), encoding="utf-8")
    target["stop_requested"] = True
    target["stop_requested_at"] = datetime.utcnow().isoformat()
    if target.get("status") in {"queued", "running"}:
        target["status"] = "stopping"
    _save_jobs(jobs)
    return {
        "ok": True,
        "job_id": job_id,
        "status": target.get("status"),
        "message": "Đã gửi yêu cầu dừng. Job sẽ lưu dữ liệu OCR đã xong rồi dừng an toàn.",
    }


def _iter_image_files(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _merge_worker_ocr_outputs(output_dir: Path, worker_count: int, save_debug_images: bool) -> dict[str, Any]:
    root_images = output_dir / "images"
    root_ocr_json = output_dir / "ocr_json"
    root_debug = output_dir / "debug_boxes"
    root_csv = output_dir / "nodes_to_label.csv"
    root_failed = output_dir / "failed_images.txt"
    root_empty = output_dir / "empty_ocr_images.txt"
    root_corrupted = output_dir / "corrupted_ocr_json.txt"

    for p in [root_images, root_ocr_json, root_debug]:
        p.mkdir(parents=True, exist_ok=True)
        for old in p.rglob("*"):
            if old.is_file():
                old.unlink()

    if root_csv.exists():
        root_csv.unlink()
    if root_failed.exists():
        root_failed.unlink()
    if root_empty.exists():
        root_empty.unlink()
    if root_corrupted.exists():
        root_corrupted.unlink()

    merged_rows: list[dict[str, Any]] = []
    fields: list[str] | None = None
    failed_lines: list[str] = []
    empty_lines: list[str] = []
    corrupted_lines: list[str] = []
    merged_docs = 0

    for worker_index in range(worker_count):
        worker_dir = output_dir / f"worker_{worker_index}"
        if not worker_dir.exists():
            continue

        worker_csv = worker_dir / "nodes_to_label.csv"
        if worker_csv.exists():
            with worker_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and fields is None:
                    fields = list(reader.fieldnames)
                for row in reader:
                    merged_rows.append(row)

        for src_dir, dst_dir in [
            (worker_dir / "images", root_images),
            (worker_dir / "ocr_json", root_ocr_json),
            (worker_dir / "debug_boxes", root_debug),
        ]:
            if src_dir.exists():
                for src in src_dir.rglob("*"):
                    if src.is_file():
                        if src_dir.name == "images":
                            rel = src.relative_to(src_dir)
                            target = dst_dir / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            target = dst_dir / src.name
                        shutil.copy2(src, target)
                        if src_dir.name == "ocr_json":
                            merged_docs += 1

        worker_failed = worker_dir / "failed_images.txt"
        if worker_failed.exists():
            failed_lines.extend([x for x in worker_failed.read_text(encoding="utf-8").splitlines() if x.strip()])

        worker_empty = worker_dir / "empty_ocr_images.txt"
        if worker_empty.exists():
            empty_lines.extend([x for x in worker_empty.read_text(encoding="utf-8").splitlines() if x.strip()])

        worker_corrupted = worker_dir / "corrupted_ocr_json.txt"
        if worker_corrupted.exists():
            corrupted_lines.extend([x for x in worker_corrupted.read_text(encoding="utf-8").splitlines() if x.strip()])

    if fields is None:
        fields = ["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"]
    _write_csv_rows(root_csv, fields, merged_rows)
    if failed_lines:
        root_failed.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
    if empty_lines:
        root_empty.write_text("\n".join(empty_lines) + "\n", encoding="utf-8")
    if corrupted_lines:
        root_corrupted.write_text("\n".join(corrupted_lines) + "\n", encoding="utf-8")

    if not save_debug_images and root_debug.exists():
        for p in root_debug.glob("*"):
            if p.is_file():
                p.unlink()

    return {
        "nodes_csv": str(root_csv),
        "images_dir": str(root_images),
        "ocr_json_dir": str(root_ocr_json),
        "debug_boxes_dir": str(root_debug) if save_debug_images else "",
        "merged_rows": len(merged_rows),
        "merged_docs": merged_docs,
        "failed_images_count": len(failed_lines),
        "empty_ocr_images_count": len(empty_lines),
        "corrupted_ocr_json_count": len(corrupted_lines),
    }


def _run_prepare_ocr_parallel_job(job_id: str, args: dict[str, Any]) -> None:
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == job_id:
            j["status"] = "running"
            j["started_at"] = datetime.utcnow().isoformat()
            j["stdout"] = "Khởi động OCR song song nhiều worker..."
    _save_jobs(jobs)

    input_dir = _resolve_project_path(args["input_dir"])
    output_dir = _resolve_project_path(args.get("output_dir", "data/labeling_stage_b"))
    worker_count = max(1, min(int(args.get("num_workers", 1)), 12))
    save_debug_images = int(args.get("save_debug_images", 1)) == 1
    save_every_images = max(1, int(args.get("save_every_images", 10)))
    overwrite_existing = int(args.get("overwrite_existing", 1)) == 1
    parent_stop_flag = _job_stop_flag_path(job_id)
    ocr_use_gpu_raw = str(os.getenv("OCR_USE_GPU", "")).strip().lower()
    ocr_use_gpu = ocr_use_gpu_raw in {"1", "true", "yes", "on"}

    image_files = _iter_image_files(input_dir)
    total_images = len(image_files)
    if total_images == 0:
        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["status"] = "failed"
                j["stdout"] = "Không tìm thấy ảnh đầu vào để OCR."
                j["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs(jobs)
        return

    if worker_count > total_images:
        worker_count = total_images

    forced_single_worker_reason = None
    if ocr_use_gpu and worker_count > 1:
        forced_single_worker_reason = (
            "Phát hiện OCR_USE_GPU=1. PaddleOCR GPU nhiều process trên Windows dễ trả empty boxes/0 node, "
            "nên hệ thống tự ép về 1 worker để tránh OCR rỗng."
        )
        worker_count = 1

    worker_states: dict[int, dict[str, Any]] = {
        i: {"current": 0, "total": 0, "current_file": "", "log_lines": [], "return_code": None}
        for i in range(worker_count)
    }
    procs: list[tuple[int, subprocess.Popen, Path]] = []
    threads: list[threading.Thread] = []

    def stream_worker_stdout(worker_index: int, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            line_text = line.rstrip("\n")
            state = worker_states[worker_index]
            state["log_lines"].append(line_text)
            if len(state["log_lines"]) > 120:
                state["log_lines"] = state["log_lines"][-120:]
            m = re.search(r"OCR\s+\[(\d+)/(\d+)\]:\s*(.+)$", line_text)
            if m:
                state["current"] = int(m.group(1))
                state["total"] = int(m.group(2))
                state["current_file"] = m.group(3).strip()

    try:
        for worker_index in range(worker_count):
            child_stop_flag = _job_stop_flag_path(f"{job_id}_worker_{worker_index}")
            child_args = dict(args)
            child_args["num_workers"] = worker_count
            child_args["worker_index"] = worker_index
            child_args["save_every_images"] = save_every_images
            child_args["overwrite_existing"] = 1 if overwrite_existing else 0
            child_args["stop_flag_file"] = str(child_stop_flag)
            cmd = _build_cmd("prepare_ocr_labeling", child_args)
            proc = subprocess.Popen(
                cmd,
                cwd=str(SRC_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_build_subprocess_env(),
            )
            procs.append((worker_index, proc, child_stop_flag))
            t = threading.Thread(target=stream_worker_stdout, args=(worker_index, proc), daemon=True)
            t.start()
            threads.append(t)

        while True:
            alive = False
            if parent_stop_flag.exists():
                for _, _, child_stop_flag in procs:
                    if not child_stop_flag.exists():
                        child_stop_flag.write_text(str(time.time()), encoding="utf-8")

            current = 0
            total = 0
            active_files: list[str] = []
            logs: list[str] = [f"OCR song song {worker_count} worker | tổng ảnh đầu vào: {total_images}"]
            for worker_index, proc, _ in procs:
                if proc.poll() is None:
                    alive = True
                state = worker_states[worker_index]
                current += int(state.get("current") or 0)
                total += int(state.get("total") or 0)
                if state.get("current_file"):
                    active_files.append(f"W{worker_index}: {state['current_file']}")
                worker_status = "running" if proc.poll() is None else f"done(rc={proc.poll()})"
                logs.append(
                    f"W{worker_index}: {state.get('current', 0)}/{state.get('total', 0)} | {worker_status}"
                )
                logs.extend(state.get("log_lines", [])[-2:])

            pct_base = total if total > 0 else total_images
            pct = int((current * 100) / pct_base) if pct_base > 0 else 0
            jobs = _load_jobs()
            for j in jobs:
                if j["id"] == job_id:
                    if parent_stop_flag.exists() and j.get("status") in {"queued", "running"}:
                        j["status"] = "stopping"
                    j["progress"] = {
                        "current": current,
                        "total": pct_base,
                        "percent": min(100, pct),
                        "current_file": " | ".join(active_files[:3]),
                        "worker_count": worker_count,
                    }
                    j["stdout"] = "\n".join(logs[-250:])
            _save_jobs(jobs)

            if not alive:
                break
            time.sleep(1.5)

        for _, proc, _ in procs:
            proc.wait()
        for t in threads:
            t.join(timeout=1)

        merge_info = _merge_worker_ocr_outputs(output_dir, worker_count, save_debug_images)
        any_failed = False
        for worker_index, proc, _ in procs:
            rc = proc.returncode or 0
            worker_states[worker_index]["return_code"] = rc
            if rc != 0:
                any_failed = True

        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                stopped = parent_stop_flag.exists() or bool(j.get("stop_requested"))
                if stopped and not any_failed:
                    j["status"] = "stopped"
                else:
                    j["status"] = "success" if not any_failed else "failed"
                j["progress"] = {
                    "current": merge_info["merged_docs"],
                    "total": total_images,
                    "percent": int((merge_info["merged_docs"] * 100) / total_images) if total_images else 100,
                    "current_file": "",
                    "worker_count": worker_count,
                }
                summary = [
                    f"OCR nhiều worker hoàn tất | workers={worker_count}",
                    f"merged_docs={merge_info['merged_docs']}/{total_images}",
                    f"merged_rows={merge_info['merged_rows']}",
                    f"failed_images={merge_info['failed_images_count']}",
                    f"empty_ocr_images={merge_info['empty_ocr_images_count']}",
                    f"nodes_csv={merge_info['nodes_csv']}",
                ]
                j["stdout"] = "\n".join(summary)
                j["return_code"] = 0 if not any_failed else 1
                j["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs(jobs)
    finally:
        if parent_stop_flag.exists():
            parent_stop_flag.unlink()
        for _, _, child_stop_flag in procs:
            if child_stop_flag.exists():
                child_stop_flag.unlink()


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
    with JOBS_LOCK:
        if not JOBS_FILE.exists():
            return []
        try:
            raw = JOBS_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return []
        if not raw:
            return []
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            broken = JOBS_FILE.with_suffix(".broken.json")
            try:
                shutil.copy2(JOBS_FILE, broken)
            except Exception:
                pass
            return []


def _save_jobs(jobs: list[dict[str, Any]]) -> None:
    with JOBS_LOCK:
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(JOBS_FILE.parent),
            delete=False,
            suffix=".tmp",
        ) as f:
            f.write(json.dumps(jobs, ensure_ascii=False, indent=2))
            tmp_path = Path(f.name)
        tmp_path.replace(JOBS_FILE)


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
    # Inherit the parent process environment verbatim. The in-process OCR flow
    # (single-image preview) already imports torch/VietOCR successfully using
    # this exact environment, so the subprocess must not diverge from it.
    env = os.environ.copy()

    # Keep the inherited PATH order first (conda-activated order that torch's
    # DLL loader relies on). Only APPEND conda DLL dirs that are missing, so we
    # never shadow torch's MKL/OpenMP dependencies with a different Library/bin
    # copy (the cause of `WinError 127` / `shm.dll` on Windows).
    existing_path = env.get("PATH", "")
    existing_parts = existing_path.split(os.pathsep) if existing_path else []
    existing_lower = {p.lower() for p in existing_parts}
    for candidate in (
        CONDA_ENV_DIR / "Library" / "bin",
        CONDA_ENV_DIR / "DLLs",
        CONDA_ENV_DIR / "Scripts",
        CONDA_ENV_DIR,
    ):
        if candidate.exists() and str(candidate).lower() not in existing_lower:
            existing_parts.append(str(candidate))
    env["PATH"] = os.pathsep.join(p for p in existing_parts if p)

    # Remove PYTHONHOME entirely instead of blanking it. An empty string breaks
    # Python/torch DLL initialization on Windows; absence lets the interpreter
    # resolve its own home correctly.
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(SRC_DIR)

    # Linux GPU detection: the cu118 Paddle build needs cuDNN 8 plus the cu11 CUDA
    # runtime libs on the loader path. cuDNN 8 lives in a separate dir (the env's
    # system cuDNN is v9, required by torch); the cu11 libs sit in
    # site-packages/nvidia/*/lib where only torch finds them via RPATH. Prepend both
    # so the OCR subprocess can run Paddle detection on the GPU. Gated on the cuDNN 8
    # dir existing, so machines without the GPU setup are unaffected.
    if os.name == "posix":
        import glob

        cudnn8_dir = os.getenv(
            "OCR_PADDLE_CUDNN8_DIR", os.path.expanduser("~/.local/lib/paddle_cudnn8")
        )
        if os.path.isdir(cudnn8_dir):
            gpu_lib_dirs = [cudnn8_dir]
            gpu_lib_dirs += glob.glob(
                str(CONDA_ENV_DIR.parent / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
            )
            previous = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                gpu_lib_dirs + ([previous] if previous else [])
            )
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


def _doc_id_from_image_path(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        rel = path.name
    if isinstance(rel, Path):
        rel_no_suffix = rel.with_suffix("")
        return "__".join(part.strip() for part in rel_no_suffix.parts if part.strip())
    return str(rel).replace("\\", "__").replace("/", "__").rsplit(".", 1)[0]


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
    def _candidate_doc_ids(path: Path) -> list[str]:
        values = [path.stem.lower()]
        try:
            if STAGE_B_RAW_DIR.resolve() in path.resolve().parents:
                values.append(_doc_id_from_image_path(path, STAGE_B_RAW_DIR).lower())
        except Exception:
            pass
        try:
            if path.parent.name == "images":
                values.append(_doc_id_from_image_path(path, path.parent).lower())
        except Exception:
            pass
        return values

    exact = next((p for p in image_index if did in _candidate_doc_ids(p)), None)
    if exact:
        return str(exact.relative_to(SRC_DIR))
    fuzzy = next((p for p in image_index if any(did in val for val in _candidate_doc_ids(p))), None)
    if fuzzy:
        return str(fuzzy.relative_to(SRC_DIR))
    return None


def _preview_path_from_image_relpath(base_dir: Path, image_relpath: str) -> str | None:
    rel = str(image_relpath or "").strip().replace("\\", "/")
    if not rel:
        return None
    candidates = [
        base_dir / "images" / rel,
        STAGE_B_RAW_DIR / rel,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _project_relative_path(candidate)
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
    ACTIVE_JOB_PROCS[job_id] = proc
    stop_flag_path = _job_stop_flag_path(job_id)

    log_lines: list[str] = []
    try:
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
                        if stop_flag_path.exists() and j.get("status") in {"queued", "running"}:
                            j["status"] = "stopping"
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
                if stop_flag_path.exists() and proc.poll() is None:
                    proc.terminate()
                    break

        proc.wait()

        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                stopped = stop_flag_path.exists() or bool(j.get("stop_requested"))
                if stopped and (proc.returncode or 0) == 0:
                    j["status"] = "stopped"
                else:
                    j["status"] = "success" if (proc.returncode or 0) == 0 else "failed"
                j["return_code"] = proc.returncode
                j["stdout"] = "\n".join(log_lines[-1000:])
                j["stderr"] = ""
                if j["status"] == "success" and j.get("progress", {}).get("total", 0) > 0:
                    j["progress"]["current"] = j["progress"]["total"]
                    j["progress"]["percent"] = 100
                j["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs(jobs)
    finally:
        ACTIVE_JOB_PROCS.pop(job_id, None)
        if stop_flag_path.exists():
            stop_flag_path.unlink()


def _enqueue_job(mode: str, args: dict[str, Any]) -> dict[str, Any]:
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    job_id = str(uuid.uuid4())
    args = dict(args)
    if mode == "prepare_ocr_labeling":
        args["stop_flag_file"] = str(_job_stop_flag_path(job_id))
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


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    return _request_job_stop(job_id)


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


@app.post("/api/pipeline/preprocess-gcn-dataset-now")
def preprocess_dataset_now(req: PreprocessDatasetRequest) -> dict[str, Any]:
    from pipeline.services.gcn_dataset_preprocess_service import GCNDatasetPreprocessService

    input_path = SRC_DIR / req.input_csv
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    output_path = SRC_DIR / req.output_json
    service = GCNDatasetPreprocessService()
    service.preprocess_csv(
        input_csv_path=str(input_path),
        output_json_path=str(output_path),
        doc_id_col=req.doc_id_col,
        text_col=req.text_col,
        label_col=req.label_col,
        score_col=req.score_col,
        x1_col=req.x1_col,
        y1_col=req.y1_col,
        x2_col=req.x2_col,
        y2_col=req.y2_col,
        same_line_ratio=req.same_line_ratio,
        near_threshold=req.near_threshold,
        min_nodes_per_graph=req.min_nodes_per_graph,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "input_csv": req.input_csv,
        "output_json": req.output_json,
        "meta": data.get("meta", {}),
    }


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

    updates_map = {u.row_number: u.label.strip() for u in req.updates if u.label and u.label.strip()}
    text_map = {u.row_number: u.text for u in req.updates if u.text is not None}
    if not updates_map and not text_map:
        return {"updated": 0, "text_updated": 0, "input_csv": req.input_csv}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if req.label_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing label column: {req.label_col}")
        if text_map and req.text_col not in fields:
            raise HTTPException(status_code=400, detail=f"Missing text column: {req.text_col}")
        rows = list(reader)

    updated = 0
    text_updated = 0
    for idx, row in enumerate(rows, start=2):
        if idx in updates_map:
            row[req.label_col] = updates_map[idx]
            updated += 1
        if idx in text_map:
            row[req.text_col] = text_map[idx]
            text_updated += 1

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {"updated": updated, "text_updated": text_updated, "input_csv": req.input_csv}


def _suggest_label_from_text(text: str) -> str:
    t = text.strip()
    low = t.lower()

    if not t:
        return "OTHER"
    if re.search(r"\b(khach hang|ten khach|customer|ghi chu|note)\b", low):
        return "OTHER"
    if re.search(r"\b(thoi gian|gio vao|gio ra|so ban|ban so|stt|thu tu)\b", low):
        return "OTHER"
    if re.search(
        r"^\((vua|lon|nho|size|size\s*[smlx0-9]+|hot|cold|nong|lanh|it da|it duong|mang ve|tai cho|take away)\)$",
        low,
    ):
        return "OTHER"

    if re.search(r"\b(cash|tien mat|momo|zalopay|gopay|gojek|card|visa|mastercard|atm)\b", low):
        return "PAYMENT_METHOD"
    if re.search(r"\b(thu ngan|cashier|quay|counter)\b", low):
        return "CASHIER"
    if re.search(r"\b(mst|ma so thue|tax code)\b", low):
        return "TAX_CODE"
    if re.search(r"\b(sdt|dien thoai|phone|tel|hotline)\b", low) or re.fullmatch(r"[+()0-9.\-\s]{8,20}", t):
        return "MERCHANT_PHONE"
    if (
        re.search(r"\b(dia chi|address|duong|phuong|quan|thanh pho|so nha|ap|xa|huyen)\b", low)
        or any(token in low for token in [" p.", " q.", " tp.", ", p.", ", q.", ", tp."])
    ) and any(ch.isdigit() for ch in t):
        return "MERCHANT_ADDRESS"
    if re.search(r"\b(hoa don|invoice|bill\s*no|ma hd|so hd|ref)\b", low):
        return "INVOICE_ID"
    if re.search(r"\b(ngay|date)\b", low) or re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", low):
        return "DATE"
    if re.search(r"\b(gio|time)\b", low) or re.search(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b", low):
        return "TIME"
    if re.search(r"\b(tam tinh|subtotal)\b", low):
        return "SUBTOTAL"
    if re.search(r"\b(phi dich vu|service fee)\b", low):
        return "SERVICE_FEE"
    if re.search(r"\b(giam gia|discount)\b", low):
        return "DISCOUNT"
    if re.search(r"\b(vat|thue)\b", low):
        return "TAX_AMOUNT"
    if re.search(r"\b(tong cong|thanh tien|tong thanh toan|tong tien|total)\b", low):
        return "TOTAL_AMOUNT"
    if re.search(r"^(tt|ten mon|ten hang|mat hang|sl|dvt|d\.?gia|don gia|t\.?tien)$", low):
        return "OTHER"
    if re.search(r"\b(so luong|qty|quantity|x\d+)\b", low):
        return "ITEM_QTY"
    if re.search(r"\b(don gia|unit price|price|d/|vnd)\b", low):
        return "ITEM_UNIT_PRICE"
    if re.search(r"\b(amount|tien)\b", low):
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
    # NFD decomposition drops most diacritics, but the Vietnamese "đ/Đ" is a distinct
    # letter that does NOT decompose — map it to d/D explicitly so keyword matches like
    # "don gia", "dia chi", "dvt", "dien thoai" hit text written with đ.
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
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
    summary_re = re.compile(
        r"\b(tong cong|tong tien|thanh tien|tam tinh|subtotal|vat|thue|gtgt|giam gia|"
        r"discount|phi dich vu|service fee|tien khach tra|tien mat|thoi lai|"
        r"can thanh toan|thanh toan)\b"
    )
    taxcode_re = re.compile(r"\b(ma so thue|mst|tax code)\b")
    item_header_words = ["ten hang", "mat hang", "sl", "so luong", "dvt", "don gia", "thanh tien", "qty", "amount"]

    summary_line_ids: set[int] = set()
    item_header_line_id: int | None = None
    for line_id, group in enumerate(line_groups):
        line_text = " | ".join(nodes[i]["norm_text"] for i in group if nodes[i]["norm_text"])
        # A line with >=2 column words is the item-table header (e.g. "Đơn giá | Thành
        # tiền"). It contains "thanh tien" so it looks like a summary line — classify it
        # as the header, not summary, or the whole item table gets read as summary.
        is_header_line = sum(1 for word in item_header_words if word in line_text) >= 2
        if is_header_line and item_header_line_id is None:
            item_header_line_id = line_id
        # Word boundaries so "vat" doesn't match inside "vatio" (VAT10% on an item row);
        # skip the tax-CODE line ("Mã số thuế") which would start the summary far too early.
        if not is_header_line and summary_re.search(line_text) and not taxcode_re.search(line_text):
            summary_line_ids.add(line_id)

    # Summary is below the item header; drop any earlier stray hits.
    if item_header_line_id is not None:
        summary_line_ids = {lid for lid in summary_line_ids if lid > item_header_line_id}

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
            node["is_item_header_line"] = item_header_line_id is not None and line_id == item_header_line_id
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

    # Money-column roles for the items region, from geometry + value shape (the item
    # header is often merged/garbled by OCR, so column_hint alone is unreliable). The
    # rightmost money column is the line amount; bare 1-3 digit integers are quantities;
    # everything else is a unit price. Lets the heuristic label item-row money directly
    # instead of leaning on the LLM (and stops a stray "VAT"/"giảm giá" token on the row
    # from hijacking it into TAX_AMOUNT/DISCOUNT).
    item_money_nodes = [n for n in nodes if n.get("line_region") == "items" and n.get("is_money_like")]
    if item_money_nodes:
        max_money_ncx = max(n["ncx"] for n in item_money_nodes)
        for n in item_money_nodes:
            if n["ncx"] >= max_money_ncx - 0.08:
                n["money_col_role"] = "amount"
            elif re.fullmatch(r"\d{1,3}", n["text"].strip()):
                n["money_col_role"] = "qty"
            else:
                n["money_col_role"] = "unit_price"

    doc_summary = {
        "line_count": len(line_groups),
        "header_lines": [x["text"] for x in line_meta[: min(6, len(line_meta))]],
        "item_lines": [x["text"] for x in line_meta[header_boundary : min(first_summary_line, header_boundary + 8)]],
        "summary_lines": [x["text"] for x in line_meta[first_summary_line : first_summary_line + 6]],
        "item_header_line": line_meta[item_header_line_id]["text"] if item_header_line_id is not None else "",
        "detected_regions": {
            "header_end_line": header_boundary,
            "summary_start_line": first_summary_line,
        },
        "line_region_map": [
            {"line_id": x["line_id"], "region": x["region"], "text": x["text"]}
            for x in line_meta[: min(len(line_meta), 24)]
        ],
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
    is_item_header_line = bool(node.get("is_item_header_line"))
    keyword_blob = " | ".join(filter(None, [norm, around]))

    address_tokens = ["dia chi", "address", "duong", "phuong", "quan", "thanh pho", "tp", "so nha", "ap", "xa", "huyen", "q.", "p."]
    header_row_tokens = ["tt", "ten mon", "ten hang", "mat hang", "sl", "so luong", "dvt", "don gia", "dgia", "thanh tien"]
    admin_tokens = ["khach hang", "customer", "ghi chu", "note", "thoi gian", "gio", "ngay", "so hd", "hoa don", "invoice", "mst", "thu ngan"]
    item_modifier_pattern = re.compile(
        r"^\(?\s*(vua|lon|nho|size\s*[smlx0-9]+|hot|cold|nong|lanh|it da|it duong|mang ve|tai cho|take away|da)\s*\)?$"
    )

    if not text.strip():
        return "OTHER", 1.0, "empty_text"
    if is_item_header_line:
        return "OTHER", 0.99, "item_header_line"
    if item_modifier_pattern.search(norm):
        return "OTHER", 0.96, "item_modifier_like"
    if re.fullmatch(r"[()\[\]\-_/+*.:,'\" ]+", text):
        return "OTHER", 0.99, "punctuation_only"
    if any(token in norm for token in ["khach hang", "customer", "ghi chu", "note"]):
        return "OTHER", 0.94, "customer_or_note_context"
    if sum(1 for token in admin_tokens if token in keyword_blob) >= 2 and region != "items":
        return "OTHER", 0.9, "mixed_admin_context"

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
    if region == "header":
        address_score = sum(1 for token in address_tokens if token in keyword_blob)
        if address_score >= 2 and any(ch.isdigit() for ch in text):
            return "MERCHANT_ADDRESS", 0.9, "header_address_pattern"
    if re.search(r"\b(hoa don|invoice|bill no|so hd|ma hd|ref)\b", norm):
        return "INVOICE_ID", 0.92, "invoice_keyword"
    if re.fullmatch(r"[0-9A-Za-z\-_/]{3,30}", text.strip()) and any(x in around for x in ["so hd", "hoa don", "invoice", "ma hd", "ref"]):
        return "INVOICE_ID", 0.88, "invoice_value_context"

    if node["is_money_like"]:
        money_role = node.get("money_col_role", "")
        # Items region: classify by column geometry FIRST so a stray "VAT"/"giảm giá"
        # token sharing the row can't hijack a quantity / unit price / line amount.
        if region == "items":
            # Geometry role is more reliable than column_hint: OCR often merges the whole
            # item header into one node, so column_hint matches several columns at once.
            if money_role == "qty":
                return "ITEM_QTY", 0.92, "item_qty_geometry"
            if money_role == "amount":
                return "ITEM_AMOUNT", 0.92, "item_amount_geometry"
            if money_role == "unit_price":
                return "ITEM_UNIT_PRICE", 0.92, "item_unit_price_geometry"
            if any(x in column_hint for x in ["sl", "so luong", "qty"]):
                return "ITEM_QTY", 0.9, "item_qty_column"
            if any(x in column_hint for x in ["thanh tien", "amount"]):
                return "ITEM_AMOUNT", 0.9, "item_amount_column"
            if any(x in column_hint for x in ["don gia", "dgia", "price"]):
                return "ITEM_UNIT_PRICE", 0.9, "item_unit_price_column"
        # Summary / keyword-based context. Word boundaries so "vat" doesn't match inside
        # an OCR garble like "vatio" (VAT10%).
        if re.search(r"\b(giam gia|discount)\b", around):
            return "DISCOUNT", 0.96, "discount_context"
        if re.search(r"\b(vat|thue|gtgt)\b", around):
            return "TAX_AMOUNT", 0.96, "tax_amount_context"
        if re.search(r"\b(phi dich vu|service fee)\b", around):
            return "SERVICE_FEE", 0.96, "service_fee_context"
        if re.search(r"\b(tam tinh|subtotal)\b", around):
            return "SUBTOTAL", 0.96, "subtotal_context"
        if re.search(r"\b(tong cong|tong tien|can thanh toan|thanh toan|tong thanh toan)\b", around):
            return "TOTAL_AMOUNT", 0.97, "total_context"
        if region == "summary":
            return "TOTAL_AMOUNT", 0.65, "summary_money_fallback"

    if region == "header":
        if len(norm) >= 6 and any(ch.isalpha() for ch in norm):
            if any(x in norm for x in ["vin", "mart", "co.op", "minimart", "store", "shop", "coffee", "tra sua", "quan"]):
                return "MERCHANT_NAME", 0.85, "merchant_name_like"
            if "dia chi" in around:
                return "MERCHANT_ADDRESS", 0.8, "header_address_context"
    if region == "items":
        if any(token in norm for token in header_row_tokens):
            return "OTHER", 0.95, "item_table_header"
        if any(x in column_hint for x in ["ten hang", "mat hang", "item", "hang"]):
            if any(ch.isalpha() for ch in norm):
                if any(token in norm for token in ["khach hang", "thu ngan", "thoi gian", "so hd", "hoa don"]):
                    return "OTHER", 0.92, "item_column_but_admin_text"
                return "ITEM_NAME", 0.88, "item_name_column"
        if any(ch.isalpha() for ch in norm) and not node["is_money_like"]:
            if len(norm.split()) <= 2 and item_modifier_pattern.search(norm):
                return "OTHER", 0.94, "short_item_modifier"
            if sum(1 for token in admin_tokens if token in keyword_blob) >= 1:
                return "OTHER", 0.84, "admin_text_in_items_region"
            return "ITEM_NAME", 0.62, "item_alpha_fallback"

    return None, 0.0, "needs_llm"

def _build_doc_llm_payload(nodes: list[dict[str, Any]], target_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    row_to_node = {n["row_number"]: n for n in nodes}

    def _shown_heuristic(n: dict[str, Any]) -> tuple[str | None, float, str]:
        # Only surface heuristic guidance the LLM should actually trust.
        # Weak fallbacks (e.g. item_alpha_fallback -> ITEM_NAME at 0.62) bias
        # the model toward ITEM_NAME, so hide them and let the LLM decide fresh.
        conf = float(n.get("heuristic_confidence", 0.0) or 0.0)
        if conf >= 0.75:
            return n.get("heuristic_label"), round(conf, 4), n.get("heuristic_reason", "")
        return None, round(conf, 4), ""
    line_to_nodes: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        line_id = int(node.get("line_id", -1))
        line_to_nodes.setdefault(line_id, []).append(node)
    for line_nodes in line_to_nodes.values():
        line_nodes.sort(key=lambda x: x.get("line_pos", 0))

    def _compact_node(n: dict[str, Any]) -> dict[str, Any]:
        h_label, h_conf, h_reason = _shown_heuristic(n)
        return {
            "row_number": n["row_number"],
            "text": n["text"],
            "region": n.get("line_region"),
            "column_hint": n.get("column_hint", ""),
            "bbox_norm": [n["nx1"], n["ny1"], n["nx2"], n["ny2"]],
            "heuristic_label": h_label,
            "heuristic_confidence": h_conf,
            "heuristic_reason": h_reason,
            "current_label": n.get("label", ""),
        }

    def _target_context(n: dict[str, Any]) -> dict[str, Any]:
        line_id = int(n.get("line_id", -1))
        same_line_nodes = line_to_nodes.get(line_id, [])
        prev_line_nodes = line_to_nodes.get(line_id - 1, [])
        next_line_nodes = line_to_nodes.get(line_id + 1, [])
        seeded_same_line = [
            _compact_node(x)
            for x in same_line_nodes
            if x["row_number"] != n["row_number"] and x.get("heuristic_label") and float(x.get("heuristic_confidence", 0.0)) >= 0.75
        ][:8]
        vertical_neighbors = []
        for key in ("top_text", "bottom_text"):
            txt = str(n.get(key, "")).strip()
            if not txt:
                continue
            vertical_neighbors.append({"direction": key.replace("_text", ""), "text": txt})
        h_label, h_conf, h_reason = _shown_heuristic(n)
        return {
            "target": {
                "row_number": n["row_number"],
                "text": n["text"],
                "bbox_norm": [n["nx1"], n["ny1"], n["nx2"], n["ny2"]],
                "line_id": line_id,
                "line_region": n.get("line_region"),
                "row_role_hint": n.get("row_role_hint"),
                "column_hint": n.get("column_hint", ""),
                "ocr_score": n.get("score", 1.0),
                "heuristic_label": h_label,
                "heuristic_confidence": h_conf,
                "heuristic_reason": h_reason,
                "current_label": n.get("label", ""),
                "is_money_like": n.get("is_money_like"),
                "is_date_like": n.get("is_date_like"),
                "is_time_like": n.get("is_time_like"),
                "is_phone_like": n.get("is_phone_like"),
                "is_tax_code_like": n.get("is_tax_code_like"),
            },
            "same_line_texts": n.get("same_line_texts", []),
            "same_line_nodes": [_compact_node(x) for x in same_line_nodes[:12]],
            "same_line_seeded": seeded_same_line,
            "prev_line_texts": [x["text"] for x in prev_line_nodes[:12]],
            "next_line_texts": [x["text"] for x in next_line_nodes[:12]],
            "left_text": n.get("left_text", ""),
            "right_text": n.get("right_text", ""),
            "vertical_neighbors": vertical_neighbors,
        }

    compact_nodes = [
        {
            "row_number": n["row_number"],
            "text": n["text"],
            "line_id": n.get("line_id"),
            "region": n.get("line_region"),
            "column_hint": n.get("column_hint", ""),
            "bbox_norm": [n["nx1"], n["ny1"], n["nx2"], n["ny2"]],
            "current_label": n.get("label", ""),
            "heuristic_label": _shown_heuristic(n)[0],
            "heuristic_confidence": _shown_heuristic(n)[1],
            "heuristic_reason": _shown_heuristic(n)[2],
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
            "heuristic_label": _shown_heuristic(n)[0],
            "heuristic_confidence": _shown_heuristic(n)[1],
            "heuristic_reason": _shown_heuristic(n)[2],
        }
        for n in target_nodes
    ]
    seeded_nodes = [
        {
            "row_number": n["row_number"],
            "text": n["text"],
            "heuristic_label": _shown_heuristic(n)[0],
            "heuristic_confidence": _shown_heuristic(n)[1],
            "region": n.get("line_region"),
            "column_hint": n.get("column_hint", ""),
        }
        for n in nodes
        if _shown_heuristic(n)[0]
    ]
    focus_contexts = [_target_context(n) for n in target_nodes]
    seeded_by_label: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        label = str(_shown_heuristic(n)[0] or "").strip()
        if not label:
            continue
        seeded_by_label.setdefault(label, [])
        if len(seeded_by_label[label]) < 8:
            seeded_by_label[label].append(_compact_node(n))
    return {
        "document_nodes": compact_nodes,
        "seeded_nodes": seeded_nodes[:80],
        "target_nodes": detailed_targets,
        "target_contexts": focus_contexts,
        "seeded_examples_by_label": seeded_by_label,
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

    # Canonical numeric ids (same numbering as training, see schema.LABEL_MAP).
    # The model answers with the integer id instead of the long label name to
    # cut output tokens; we map ids back to label strings after the call.
    label_ids = {str(lid): name for lid, name in LABEL_MAP.items() if name in labels}

    prompt = {
        "task": "Classify OCR nodes from one Vietnamese invoice/receipt using document structure, neighboring texts, line grouping, and node position.",
        "label_ids": label_ids,
        "rules": [
            "Return exactly one label id (integer) per target node, taken from label_ids.",
            "Use node position, same-line texts, left/right/top/bottom neighbors, and document region.",
            "You will also receive heuristic labels already assigned to many nodes in the same document.",
            "Treat high-confidence heuristic labels on nearby nodes as strong context for classifying target nodes.",
            "If a target node itself has a heuristic_label with high confidence, keep it unless surrounding document structure strongly contradicts it.",
            "Do not classify nodes as if they were isolated strings. Always reason from the whole line, nearby lines, region, and column role.",
            "When same-line seeded nodes already indicate an item row structure, infer the remaining target node by column role and neighboring seeded nodes.",
            "If one OCR node contains multiple header/admin fields merged together, such as time + invoice id + extra metadata in one text block, prefer OTHER unless one label is overwhelmingly dominant.",
            "A numeric text alone is ambiguous. Do not classify money values from text alone.",
            "If a money value is in item rows near quantity/unit price/amount columns, prefer ITEM_QTY / ITEM_UNIT_PRICE / ITEM_AMOUNT.",
            "If a money value is in summary area near Tong cong / Tam tinh / VAT / Giam gia / Thanh toan, prefer the corresponding summary label.",
            "Merchant info usually appears near the top; payment method and final totals usually appear near the bottom.",
            "Short product modifiers such as (Vừa), (Lớn), nóng, lạnh, ít đá, ít đường are usually OTHER, not ITEM_NAME.",
            "Customer lines, notes, and mixed administrative lines containing time + invoice id + other header metadata should usually be OTHER unless one field is clearly isolated.",
            "Address labels should be used only for store address-like text, usually in header region with street/district/city patterns or explicit address keywords.",
            "Do not label table headers such as Ten mon, SL, D.Gia, T.Tien as ITEM_NAME; these are structure/header nodes and should be OTHER.",
            "If a node is likely just a product variant, size, topping, or modifier and not the core product name, prefer OTHER.",
            "If heuristic and structure disagree, trust structure over lexical similarity.",
            "If still uncertain, use OTHER.",
            "Do not invent new labels.",
            "If uncertain, use OTHER.",
        ],
        "decision_policy": {
            "priority_order": [
                "document_region",
                "same_line_structure",
                "neighbor_seed_labels",
                "column_hint",
                "target_heuristic",
                "raw_text_pattern",
            ],
            "overwrite_heuristic_when": [
                "heuristic says ITEM_NAME but node is table header/customer/admin/mixed metadata",
                "heuristic says ITEM_NAME but node is short size/modifier text",
                "heuristic says OTHER but line/column structure clearly implies merchant address / item qty / item unit price / item amount / total amount",
            ],
        },
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
                "node_text": "19403 Phan Huy Ich, P.14, Q. Go Vap",
                "same_line_texts": ["THE COFFEE HOUSE", "19403 Phan Huy Ich, P.14, Q. Go Vap"],
                "region": "header",
                "expected_label": "MERCHANT_ADDRESS",
            },
            {
                "node_text": "(Vua)",
                "same_line_texts": ["Ca phe sua da", "(Vua)", "1", "35000", "35000"],
                "region": "items",
                "expected_label": "OTHER",
            },
            {
                "node_text": "Khach hang: Phanh",
                "same_line_texts": ["Khach hang", "Phanh"],
                "region": "header",
                "expected_label": "OTHER",
            },
            {
                "node_text": "1980 Thoi gian:30.09.2020 08.29 So HD:19809053062020",
                "same_line_texts": ["Thoi gian", "30.09.2020 08.29", "So HD", "19809053062020"],
                "region": "header",
                "expected_label": "OTHER",
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
                    "Return JSON object only with this schema, where label is the integer "
                    "id from label_ids (NOT the name): "
                    "{\"labels\": [{\"row_number\": 12, \"label\": 17}, ...]}"
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
    max_attempts = 5
    payload = None
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urllib_request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib_error.HTTPError as exc:
            detail = ""
            err_code = ""
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
                err_obj = err_body.get("error", {})
                detail = str(err_obj.get("message") or "")
                err_code = str(err_obj.get("code") or "")
            except Exception:
                detail = ""
            # Hard quota errors are not worth retrying.
            if exc.code == 429 and err_code == "insufficient_quota":
                raise RuntimeError(f"OpenAI HTTP 429 insufficient_quota: {detail or exc.reason}") from exc
            # Retry on rate limit (429) and transient server errors (5xx).
            if exc.code == 429 or 500 <= exc.code < 600:
                last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {detail or exc.reason}")
                if attempt < max_attempts - 1:
                    retry_after = 0.0
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "") or 0)
                    except Exception:
                        retry_after = 0.0
                    backoff = retry_after if retry_after > 0 else (2.0 ** attempt)
                    time.sleep(min(30.0, backoff) + random.uniform(0, 0.5))
                    continue
                raise last_error from exc
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib_error.URLError, TimeoutError) as exc:
            last_error = RuntimeError(f"OpenAI request failed: {exc}")
            if attempt < max_attempts - 1:
                time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                continue
            raise last_error from exc
    if payload is None:
        raise last_error or RuntimeError("OpenAI request failed without response")
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
        raw = str(item.get("label", "")).strip()
        # Preferred path: model returns the numeric id -> map back to the label
        # string here so the CSV stays human-readable. Fall back to name parsing
        # for backward compatibility if the model still answers with a name.
        if raw.lstrip("-").isdigit():
            label = LABEL_MAP.get(int(raw), "OTHER")
        else:
            label = raw.upper()
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
    llm_unlabeled = 0  # nodes left blank because the LLM failed/omitted them
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
                    # Only accept labels the LLM actually returned. If it omitted a
                    # node, leave it blank for a later pass instead of guessing with
                    # the rule fallback (which biases everything to ITEM_NAME).
                    if node["row_number"] in llm_map:
                        assigned[node["row_number"]] = llm_map[node["row_number"]]
                    else:
                        llm_unlabeled += 1
            except Exception as exc:
                # LLM failed (rate limit / quota / network). Do NOT write rule
                # fallback labels — that is exactly what corrupted the dataset
                # before. Leave these rows blank so they can be retried.
                llm_errors.append(str(exc))
                strategy_used = "llm_error_skipped"
                llm_unlabeled += len(chunk)
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
        "llm_unlabeled": llm_unlabeled,
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
    # only_empty=1 (default): keep existing labels, only fill blanks (incremental rerun).
    # only_empty=0: force re-label every node, overwriting existing labels.
    only_empty = bool(int(args.get("only_empty", 1)))
    llm_model = args.get("llm_model", "gpt-4.1-mini")
    batch_docs = max(1, min(int(args.get("batch_docs", 10)), 50))
    llm_text_batch_size = max(1, min(int(args.get("llm_text_batch_size", 10)), 50))
    require_llm = int(args.get("require_llm", 1)) == 1
    stop_flag = _job_stop_flag_path(job_id)

    try:
        if require_llm and not os.getenv("OPENAI_API_KEY", "").strip():
            # Fail loudly instead of silently degrading to the rule fallback,
            # which blindly labels most alphabetic text as ITEM_NAME.
            raise RuntimeError(
                "Missing OPENAI_API_KEY. LLM labeling is enabled (require_llm=1) but no API key is configured. "
                "Add OPENAI_API_KEY to DATN/.env and restart the backend, or disable LLM (require_llm=0)."
            )

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
                    f"llm_targets={stats['llm_targets']} llm_batches={stats['llm_batches']} "
                    f"unlabeled={stats.get('llm_unlabeled', 0)} mode={stats['strategy_used']}"
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
                    if stop_flag.exists() and j.get("status") in {"queued", "running"}:
                        j["status"] = "stopping"
                    j["progress"] = {
                        "current_docs": done_docs,
                        "total_docs": total_docs,
                        "percent": pct,
                        "current_batch_docs": len(batch_doc_ids),
                    }
                    j["stdout"] = "\n".join(logs[-120:])
            _save_jobs(jobs)

            if stop_flag.exists():
                logs.append("Da nhan yeu cau dung. Job se dung an toan sau batch vua luu.")
                break

        _write_csv_rows(csv_path, fields, rows)

        jobs = _load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                stopped = stop_flag.exists() or bool(j.get("stop_requested"))
                j["status"] = "stopped" if stopped else "success"
                j["return_code"] = 0
                j["stdout"] = "\n".join(logs[-200:] + (["Da dung AI goi y nhan an toan."] if stopped else ["Hoan tat AI goi y nhan."]))
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
    finally:
        if stop_flag.exists():
            stop_flag.unlink()


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
        filtered_total = 0
        empty_count = 0
        limit = max(1, min(req.limit, 500))
        page = max(1, req.page)
        only_empty = bool(req.only_empty)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            cur_label = str(row.get(req.label_col, "") or "").strip()
            if not cur_label:
                empty_count += 1
            if only_empty and cur_label:
                continue
            filtered_total += 1
            row_pos = filtered_total - 1
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
        "filtered_total_rows": filtered_total,
        "page": page,
        "page_size": limit,
        "total_pages": max(1, (filtered_total + limit - 1) // limit),
        "empty_label_count": empty_count,
        "only_empty": only_empty,
        "rows": rows,
        "allowed_labels": INVOICE_LABELS,
    }


@app.post("/api/pipeline/labeling-by-doc")
def labeling_by_doc(req: LabelingByDocRequest) -> dict[str, Any]:
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    def _new_doc_slot(doc_id: str) -> dict[str, Any]:
        return {"doc_id": doc_id, "total_nodes": 0, "empty_labels": 0, "labels": {}, "samples": [], "image_relpath": ""}

    docs: list[dict[str, Any]] = []
    current_doc_id: str | None = None
    current_slot: dict[str, Any] | None = None
    total_docs = 0
    has_next = False
    stop_reading = False
    page_size = max(1, min(req.page_size, 200))
    page = max(1, req.page)
    start = (page - 1) * page_size
    end = start + page_size

    def _flush_current() -> None:
        nonlocal current_doc_id, current_slot, total_docs, docs, has_next, stop_reading
        if current_slot is None:
            return
        idx = total_docs
        total_docs += 1
        if start <= idx < end:
            docs.append(current_slot)
        elif idx >= end:
            has_next = True
            stop_reading = True
        current_doc_id = None
        current_slot = None

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
            image_relpath = str(row.get("image_relpath", "") or "").strip()
            if current_doc_id != doc_id:
                _flush_current()
                if stop_reading:
                    break
                current_doc_id = doc_id
                current_slot = _new_doc_slot(doc_id)
            slot = current_slot
            if slot is None:
                continue
            slot["total_nodes"] += 1
            if label == "UNLABELED":
                slot["empty_labels"] += 1
            slot["labels"][label] = slot["labels"].get(label, 0) + 1
            if not slot["image_relpath"] and image_relpath:
                slot["image_relpath"] = image_relpath
            if len(slot["samples"]) < 8:
                slot["samples"].append({"text": text, "label": label})
    _flush_current()

    for d in docs:
        d["preview_path"] = _preview_path_from_image_relpath(csv_path.parent, str(d.get("image_relpath", "")))
        d.pop("image_relpath", None)

    return {
        "input_csv": req.input_csv,
        "page": page,
        "page_size": page_size,
        "has_next": has_next,
        "returned_docs": len(docs),
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

    preview_path = _preview_path_from_image_relpath(
        csv_path.parent,
        str(doc_rows[0].get("image_relpath", "") if doc_rows else ""),
    )
    if not preview_path:
        image_index = _stage_b_image_index(csv_path.parent)
        preview_path = _find_image_for_doc(req.doc_id, image_index)
    feature_names = [
        "text_len", "has_digit", "has_money_token", "cx_norm", "cy_norm", "w_norm", "h_norm", "ocr_score",
        "looks_money", "has_total_kw", "has_item_header_kw", "has_date", "has_time", "has_phone",
        "has_tax_kw", "has_invoice_kw", "has_payment_kw", "is_top", "is_middle", "is_bottom",
        "is_left", "is_center", "is_right", "node_order", "has_alpha", "has_vnd",
        "has_subtotal_kw", "has_discount_kw", "has_service_kw", "has_cashier_kw", "has_address_kw",
        "has_receipt_title_kw", "has_footer_kw", "has_unit_word", "qty_candidate", "long_numeric",
        "digit_count_norm", "numeric_value_norm", "x_band_0", "x_band_1", "x_band_2", "x_band_3", "x_band_4",
    ]
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


@app.post("/api/pipeline/labeling-suggest-doc")
def labeling_suggest_doc(req: LabelingSuggestDocRequest) -> dict[str, Any]:
    """AI-suggest labels for the nodes of a single image (doc_id).

    Returns suggestions only (does not write). The frontend fills them into the
    review view and persists via /api/pipeline/labeling-apply so the user can
    still adjust before saving.
    """
    if req.require_llm:
        _ensure_openai_api_key()

    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (req.label_col, req.text_col, req.doc_id_col):
            if col not in fields:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")
        rows = list(reader)

    target_doc = req.doc_id.strip()
    doc_row_indices = [
        i for i, row in enumerate(rows)
        if str(row.get(req.doc_id_col, "") or "").strip() == target_doc
    ]
    if not doc_row_indices:
        raise HTTPException(status_code=404, detail=f"No rows for doc_id: {req.doc_id}")

    if req.only_empty:
        wanted = {
            i + 2 for i in doc_row_indices
            if not str(rows[i].get(req.label_col, "") or "").strip()
        }
    else:
        wanted = {i + 2 for i in doc_row_indices}

    if not wanted:
        return {
            "doc_id": target_doc,
            "input_csv": req.input_csv,
            "suggestions": [],
            "node_count": len(doc_row_indices),
            "only_empty": bool(req.only_empty),
            "stats": {"strategy_used": "none", "note": "no_target_rows"},
        }

    # Use the full document as context, then keep only the requested rows.
    assigned, stats = _suggest_labels_for_document(
        rows,
        doc_row_indices,
        text_col=req.text_col,
        label_col=req.label_col,
        llm_model=req.llm_model,
        use_llm=bool(req.require_llm),
        max_targets_per_request=max(1, min(int(req.batch_size), 50)),
    )

    text_by_row = {i + 2: str(rows[i].get(req.text_col, "") or "") for i in doc_row_indices}
    suggestions = [
        {"row_number": rn, "label": label, "text": text_by_row.get(rn, "")}
        for rn, label in sorted(assigned.items())
        if rn in wanted
    ]

    return {
        "doc_id": target_doc,
        "input_csv": req.input_csv,
        "suggestions": suggestions,
        "node_count": len(doc_row_indices),
        "suggested_count": len(suggestions),
        "only_empty": bool(req.only_empty),
        "llm_model": req.llm_model if "llm" in stats.get("strategy_used", "") else None,
        "labels": INVOICE_LABELS,
        "stats": {
            "strategy_used": stats.get("strategy_used"),
            "heuristic_hits": stats.get("heuristic_hits"),
            "llm_targets": stats.get("llm_targets"),
            "llm_batches": stats.get("llm_batches"),
            "llm_unlabeled": stats.get("llm_unlabeled", 0),
            "llm_errors": stats.get("llm_errors", []),
        },
    }


@app.post("/api/pipeline/export-train-subset")
def export_train_subset(req: ExportTrainSubsetRequest) -> dict[str, Any]:
    """Copy the first N fully-labeled images into a separate training folder.

    A doc/image counts toward the cap only when EVERY one of its OCR nodes has a
    non-empty label. Docs are scanned in CSV order; once `limit` fully-labeled
    images are collected the export stops. The selected rows are written to
    <output_dir>/nodes_to_label.csv and (optionally) their images + ocr_json are
    copied alongside, so training reads from this clean subset only.
    """
    csv_path = SRC_DIR / req.input_csv
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.input_csv}")
    if req.limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    src_root = csv_path.parent
    src_images = src_root / "images"
    src_ocr_json = src_root / "ocr_json"

    out_root = SRC_DIR / req.output_dir
    out_images = out_root / "images"
    out_ocr_json = out_root / "ocr_json"

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (req.label_col, req.doc_id_col):
            if col not in fields:
                raise HTTPException(status_code=400, detail=f"Missing column: {col}")
        all_rows = list(reader)

    # Group rows by doc_id, preserving first-seen order.
    doc_order: list[str] = []
    doc_rows: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        did = str(row.get(req.doc_id_col, "") or "").strip() or "UNKNOWN_DOC"
        if did not in doc_rows:
            doc_rows[did] = []
            doc_order.append(did)
        doc_rows[did].append(row)

    total_docs = len(doc_order)
    selected_docs: list[str] = []
    for did in doc_order:
        rows = doc_rows[did]
        if all(str(r.get(req.label_col, "") or "").strip() for r in rows):
            selected_docs.append(did)
            if len(selected_docs) >= req.limit:
                break

    reached_limit = len(selected_docs) >= req.limit

    out_root.mkdir(parents=True, exist_ok=True)
    selected_rows = [r for did in selected_docs for r in doc_rows[did]]

    # Write the subset CSV.
    out_csv = out_root / "nodes_to_label.csv"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=str(out_root), delete=False, suffix=".tmp"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)
        tmp_path = Path(f.name)
    tmp_path.replace(out_csv)

    copied_images = 0
    missing_images: list[str] = []
    copied_json = 0
    missing_json: list[str] = []
    for did in selected_docs:
        if int(req.copy_images):
            relpath = str(doc_rows[did][0].get("image_relpath", "") or "").strip()
            if relpath:
                src_img = src_images / relpath
                if src_img.exists():
                    dst_img = out_images / relpath
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img)
                    copied_images += 1
                else:
                    missing_images.append(relpath)
        if int(req.copy_ocr_json):
            src_j = src_ocr_json / f"{did}.json"
            if src_j.exists():
                out_ocr_json.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_j, out_ocr_json / f"{did}.json")
                copied_json += 1
            else:
                missing_json.append(did)

    return {
        "input_csv": req.input_csv,
        "output_dir": str(out_root.relative_to(SRC_DIR)) if out_root.is_relative_to(SRC_DIR) else str(out_root),
        "subset_csv": str(out_csv.relative_to(SRC_DIR)) if out_csv.is_relative_to(SRC_DIR) else str(out_csv),
        "total_docs": total_docs,
        "fully_labeled_available": "unknown_capped" if reached_limit else len(selected_docs),
        "exported_docs": len(selected_docs),
        "exported_rows": len(selected_rows),
        "limit": req.limit,
        "reached_limit": reached_limit,
        "copied_images": copied_images,
        "missing_images_count": len(missing_images),
        "copied_ocr_json": copied_json,
        "missing_ocr_json_count": len(missing_json),
    }


@app.post("/api/pipeline/single-image-preview")
def single_image_preview(req: SingleImagePreviewRequest) -> dict[str, Any]:
    from pipeline.core.ocr_engine import get_last_ocr_runtime_info
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
    ocr_runtime = get_last_ocr_runtime_info()

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

    feature_names = [
        "text_len", "has_digit", "has_money_token", "cx_norm", "cy_norm", "w_norm", "h_norm", "ocr_score",
        "looks_money", "has_total_kw", "has_item_header_kw", "has_date", "has_time", "has_phone",
        "has_tax_kw", "has_invoice_kw", "has_payment_kw", "is_top", "is_middle", "is_bottom",
        "is_left", "is_center", "is_right", "node_order", "has_alpha", "has_vnd",
        "has_subtotal_kw", "has_discount_kw", "has_service_kw", "has_cashier_kw", "has_address_kw",
        "has_receipt_title_kw", "has_footer_kw", "has_unit_word", "qty_candidate", "long_numeric",
        "digit_count_norm", "numeric_value_norm", "x_band_0", "x_band_1", "x_band_2", "x_band_3", "x_band_4",
    ]
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
        "original_image_path": _project_relative_path(image_path),
        "preview_path": preview_image_rel,
        "normalized_image_path": preview_image_rel,
        "ocr_boxes_image": debug_image_rel,
        "ocr_config": ocr_overrides,
        "ocr_runtime": ocr_runtime,
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


@app.get("/api/pipeline/labeling-gallery")
def labeling_gallery(
    dir: str = "data/labeling_top1000_ppocrv6",
    page: int = 1,
    page_size: int = 60,
    sort: str = "score_desc",
    label_csv: str = "",
    label_col: str = "label",
    doc_id_col: str = "doc_id",
) -> dict[str, Any]:
    """List images in an OCR labeling folder so the UI can show a quality gallery.

    Reads each image's OCR JSON to compute num_nodes + mean recognition score,
    sorts (sort = score_desc|score_asc|nodes_desc|nodes_asc) and paginates.
    Returns project-relative paths usable by /api/files/image (processed image +
    OCR debug-box overlay when available).

    When ``label_csv`` is given, cross-references that CSV to report, per image,
    how many rows already carry a non-empty label (``num_labeled`` / ``num_rows``)
    so the UI can highlight images that have been labeled already.
    """
    base = _resolve_project_path(dir)
    ocr_json_dir = base / "ocr_json"
    if not ocr_json_dir.exists():
        raise HTTPException(status_code=404, detail=f"No ocr_json folder in {dir}")

    # doc_id -> [total rows, labeled rows] from the labeling CSV (if provided).
    label_counts: dict[str, list[int]] = {}
    if label_csv:
        csv_path = SRC_DIR / label_csv
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if doc_id_col in fields and label_col in fields:
                    for row in reader:
                        did = str(row.get(doc_id_col, "")).strip()
                        if not did:
                            continue
                        entry = label_counts.setdefault(did, [0, 0])
                        entry[0] += 1
                        if str(row.get(label_col, "")).strip():
                            entry[1] += 1

    metas: list[dict[str, Any]] = []
    for jp in ocr_json_dir.glob("*.json"):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        nodes = payload.get("nodes", []) or []
        image_relpath = str(payload.get("image_relpath", "")).strip()
        if image_relpath and not (base / "images" / image_relpath).exists():
            continue
        scores = [float(n.get("score", 0.0)) for n in nodes]
        metas.append(
            {
                "doc_id": str(payload.get("doc_id", jp.stem)),
                "image_relpath": image_relpath,
                "num_nodes": len(nodes),
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }
        )

    sort_key = "num_nodes" if "node" in sort else "mean_score"
    metas.sort(key=lambda m: m[sort_key], reverse=not sort.endswith("asc"))

    total = len(metas)
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    start = (page - 1) * page_size
    chunk = metas[start : start + page_size]

    def _rel(p: Path) -> str:
        for root in (SRC_DIR, ROOT):
            try:
                return str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                continue
        return str(p).replace("\\", "/")

    base_rel = _rel(base)
    stage_b_debug = SRC_DIR / "data" / "labeling_stage_b" / "debug_boxes"
    items: list[dict[str, Any]] = []
    for m in chunk:
        relpath = m["image_relpath"]
        image_path = f"{base_rel}/images/{relpath}" if relpath else ""
        debug_path = ""
        for cand in (
            base / "debug_boxes" / f"{m['doc_id']}_boxes.jpg",
            stage_b_debug / f"{m['doc_id']}_boxes.jpg",
        ):
            if cand.exists():
                debug_path = _rel(cand)
                break
        counts = label_counts.get(m["doc_id"])
        num_rows = counts[0] if counts else 0
        num_labeled = counts[1] if counts else 0
        items.append({
            **m,
            "image_path": image_path,
            "debug_path": debug_path,
            "num_rows": num_rows,
            "num_labeled": num_labeled,
        })

    return {
        "dir": dir,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
        "sort": sort,
        "items": items,
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
    data_dropdown_mode = Path(dir).as_posix().strip("/\\") == "data"
    allowed_data_suffixes = {".csv", ".json"}
    for p in target.rglob("*"):
        if p.is_file():
            if data_dropdown_mode and p.suffix.lower() not in allowed_data_suffixes:
                continue
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
def file_image(path: str, max_side: int | None = None) -> Response:
    target = _resolve_project_path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if not max_side or max_side <= 0:
        return FileResponse(str(target))
    safe_side = max(48, min(int(max_side), 2048))
    try:
        with Image.open(target) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            im.thumbnail((safe_side, safe_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82, optimize=True)
            return Response(
                content=buf.getvalue(),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )
    except Exception:
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
