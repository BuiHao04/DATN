from __future__ import annotations

import os
import inspect
import re
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Tuple

# The cu118 GPU Paddle build needs cuDNN 8, but this env ships cuDNN 9 (required by
# torch/VietOCR). cuDNN 8 is kept in a separate dir and made loadable by putting it on
# LD_LIBRARY_PATH at process start (see _build_subprocess_env in the backend); the .so.8
# vs .so.9 sonames let both coexist in one process. Paddle resolves cuDNN through the
# system loader, so this dir's presence on LD_LIBRARY_PATH is what gates GPU detection.
_CUDNN8_DIR = os.getenv("OCR_PADDLE_CUDNN8_DIR", os.path.expanduser("~/.local/lib/paddle_cudnn8"))

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
from PIL import Image

from pipeline.core.schema import OCRNode

_LAST_OCR_RUNTIME_INFO: dict[str, object] = {}


def _set_last_ocr_runtime_info(**kwargs) -> None:
    global _LAST_OCR_RUNTIME_INFO
    _LAST_OCR_RUNTIME_INFO = dict(kwargs)


def get_last_ocr_runtime_info() -> dict[str, object]:
    return dict(_LAST_OCR_RUNTIME_INFO)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_use_gpu() -> bool:
    env_force = os.getenv("OCR_USE_GPU", "").strip().lower()
    if env_force in {"0", "false", "no", "off"}:
        return False
    # Use the GPU only when the Paddle build is CUDA-capable AND its cuDNN 8 dir is on
    # LD_LIBRARY_PATH at process start — Paddle resolves cuDNN via the system loader, so
    # without it the first GPU op aborts the whole process (uncatchable SIGABRT). Staying
    # on CPU here keeps the in-process single-image preview safe; the batch OCR subprocess
    # gets the dir injected by _build_subprocess_env and runs detection on the GPU.
    try:
        cuda_ok = bool(paddle.is_compiled_with_cuda()) and paddle.device.cuda.device_count() > 0
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return False
    ld_dirs = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if os.path.isdir(_CUDNN8_DIR) and _CUDNN8_DIR in ld_dirs:
        return True
    if env_force in {"1", "true", "yes", "on"}:
        print("[OCR] GPU requested but cuDNN 8 dir not on LD_LIBRARY_PATH; using CPU detect.")
    return False


def _resolve_vietocr_gpu() -> bool:
    """VietOCR (torch) GPU is decided independently of Paddle.

    Paddle here is often a CPU-only build used just for detection, while the
    slow recognition step (VietOCR/torch) benefits most from the GPU. Default
    is auto: use CUDA whenever torch can see it.
    """
    env_force = os.getenv("OCR_VIETOCR_GPU", "").strip().lower()
    if env_force in {"1", "true", "yes", "on"}:
        return True
    if env_force in {"0", "false", "no", "off"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_ocr_engine(engine: str | None = None) -> str:
    value = str(engine or os.getenv("OCR_ENGINE", "paddle")).strip().lower()
    if value in {"vietocr", "paddle_vietocr", "paddle+vietocr"}:
        return "vietocr"
    return "paddle"


def _ocr_config_signature() -> tuple:
    return (
        os.getenv("OCR_TEXT_DETECTION_MODEL", _default_ppocrv6_model("det")),
        os.getenv("OCR_TEXT_RECOGNITION_MODEL", _default_ppocrv6_model("rec")),
        _env_bool("OCR_USE_DOCLINE_ORIENTATION", True),
        _env_bool("OCR_USE_DOC_UNWARPING", False),
        _env_int("OCR_DET_LIMIT_SIDE_LEN", 1536),
        os.getenv("OCR_DET_LIMIT_TYPE", "max"),
        _env_float("OCR_DET_DB_THRESH", 0.25),
        _env_float("OCR_DET_DB_BOX_THRESH", 0.58),
        _env_float("OCR_DET_DB_UNCLIP_RATIO", 1.25),
        _env_bool("OCR_USE_DILATION", False),
        _env_float("OCR_DROP_SCORE", 0.45),
        _env_float("OCR_REC_SCORE_THRESH", 0.25),
        _env_int("OCR_MAX_BATCH_SIZE", 10),
        _env_bool("OCR_SHOW_LOG", True),
    )


_OCR_OVERRIDE_ENV_MAP = {
    "det_limit_side_len": "OCR_DET_LIMIT_SIDE_LEN",
    "det_limit_type": "OCR_DET_LIMIT_TYPE",
    "det_db_thresh": "OCR_DET_DB_THRESH",
    "det_db_box_thresh": "OCR_DET_DB_BOX_THRESH",
    "det_db_unclip_ratio": "OCR_DET_DB_UNCLIP_RATIO",
    "use_dilation": "OCR_USE_DILATION",
    "drop_score": "OCR_DROP_SCORE",
    "rec_score_thresh": "OCR_REC_SCORE_THRESH",
    "max_batch_size": "OCR_MAX_BATCH_SIZE",
    "show_log": "OCR_SHOW_LOG",
    "upscale_min_side": "OCR_UPSCALE_MIN_SIDE",
    "upscale_factor": "OCR_UPSCALE_FACTOR",
    "text_detection_model_name": "OCR_TEXT_DETECTION_MODEL",
    "text_recognition_model_name": "OCR_TEXT_RECOGNITION_MODEL",
    "use_textline_orientation": "OCR_USE_DOCLINE_ORIENTATION",
    "use_doc_unwarping": "OCR_USE_DOC_UNWARPING",
}


@contextmanager
def temporary_ocr_config(overrides: dict | None = None):
    if not overrides:
        yield
        return

    env_map = _OCR_OVERRIDE_ENV_MAP
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            env_name = env_map.get(str(key))
            if not env_name or value is None or value == "":
                continue
            previous[env_name] = os.environ.get(env_name)
            os.environ[env_name] = str(value)
        _get_paddle_ocr.cache_clear()
        yield
    finally:
        for env_name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old_value
        if previous:
            _get_paddle_ocr.cache_clear()


def _prepare_image_for_ocr(image_path: str) -> Tuple[np.ndarray | str, float, float]:
    if not _env_bool("OCR_PREPROCESS", True):
        return image_path, 1.0, 1.0

    image = cv2.imread(image_path)
    if image is None:
        return image_path, 1.0, 1.0

    if _env_bool("OCR_DOCUMENT_NORMALIZE", True):
        image = _normalize_document_image(image)

    if _env_bool("OCR_MASK_RECEIPT_REGION", True):
        image = _apply_receipt_region_mask(image)

    orig_h, orig_w = image.shape[:2]
    scale = 1.0
    if max(orig_h, orig_w) < _env_int("OCR_UPSCALE_MIN_SIDE", 1400):
        scale = min(_env_float("OCR_UPSCALE_FACTOR", 1.0), 2.0)
        if scale > 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scale = 1.0

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=_env_float("OCR_CLAHE_CLIP", 2.0),
        tileGridSize=(8, 8),
    )
    l_channel = clahe.apply(l_channel)
    image = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if _env_bool("OCR_DENOISE", True):
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            _env_int("OCR_DENOISE_H", 5),
            _env_int("OCR_DENOISE_H_COLOR", 5),
            7,
            21,
        )

    if _env_bool("OCR_SHARPEN", True):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        image = cv2.filter2D(image, -1, kernel)

    return image, scale, scale


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def _four_point_warp(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_quad_points(points)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(int(round(max(width_a, width_b))), 1)
    height = max(int(round(max(height_a, height_b))), 1)
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def _find_document_quad(image: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = preview.shape[0] * preview.shape[1] * _env_float("OCR_DOCUMENT_MIN_AREA_RATIO", 0.18)
    best_quad = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area
    if best_quad is None:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= min_area:
            rect = cv2.minAreaRect(contour)
            best_quad = cv2.boxPoints(rect).astype(np.float32)
    if best_quad is None:
        return None
    if scale != 1.0:
        best_quad = best_quad / scale
    return best_quad


def _extract_receipt_mask(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    light_mask = cv2.inRange(l_channel, _env_int("OCR_MASK_L_MIN", 155), 255)
    a_center_mask = cv2.inRange(a_channel, 110, 145)
    b_center_mask = cv2.inRange(b_channel, 110, 150)
    mask = cv2.bitwise_and(light_mask, cv2.bitwise_and(a_center_mask, b_center_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    min_area = image.shape[0] * image.shape[1] * _env_float("OCR_MASK_MIN_AREA_RATIO", 0.18)
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return mask
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


def _apply_receipt_region_mask(image: np.ndarray) -> np.ndarray:
    mask = _extract_receipt_mask(image)
    if mask is None or not np.any(mask):
        return image
    out = image.copy()
    bg_color = tuple(int(_env_int("OCR_MASK_BG_VALUE", 255)) for _ in range(3))
    out[mask == 0] = bg_color
    return out


def _deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 20:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 18:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _normalize_document_image(image: np.ndarray) -> np.ndarray:
    normalized = image
    if _env_bool("OCR_DOCUMENT_CROP", True):
        quad = _find_document_quad(normalized)
        if quad is not None:
            try:
                normalized = _four_point_warp(normalized, quad)
            except Exception:
                normalized = image
    if _env_bool("OCR_DESKEW", True):
        normalized = _deskew_image(normalized)
    return normalized


def _default_ppocrv6_model(kind: str) -> str:
    tier = os.getenv("OCR_PPOCRV6_TIER", "small").strip().lower()
    if tier not in {"tiny", "small", "medium"}:
        tier = "small"
    return f"PP-OCRv6_{tier}_{kind}"


def _is_paddleocr_v3_api() -> bool:
    try:
        return "text_detection_model_name" in inspect.signature(PaddleOCR).parameters
    except Exception:
        return False


def _resolve_model_name(kind: str) -> str | None:
    explicit = os.getenv("OCR_TEXT_DETECTION_MODEL" if kind == "det" else "OCR_TEXT_RECOGNITION_MODEL", "").strip()
    if explicit:
        return explicit
    requested = os.getenv("OCR_PPOCR_VERSION", "v6").strip().lower()
    if requested in {"v5", "ppocrv5", "pp-ocrv5"}:
        return "PP-OCRv5_server_det" if kind == "det" else "PP-OCRv5_server_rec"
    if requested in {"v4", "ppocrv4", "pp-ocrv4"}:
        return "PP-OCRv4_server_det" if kind == "det" else "PP-OCRv4_server_rec"
    return _default_ppocrv6_model(kind)


@lru_cache(maxsize=24)
def _get_paddle_ocr(lang: str, use_gpu: bool, det_only: bool, _config_signature: tuple) -> PaddleOCR:
    if _is_paddleocr_v3_api():
        return PaddleOCR(
            text_detection_model_name=_resolve_model_name("det"),
            text_recognition_model_name=_resolve_model_name("rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=_env_bool("OCR_USE_DOC_UNWARPING", False),
            use_textline_orientation=_env_bool("OCR_USE_DOCLINE_ORIENTATION", True),
            text_det_limit_side_len=_env_int("OCR_DET_LIMIT_SIDE_LEN", 1920),
            text_det_limit_type=os.getenv("OCR_DET_LIMIT_TYPE", "max"),
            text_det_thresh=_env_float("OCR_DET_DB_THRESH", 0.20),
            text_det_box_thresh=_env_float("OCR_DET_DB_BOX_THRESH", 0.45),
            text_det_unclip_ratio=_env_float("OCR_DET_DB_UNCLIP_RATIO", 1.80),
            text_rec_score_thresh=_env_float("OCR_REC_SCORE_THRESH", 0.25),
            text_recognition_batch_size=_env_int("OCR_MAX_BATCH_SIZE", 16),
            textline_orientation_batch_size=_env_int("OCR_MAX_BATCH_SIZE", 16),
        )
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=use_gpu,
        det=True,
        rec=not det_only,
        det_limit_side_len=_env_int("OCR_DET_LIMIT_SIDE_LEN", 1536),
        det_limit_type=os.getenv("OCR_DET_LIMIT_TYPE", "max"),
        det_db_thresh=_env_float("OCR_DET_DB_THRESH", 0.25),
        det_db_box_thresh=_env_float("OCR_DET_DB_BOX_THRESH", 0.58),
        det_db_unclip_ratio=_env_float("OCR_DET_DB_UNCLIP_RATIO", 1.25),
        use_dilation=_env_bool("OCR_USE_DILATION", False),
        drop_score=_env_float("OCR_DROP_SCORE", 0.45),
        max_batch_size=_env_int("OCR_MAX_BATCH_SIZE", 10),
        show_log=_env_bool("OCR_SHOW_LOG", True),
    )


@lru_cache(maxsize=4)
def _get_vietocr_predictor(use_gpu: bool):
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError as exc:
        raise RuntimeError(
            "VietOCR chưa được cài. Hãy cài `vietocr` rồi chạy lại nếu muốn dùng engine vietocr."
        ) from exc

    config_name = os.getenv("VIETOCR_CONFIG", "vgg_transformer")
    config = Cfg.load_config_from_name(config_name)
    config["device"] = "cuda:0" if use_gpu else "cpu"
    if "predictor" in config:
        config["predictor"]["beamsearch"] = _env_bool("VIETOCR_BEAMSEARCH", False)
    weights = os.getenv("VIETOCR_WEIGHTS", "").strip()
    if weights:
        config["weights"] = weights
    return Predictor(config)


def _normalize_lines(result):
    if not result:
        return []
    return result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result


def _box_to_rect(box, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x_coords = [p[0] / scale_x for p in box]
    y_coords = [p[1] / scale_y for p in box]
    x1, y1 = float(min(x_coords)), float(min(y_coords))
    x2, y2 = float(max(x_coords)), float(max(y_coords))
    return x1, y1, x2, y2


def _box_to_quad(box, scale_x: float, scale_y: float) -> tuple[tuple[float, float], ...]:
    return tuple((float(p[0] / scale_x), float(p[1] / scale_y)) for p in box[:4])


def _paddle_v3_result_payload(result_item) -> dict:
    if result_item is None:
        return {}
    if isinstance(result_item, dict):
        return result_item.get("res", result_item)
    json_payload = getattr(result_item, "json", None)
    if isinstance(json_payload, dict):
        return json_payload.get("res", json_payload)
    try:
        as_dict = dict(result_item)
        return as_dict.get("res", as_dict)
    except Exception:
        return {}


def _poly_to_box_points(poly) -> list[list[float]]:
    points = []
    for pt in list(poly)[:4]:
        try:
            points.append([float(pt[0]), float(pt[1])])
        except Exception:
            pass
    if len(points) == 4:
        return points
    if len(poly) >= 4 and not isinstance(poly[0], (list, tuple, np.ndarray)):
        x1, y1, x2, y2 = [float(v) for v in list(poly)[:4]]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return points


def _run_paddle_predict(ocr: PaddleOCR, image_input):
    if hasattr(ocr, "predict"):
        return list(ocr.predict(image_input))
    try:
        return ocr.ocr(image_input, cls=True)
    except TypeError:
        return ocr.ocr(image_input)


def _paddle_records(image_input, lang: str, use_gpu: bool, det_only: bool):
    ocr = _get_paddle_ocr(lang or "vi", use_gpu, det_only=det_only, _config_signature=_ocr_config_signature())
    result = _run_paddle_predict(ocr, image_input)
    if _is_paddleocr_v3_api():
        records = []
        for item in result:
            payload = _paddle_v3_result_payload(item)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            polys = payload.get("rec_polys") or payload.get("dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                box = _poly_to_box_points(poly)
                if len(box) == 4:
                    records.append({"box": box, "text": str(text).strip(), "score": float(score)})
        return records

    records = []
    for line in _normalize_lines(result):
        if not line or len(line) < 2:
            continue
        text = str(line[1][0]).strip() if line[1] else ""
        score = float(line[1][1]) if line[1] and len(line[1]) > 1 else 0.0
        records.append({"box": line[0], "text": text, "score": score})
    return records


def _crop_from_box(image: np.ndarray, box) -> Image.Image | None:
    h, w = image.shape[:2]
    xs = [int(round(p[0])) for p in box]
    ys = [int(round(p[1])) for p in box]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(w, max(xs))
    y2 = min(h, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def _perspective_crop_from_box(image: np.ndarray, box) -> Image.Image | None:
    if image is None or len(box) < 4:
        return None

    pts = np.array(box[:4], dtype=np.float32)
    width_a = np.linalg.norm(pts[2] - pts[3])
    width_b = np.linalg.norm(pts[1] - pts[0])
    height_a = np.linalg.norm(pts[1] - pts[2])
    height_b = np.linalg.norm(pts[0] - pts[3])
    width = max(int(round(max(width_a, width_b))), 1)
    height = max(int(round(max(height_a, height_b))), 1)
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    if warped is None or warped.size == 0:
        return _crop_from_box(image, box)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


def _run_paddle_ocr(image_input, lang: str, use_gpu: bool, scale_x: float, scale_y: float) -> List[OCRNode]:
    nodes: List[OCRNode] = []
    for record in _paddle_records(image_input, lang, use_gpu, det_only=False):
        box = record["box"]
        text = _normalize_ocr_text(record["text"])
        score = float(record["score"])
        if not text:
            continue
        x1, y1, x2, y2 = _box_to_rect(box, scale_x, scale_y)
        quad = _box_to_quad(box, scale_x, scale_y)
        nodes.append(
            OCRNode(
                text=text,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
                quad=quad,
            )
        )
    return _merge_nodes_same_line(nodes)


def _run_vietocr_hybrid(image_input, lang: str, use_gpu: bool, scale_x: float, scale_y: float) -> List[OCRNode]:
    viet_use_gpu = _resolve_vietocr_gpu()
    predictor = None
    vietocr_available = True
    vietocr_error = ""
    try:
        predictor = _get_vietocr_predictor(viet_use_gpu)
    except Exception as exc:
        vietocr_available = False
        vietocr_error = str(exc)
        print(f"[OCR] VietOCR unavailable, fallback to Paddle text: {exc}")
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input
    if image is None:
        _set_last_ocr_runtime_info(
            requested_engine="vietocr",
            actual_engine="empty",
            recognizer_backend="none",
            paddleocr_api="v3" if _is_paddleocr_v3_api() else "v2",
            text_detection_model=_resolve_model_name("det") if _is_paddleocr_v3_api() else "",
            text_recognition_model=_resolve_model_name("rec") if _is_paddleocr_v3_api() else "",
            vietocr_available=vietocr_available,
            vietocr_error=vietocr_error,
            used_gpu=viet_use_gpu,
            node_count=0,
        )
        return []

    # Pass 1: collect detected boxes + Paddle fallback text, and crop each box.
    records: list[dict] = []
    for paddle_record in _paddle_records(image_input, lang, use_gpu, det_only=False):
        box = paddle_record["box"]
        paddle_text = str(paddle_record["text"]).strip()
        paddle_score = float(paddle_record["score"])
        crop = _perspective_crop_from_box(image, box) if (vietocr_available and predictor is not None) else None
        records.append({"box": box, "text": paddle_text, "score": paddle_score, "crop": crop})

    # Pass 2: recognize all crops in a single batched VietOCR call (GPU-friendly).
    if vietocr_available and predictor is not None:
        batch_imgs = [r["crop"] for r in records if r["crop"] is not None]
        batch_pos = [i for i, r in enumerate(records) if r["crop"] is not None]
        if batch_imgs:
            try:
                viet_texts, viet_probs = predictor.predict_batch(batch_imgs, return_prob=True)
                for pos, vtext, vprob in zip(batch_pos, viet_texts, viet_probs):
                    vtext = str(vtext).strip()
                    if vtext:
                        text, score, source = _choose_hybrid_text(
                            records[pos].get("text", ""),
                            float(records[pos].get("score", 0.0) or 0.0),
                            vtext,
                            float(vprob),
                        )
                        records[pos]["text"] = text
                        records[pos]["text_source"] = source
                        try:
                            records[pos]["score"] = float(score)
                        except Exception:
                            records[pos]["score"] = 1.0
            except Exception as exc:
                print(f"[OCR] VietOCR batch predict failed, fallback to Paddle text: {exc}")

    # Pass 3: build nodes.
    nodes: List[OCRNode] = []
    for r in records:
        text = _normalize_ocr_text(r["text"])
        if not text:
            continue
        x1, y1, x2, y2 = _box_to_rect(r["box"], scale_x, scale_y)
        quad = _box_to_quad(r["box"], scale_x, scale_y)
        nodes.append(
            OCRNode(
                text=text,
                score=r["score"],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
                quad=quad,
            )
        )
    _set_last_ocr_runtime_info(
        requested_engine="vietocr",
        actual_engine="vietocr" if vietocr_available and predictor is not None else "paddle_fallback",
        recognizer_backend="vietocr" if vietocr_available and predictor is not None else "paddle",
        paddleocr_api="v3" if _is_paddleocr_v3_api() else "v2",
        text_detection_model=_resolve_model_name("det") if _is_paddleocr_v3_api() else "",
        text_recognition_model=_resolve_model_name("rec") if _is_paddleocr_v3_api() else "",
        vietocr_available=vietocr_available,
        vietocr_error=vietocr_error,
        used_gpu=viet_use_gpu,
        node_count=len(nodes),
    )
    return _merge_nodes_same_line(nodes)


def _merge_node_group(group: list[OCRNode]) -> OCRNode:
    group = sorted(group, key=lambda n: (n.x1, n.y1))
    text = " ".join(str(n.text).strip() for n in group if str(n.text).strip())
    score = sum(float(n.score) for n in group) / max(len(group), 1)
    x1 = min(n.x1 for n in group)
    y1 = min(n.y1 for n in group)
    x2 = max(n.x2 for n in group)
    y2 = max(n.y2 for n in group)
    quad = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return OCRNode(
        text=text,
        score=score,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx=(x1 + x2) / 2,
        cy=(y1 + y2) / 2,
        w=x2 - x1,
        h=y2 - y1,
        quad=quad,
    )


def _is_numeric_like(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    return bool(any(ch.isdigit() for ch in t)) and bool(
        all(ch.isdigit() or ch in ".,:/-()% " for ch in t)
    )


def _has_vietnamese_letters(text: str) -> bool:
    return bool(re.search(r"[ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", str(text or "").lower()))


def _normalize_ocr_text(text: str) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    replacements = {
        "HÒA ĐƠN": "HÓA ĐƠN",
        "HOA ĐƠN": "HÓA ĐƠN",
        "HÓA DON": "HÓA ĐƠN",
        "BẢN HÀNG": "BÁN HÀNG",
        "Thành tiến": "Thành tiền",
        "Thành tien": "Thành tiền",
        "Tőng": "Tổng",
        "T6ng": "Tổng",
        "T8ng": "Tổng",
        "Tién": "Tiền",
        "trã": "trả",
        "cám an": "cảm ơn",
        "hen găp lai": "hẹn gặp lại",
        "găp lai": "gặp lại",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"(Ngày:\s*\d{1,2}/\d{1,2}/\d{4}-\d{1,2})[.](\d{2})", r"\1:\2", out)
    return out


def _choose_hybrid_text(paddle_text: str, paddle_score: float, viet_text: str, viet_score: float) -> tuple[str, float, str]:
    paddle_text = _normalize_ocr_text(paddle_text)
    viet_text = _normalize_ocr_text(viet_text)
    if not viet_text:
        return paddle_text, paddle_score, "paddle_empty_vietocr"
    if not paddle_text:
        return viet_text, viet_score, "vietocr_empty_paddle"

    compact_paddle = re.sub(r"\s+", "", paddle_text)
    compact_viet = re.sub(r"\s+", "", viet_text)
    paddle_alpha_upper = bool(re.fullmatch(r"[A-Z0-9 &.-]+", paddle_text))
    viet_lost_spaces = " " in paddle_text and " " not in viet_text and compact_paddle.lower() == compact_viet.lower()
    if paddle_alpha_upper and viet_lost_spaces and paddle_score >= 0.95:
        return paddle_text, paddle_score, "paddle_preserve_spaces"

    if _is_numeric_like(paddle_text) or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", paddle_text):
        if paddle_score >= 0.90:
            return paddle_text, paddle_score, "paddle_numeric"

    keyword_gain = any(k in viet_text.lower() for k in ["tổng", "tiền", "đơn giá", "hàng", "ngân", "cảm ơn", "hẹn gặp"])
    if _has_vietnamese_letters(viet_text) and (keyword_gain or viet_score >= 0.82):
        return viet_text, viet_score, "vietocr_vietnamese"

    if viet_score >= paddle_score + 0.05:
        return viet_text, viet_score, "vietocr_score"
    return paddle_text, paddle_score, "paddle_score"


def _line_groups(nodes: List[OCRNode]) -> list[list[OCRNode]]:
    if not nodes:
        return []
    ordered = sorted(nodes, key=lambda n: (n.cy, n.x1))
    groups: list[list[OCRNode]] = []
    for node in ordered:
        if not groups:
            groups.append([node])
            continue
        current = groups[-1]
        ref_cy = sum(n.cy for n in current) / len(current)
        ref_h = max(max(n.h for n in current), node.h, 1.0)
        if abs(node.cy - ref_cy) <= ref_h * _env_float("OCR_LINE_CLUSTER_RATIO", 0.55):
            current.append(node)
        else:
            groups.append([node])
    return [sorted(group, key=lambda n: n.x1) for group in groups]


def _should_merge_table_nodes(a: OCRNode, b: OCRNode, row_size: int) -> bool:
    if row_size < _env_int("OCR_TABLE_ROW_MIN_CELLS", 4):
        return False
    if _is_numeric_like(a.text) and _is_numeric_like(b.text):
        return False
    if _is_numeric_like(b.text) and len(str(a.text or "").strip()) >= 4:
        return False
    if _is_numeric_like(a.text) and len(str(b.text or "").strip()) >= 4:
        return False
    height_ref = max(a.h, b.h, 1.0)
    gap = b.x1 - a.x2
    center_delta = abs(a.cy - b.cy)
    if center_delta > height_ref * _env_float("OCR_TABLE_MERGE_LINE_Y_RATIO", 0.38):
        return False
    if gap < -height_ref * 0.15:
        return False
    if gap > height_ref * _env_float("OCR_TABLE_MERGE_GAP_RATIO", 0.7):
        return False
    return True


def _should_merge_nodes(a: OCRNode, b: OCRNode) -> bool:
    if not str(a.text).strip() or not str(b.text).strip():
        return False
    height_ref = max(a.h, b.h, 1.0)
    center_delta = abs(a.cy - b.cy)
    if center_delta > height_ref * _env_float("OCR_MERGE_LINE_Y_RATIO", 0.45):
        return False
    gap = b.x1 - a.x2
    if gap < -height_ref * 0.25:
        return False
    if gap > height_ref * _env_float("OCR_MERGE_GAP_RATIO", 1.2):
        return False
    height_ratio = max(a.h, b.h) / max(min(a.h, b.h), 1.0)
    if height_ratio > _env_float("OCR_MERGE_HEIGHT_RATIO", 1.8):
        return False
    return True


def _merge_nodes_same_line(nodes: List[OCRNode]) -> List[OCRNode]:
    if not _env_bool("OCR_MERGE_SAME_LINE", True) or len(nodes) <= 1:
        return nodes
    merged: list[OCRNode] = []
    for row in _line_groups(nodes):
        current_group: list[OCRNode] = []
        row_size = len(row)
        for node in row:
            if not current_group:
                current_group = [node]
                continue
            prev = current_group[-1]
            can_merge = _should_merge_nodes(prev, node)
            if not can_merge and _env_bool("OCR_TABLE_AWARE_MERGE", True):
                can_merge = _should_merge_table_nodes(prev, node, row_size)
            if can_merge:
                current_group.append(node)
            else:
                merged.append(_merge_node_group(current_group))
                current_group = [node]
        if current_group:
            merged.append(_merge_node_group(current_group))
    return merged


def _dispatch_engine(
    image_input, lang: str, use_gpu: bool, selected_engine: str, scale_x: float, scale_y: float
) -> List[OCRNode]:
    if selected_engine == "vietocr":
        return _run_vietocr_hybrid(image_input, lang, use_gpu, scale_x, scale_y)
    nodes = _run_paddle_ocr(image_input, lang, use_gpu, scale_x, scale_y)
    _set_last_ocr_runtime_info(
        requested_engine="paddle",
        actual_engine="paddle",
        recognizer_backend="paddle",
        paddleocr_api="v3" if _is_paddleocr_v3_api() else "v2",
        text_detection_model=_resolve_model_name("det") if _is_paddleocr_v3_api() else "",
        text_recognition_model=_resolve_model_name("rec") if _is_paddleocr_v3_api() else "",
        vietocr_available=False,
        vietocr_error="",
        used_gpu=use_gpu,
        node_count=len(nodes),
    )
    return nodes


def _processed_at_original_scale(image_input, scale_x: float, scale_y: float) -> np.ndarray | None:
    if isinstance(image_input, str):
        return cv2.imread(image_input)
    if image_input is not None and (scale_x != 1.0 or scale_y != 1.0):
        target_w = max(1, int(round(image_input.shape[1] / max(scale_x, 1e-6))))
        target_h = max(1, int(round(image_input.shape[0] / max(scale_y, 1e-6))))
        return cv2.resize(image_input, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return image_input


def run_ocr(image_path: str, lang: str = "vi", engine: str | None = None, overrides: dict | None = None) -> List[OCRNode]:
    with temporary_ocr_config(overrides):
        use_gpu = _resolve_use_gpu()
        image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
        selected_engine = _resolve_ocr_engine(engine)
        return _dispatch_engine(image_input, lang, use_gpu, selected_engine, scale_x, scale_y)


def run_ocr_on_prepared(
    image_input,
    scale_x: float,
    scale_y: float,
    lang: str = "vi",
    engine: str | None = None,
) -> Tuple[List[OCRNode], np.ndarray | None]:
    """Detect+recognize stage on an already-normalized image. Caller must hold the OCR
    config context (env) for the duration; used by ``run_ocr_with_processed``."""
    use_gpu = _resolve_use_gpu()
    selected_engine = _resolve_ocr_engine(engine)
    nodes = _dispatch_engine(image_input, lang, use_gpu, selected_engine, scale_x, scale_y)
    processed = _processed_at_original_scale(image_input, scale_x, scale_y)
    return nodes, processed


def run_ocr_with_processed(
    image_path: str, lang: str = "vi", engine: str | None = None, overrides: dict | None = None
) -> Tuple[List[OCRNode], np.ndarray | None]:
    """Run OCR and return ``(nodes, processed_image)`` from a single preprocessing pass.

    The batch labeling prep needs both the OCR nodes and the preprocessed image (to
    save/draw debug boxes). Calling ``run_ocr`` and ``prepare_ocr_image`` separately
    would run the heavy normalize/denoise pipeline twice per image; this does it once.
    """
    with temporary_ocr_config(overrides):
        image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
        return run_ocr_on_prepared(image_input, scale_x, scale_y, lang, engine)


def prepare_ocr_image(image_path: str, overrides: dict | None = None) -> np.ndarray | None:
    with temporary_ocr_config(overrides):
        image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
        return _processed_at_original_scale(image_input, scale_x, scale_y)
