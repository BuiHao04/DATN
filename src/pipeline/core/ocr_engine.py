from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
from PIL import Image

from pipeline.core.schema import OCRNode


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
    if env_force in {"1", "true", "yes"}:
        return True
    if env_force in {"0", "false", "no"}:
        return False
    return bool(paddle.is_compiled_with_cuda())


def _resolve_ocr_engine(engine: str | None = None) -> str:
    value = str(engine or os.getenv("OCR_ENGINE", "paddle")).strip().lower()
    if value in {"vietocr", "paddle_vietocr", "paddle+vietocr"}:
        return "vietocr"
    return "paddle"


def _prepare_image_for_ocr(image_path: str) -> Tuple[np.ndarray | str, float, float]:
    if not _env_bool("OCR_PREPROCESS", True):
        return image_path, 1.0, 1.0

    image = cv2.imread(image_path)
    if image is None:
        return image_path, 1.0, 1.0

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


@lru_cache(maxsize=12)
def _get_paddle_ocr(lang: str, use_gpu: bool, det_only: bool) -> PaddleOCR:
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=use_gpu,
        det=True,
        rec=not det_only,
        det_limit_side_len=_env_int("OCR_DET_LIMIT_SIDE_LEN", 1280),
        det_limit_type=os.getenv("OCR_DET_LIMIT_TYPE", "max"),
        det_db_thresh=_env_float("OCR_DET_DB_THRESH", 0.25),
        det_db_box_thresh=_env_float("OCR_DET_DB_BOX_THRESH", 0.5),
        det_db_unclip_ratio=_env_float("OCR_DET_DB_UNCLIP_RATIO", 1.6),
        use_dilation=_env_bool("OCR_USE_DILATION", False),
        drop_score=_env_float("OCR_DROP_SCORE", 0.4),
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


def _run_paddle_ocr(image_input, lang: str, use_gpu: bool, scale_x: float, scale_y: float) -> List[OCRNode]:
    ocr = _get_paddle_ocr(lang or "vi", use_gpu, det_only=False)
    try:
        result = ocr.ocr(image_input, cls=True)
    except TypeError:
        result = ocr.ocr(image_input)

    nodes: List[OCRNode] = []
    for line in _normalize_lines(result):
        if not line or len(line) < 2:
            continue
        box = line[0]
        text = line[1][0]
        score = float(line[1][1])
        x1, y1, x2, y2 = _box_to_rect(box, scale_x, scale_y)
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
            )
        )
    return nodes


def _run_vietocr_hybrid(image_input, lang: str, use_gpu: bool, scale_x: float, scale_y: float) -> List[OCRNode]:
    detector = _get_paddle_ocr(lang or "vi", use_gpu, det_only=False)
    predictor = _get_vietocr_predictor(use_gpu)
    try:
        result = detector.ocr(image_input, cls=True)
    except TypeError:
        result = detector.ocr(image_input)

    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input
    if image is None:
        return []

    nodes: List[OCRNode] = []
    for line in _normalize_lines(result):
        if not line or len(line) < 2:
            continue
        box = line[0]
        crop = _crop_from_box(image, box)
        if crop is None:
            continue
        try:
            text = str(predictor.predict(crop)).strip()
        except Exception:
            text = ""
        if not text:
            continue
        x1, y1, x2, y2 = _box_to_rect(box, scale_x, scale_y)
        nodes.append(
            OCRNode(
                text=text,
                score=1.0,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                w=x2 - x1,
                h=y2 - y1,
            )
        )
    return nodes


def run_ocr(image_path: str, lang: str = "vi", engine: str | None = None) -> List[OCRNode]:
    use_gpu = _resolve_use_gpu()
    image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
    selected_engine = _resolve_ocr_engine(engine)
    if selected_engine == "vietocr":
        return _run_vietocr_hybrid(image_input, lang, use_gpu, scale_x, scale_y)
    return _run_paddle_ocr(image_input, lang, use_gpu, scale_x, scale_y)
