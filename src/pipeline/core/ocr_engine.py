from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

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


@lru_cache(maxsize=8)
def _get_ocr(lang: str, use_gpu: bool) -> PaddleOCR:
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=use_gpu,
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


def run_ocr(image_path: str, lang: str = "vi") -> List[OCRNode]:
    use_gpu = _resolve_use_gpu()
    ocr = _get_ocr(lang or "vi", use_gpu)
    image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)

    try:
        result = ocr.ocr(image_input, cls=True)
    except TypeError:
        result = ocr.ocr(image_input)

    nodes: List[OCRNode] = []
    if not result:
        return nodes

    lines = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result

    for line in lines:
        if not line or len(line) < 2:
            continue

        box = line[0]
        text = line[1][0]
        score = float(line[1][1])

        x_coords = [p[0] / scale_x for p in box]
        y_coords = [p[1] / scale_y for p in box]
        x1, y1 = float(min(x_coords)), float(min(y_coords))
        x2, y2 = float(max(x_coords)), float(max(y_coords))

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
