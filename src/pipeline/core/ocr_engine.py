from __future__ import annotations

import os
from contextlib import contextmanager
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


def _ocr_config_signature() -> tuple:
    return (
        _env_int("OCR_DET_LIMIT_SIDE_LEN", 1536),
        os.getenv("OCR_DET_LIMIT_TYPE", "max"),
        _env_float("OCR_DET_DB_THRESH", 0.25),
        _env_float("OCR_DET_DB_BOX_THRESH", 0.58),
        _env_float("OCR_DET_DB_UNCLIP_RATIO", 1.25),
        _env_bool("OCR_USE_DILATION", False),
        _env_float("OCR_DROP_SCORE", 0.45),
        _env_int("OCR_MAX_BATCH_SIZE", 10),
        _env_bool("OCR_SHOW_LOG", True),
    )


@contextmanager
def temporary_ocr_config(overrides: dict | None = None):
    if not overrides:
        yield
        return

    env_map = {
        "det_limit_side_len": "OCR_DET_LIMIT_SIDE_LEN",
        "det_limit_type": "OCR_DET_LIMIT_TYPE",
        "det_db_thresh": "OCR_DET_DB_THRESH",
        "det_db_box_thresh": "OCR_DET_DB_BOX_THRESH",
        "det_db_unclip_ratio": "OCR_DET_DB_UNCLIP_RATIO",
        "use_dilation": "OCR_USE_DILATION",
        "drop_score": "OCR_DROP_SCORE",
        "max_batch_size": "OCR_MAX_BATCH_SIZE",
        "show_log": "OCR_SHOW_LOG",
        "upscale_min_side": "OCR_UPSCALE_MIN_SIDE",
        "upscale_factor": "OCR_UPSCALE_FACTOR",
    }
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


@lru_cache(maxsize=24)
def _get_paddle_ocr(lang: str, use_gpu: bool, det_only: bool, _config_signature: tuple) -> PaddleOCR:
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
    ocr = _get_paddle_ocr(lang or "vi", use_gpu, det_only=False, _config_signature=_ocr_config_signature())
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
    detector = _get_paddle_ocr(lang or "vi", use_gpu, det_only=False, _config_signature=_ocr_config_signature())
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
        crop = _perspective_crop_from_box(image, box)
        if crop is None:
            continue
        try:
            text = str(predictor.predict(crop)).strip()
        except Exception:
            text = ""
        if not text:
            continue
        x1, y1, x2, y2 = _box_to_rect(box, scale_x, scale_y)
        quad = _box_to_quad(box, scale_x, scale_y)
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
                quad=quad,
            )
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


def run_ocr(image_path: str, lang: str = "vi", engine: str | None = None, overrides: dict | None = None) -> List[OCRNode]:
    with temporary_ocr_config(overrides):
        use_gpu = _resolve_use_gpu()
        image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
        selected_engine = _resolve_ocr_engine(engine)
        if selected_engine == "vietocr":
            return _run_vietocr_hybrid(image_input, lang, use_gpu, scale_x, scale_y)
        return _run_paddle_ocr(image_input, lang, use_gpu, scale_x, scale_y)


def prepare_ocr_image(image_path: str, overrides: dict | None = None) -> np.ndarray | None:
    with temporary_ocr_config(overrides):
        image_input, scale_x, scale_y = _prepare_image_for_ocr(image_path)
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        if image_input is not None and (scale_x != 1.0 or scale_y != 1.0):
            target_w = max(1, int(round(image_input.shape[1] / max(scale_x, 1e-6))))
            target_h = max(1, int(round(image_input.shape[0] / max(scale_y, 1e-6))))
            image_input = cv2.resize(image_input, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return image_input
