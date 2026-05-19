from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from loguru import logger

from pipeline.core.ocr_engine import run_ocr
from pipeline.core.visualize import draw_boxes


class OCRLabelingPrepService:
    """Prepare OCR outputs for manual labeling from a folder of invoice images."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def prepare(
        self,
        input_dir: str,
        output_dir: str,
        lang: str = "en",
        save_debug_images: bool = True,
        copy_images: bool = True,
    ) -> dict[str, str]:
        in_dir = Path(input_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        out_dir = Path(output_dir)
        images_out = out_dir / "images"
        ocr_json_out = out_dir / "ocr_json"
        debug_out = out_dir / "debug_boxes"

        out_dir.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)
        ocr_json_out.mkdir(parents=True, exist_ok=True)
        if save_debug_images:
            debug_out.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        image_paths = sorted(
            p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not image_paths:
            raise ValueError(f"No supported image files in: {input_dir}")

        for idx, image_path in enumerate(image_paths):
            doc_id = image_path.stem
            logger.info("OCR [{}/{}]: {}", idx + 1, len(image_paths), image_path.name)
            nodes = run_ocr(str(image_path), lang=lang)

            if copy_images:
                target_img = images_out / image_path.name
                if target_img.resolve() != image_path.resolve():
                    shutil.copy2(image_path, target_img)
            else:
                target_img = image_path

            if save_debug_images:
                debug_path = debug_out / f"{doc_id}_boxes.jpg"
                draw_boxes(str(image_path), nodes, str(debug_path))

            json_path = ocr_json_out / f"{doc_id}.json"
            json_payload = {
                "doc_id": doc_id,
                "image_name": image_path.name,
                "image_path": str(target_img),
                "num_nodes": len(nodes),
                "nodes": [
                    {
                        "text": n.text,
                        "score": n.score,
                        "bbox": [n.x1, n.y1, n.x2, n.y2],
                    }
                    for n in nodes
                ],
            }
            json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            for n in nodes:
                rows.append(
                    {
                        "doc_id": doc_id,
                        "text": n.text,
                        "label": "",  # manual labeling target
                        "x1": f"{n.x1:.2f}",
                        "y1": f"{n.y1:.2f}",
                        "x2": f"{n.x2:.2f}",
                        "y2": f"{n.y2:.2f}",
                        "score": f"{n.score:.4f}",
                    }
                )

        csv_path = out_dir / "nodes_to_label.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Prepared labeling data at: {}", out_dir)
        logger.info("CSV to label: {}", csv_path)

        return {
            "output_dir": str(out_dir),
            "nodes_csv": str(csv_path),
            "images_dir": str(images_out),
            "ocr_json_dir": str(ocr_json_out),
            "debug_boxes_dir": str(debug_out) if save_debug_images else "",
        }
