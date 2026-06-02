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
    CSV_FIELDS = ["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"]

    def _write_rows_csv(self, csv_path: Path, rows: list[dict[str, str]]) -> None:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _read_rows_csv(self, csv_path: Path) -> list[dict[str, str]]:
        if not csv_path.exists():
            return []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append({k: str(row.get(k, "")) for k in self.CSV_FIELDS})
            return rows

    def prepare(
        self,
        input_dir: str,
        output_dir: str,
        lang: str = "en",
        save_debug_images: bool = True,
        copy_images: bool = True,
        num_workers: int = 1,
        worker_index: int = 0,
        save_every_images: int = 10,
    ) -> dict[str, str]:
        in_dir = Path(input_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if worker_index < 0 or worker_index >= num_workers:
            raise ValueError("worker_index must be in [0, num_workers-1]")
        if save_every_images < 1:
            raise ValueError("save_every_images must be >= 1")

        out_dir = Path(output_dir)
        if num_workers > 1:
            out_dir = out_dir / f"worker_{worker_index}"
        images_out = out_dir / "images"
        ocr_json_out = out_dir / "ocr_json"
        debug_out = out_dir / "debug_boxes"

        out_dir.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)
        ocr_json_out.mkdir(parents=True, exist_ok=True)
        if save_debug_images:
            debug_out.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "nodes_to_label.csv"
        existing_rows = self._read_rows_csv(csv_path)
        rows: list[dict[str, str]] = list(existing_rows)
        done_doc_ids = {r.get("doc_id", "").strip() for r in existing_rows if r.get("doc_id", "").strip()}

        image_paths = sorted(
            p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not image_paths:
            raise ValueError(f"No supported image files in: {input_dir}")
        if num_workers > 1:
            image_paths = image_paths[worker_index::num_workers]
            if not image_paths:
                raise ValueError(
                    f"Worker {worker_index}/{num_workers} got 0 images. "
                    "Reduce num_workers or check input files."
                )
        total_before_resume = len(image_paths)
        if done_doc_ids:
            image_paths = [p for p in image_paths if p.stem not in done_doc_ids]
        skipped = total_before_resume - len(image_paths)
        logger.info(
            "Resume check: total={}, skipped_done={}, remaining={}",
            total_before_resume,
            skipped,
            len(image_paths),
        )
        if not image_paths:
            logger.info("No remaining images to OCR. nodes_to_label.csv already up-to-date: {}", csv_path)
            return {
                "output_dir": str(out_dir),
                "nodes_csv": str(csv_path),
                "images_dir": str(images_out),
                "ocr_json_dir": str(ocr_json_out),
                "debug_boxes_dir": str(debug_out) if save_debug_images else "",
                "skipped_done_images": str(skipped),
                "processed_images": "0",
            }

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

            # Persist progress by image-batch.
            if (idx + 1) % save_every_images == 0:
                self._write_rows_csv(csv_path, rows)
                logger.info(
                    "Saved batch CSV at {}/{} images (batch={}): {}",
                    idx + 1,
                    len(image_paths),
                    save_every_images,
                    csv_path,
                )

        self._write_rows_csv(csv_path, rows)

        logger.info(
            "Prepared labeling data at: {} (worker {}/{}, images={})",
            out_dir,
            worker_index,
            num_workers,
            len(image_paths),
        )
        logger.info("CSV to label: {}", csv_path)

        return {
            "output_dir": str(out_dir),
            "nodes_csv": str(csv_path),
            "images_dir": str(images_out),
            "ocr_json_dir": str(ocr_json_out),
            "debug_boxes_dir": str(debug_out) if save_debug_images else "",
            "skipped_done_images": str(skipped),
            "processed_images": str(len(image_paths)),
        }
