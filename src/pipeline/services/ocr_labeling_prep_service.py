from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

import cv2
from loguru import logger

from pipeline.core.ocr_engine import prepare_ocr_image, run_ocr
from pipeline.core.visualize import draw_boxes, draw_boxes_on_array


class OCRLabelingPrepService:
    """Prepare OCR outputs for manual labeling from a folder of invoice images."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    CSV_FIELDS = ["doc_id", "text", "label", "x1", "y1", "x2", "y2", "score"]

    def _write_text_lines(self, path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_rows_csv(self, csv_path: Path, rows: list[dict[str, str]]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(csv_path.parent),
            delete=False,
            suffix=".tmp",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = Path(f.name)
        tmp_path.replace(csv_path)

    def _read_rows_csv(self, csv_path: Path) -> list[dict[str, str]]:
        if not csv_path.exists():
            return []

        raw = csv_path.read_bytes()
        if b"\x00" in raw:
            backup_path = csv_path.with_suffix(csv_path.suffix + ".broken")
            try:
                shutil.copy2(csv_path, backup_path)
            except Exception:
                backup_path = None
            raw = raw.replace(b"\x00", b"")
            if not raw.strip():
                logger.warning(
                    "CSV {} is corrupted with NUL bytes and contains no recoverable rows. Backup: {}",
                    csv_path,
                    backup_path,
                )
                return []
            logger.warning(
                "CSV {} contains NUL bytes. Attempting recovery from sanitized content. Backup: {}",
                csv_path,
                backup_path,
            )

        text = raw.decode("utf-8", errors="ignore")
        reader = csv.DictReader(text.splitlines())
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({k: str(row.get(k, "")) for k in self.CSV_FIELDS})
        return rows

    def _rebuild_rows_from_ocr_json(self, ocr_json_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
        rows: list[dict[str, str]] = []
        corrupted_files: list[str] = []
        if not ocr_json_dir.exists():
            return rows, corrupted_files
        for json_path in sorted(ocr_json_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skip corrupted OCR JSON {}: {}", json_path, exc)
                corrupted_files.append(f"{json_path.name}\t{exc}")
                continue
            doc_id = str(payload.get("doc_id", json_path.stem))
            for node in payload.get("nodes", []):
                bbox = node.get("bbox", [0, 0, 0, 0])
                rows.append(
                    {
                        "doc_id": doc_id,
                        "text": str(node.get("text", "")),
                        "label": "",
                        "x1": f"{float(bbox[0]):.2f}",
                        "y1": f"{float(bbox[1]):.2f}",
                        "x2": f"{float(bbox[2]):.2f}",
                        "y2": f"{float(bbox[3]):.2f}",
                        "score": f"{float(node.get('score', 0.0)):.4f}",
                    }
                )
        return rows, corrupted_files

    def _collect_done_doc_ids_from_ocr_json(self, ocr_json_dir: Path) -> set[str]:
        if not ocr_json_dir.exists():
            return set()
        return {p.stem for p in ocr_json_dir.glob("*.json") if p.is_file()}

    def prepare(
        self,
        input_dir: str,
        output_dir: str,
        lang: str = "en",
        engine: str = "paddle",
        ocr_overrides: dict | None = None,
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
        failed_images_path = out_dir / "failed_images.txt"
        corrupted_ocr_json_path = out_dir / "corrupted_ocr_json.txt"

        out_dir.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)
        ocr_json_out.mkdir(parents=True, exist_ok=True)
        if save_debug_images:
            debug_out.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "nodes_to_label.csv"
        existing_rows = self._read_rows_csv(csv_path)
        corrupted_ocr_json_files: list[str] = []
        if not existing_rows:
            rebuilt_rows, corrupted_ocr_json_files = self._rebuild_rows_from_ocr_json(ocr_json_out)
            if rebuilt_rows:
                existing_rows = rebuilt_rows
                self._write_rows_csv(csv_path, existing_rows)
                logger.warning(
                    "Rebuilt {} CSV rows from {} OCR JSON files: {}",
                    len(existing_rows),
                    len(self._collect_done_doc_ids_from_ocr_json(ocr_json_out)),
                    csv_path,
                )
        if corrupted_ocr_json_files:
            self._write_text_lines(corrupted_ocr_json_path, corrupted_ocr_json_files)
        elif corrupted_ocr_json_path.exists():
            corrupted_ocr_json_path.unlink()

        rows: list[dict[str, str]] = list(existing_rows)
        done_doc_ids = {r.get("doc_id", "").strip() for r in existing_rows if r.get("doc_id", "").strip()}
        if not done_doc_ids:
            done_doc_ids = self._collect_done_doc_ids_from_ocr_json(ocr_json_out)
            if done_doc_ids:
                logger.warning(
                    "Resume fallback: CSV has no recoverable rows. Using {} OCR JSON files as processed markers.",
                    len(done_doc_ids),
                )
        failed_images: list[str] = []

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
            try:
                processed_image = prepare_ocr_image(str(image_path), overrides=ocr_overrides)
                nodes = run_ocr(str(image_path), lang=lang, engine=engine, overrides=ocr_overrides)

                if copy_images:
                    target_img = images_out / image_path.name
                    if processed_image is not None:
                        cv2.imwrite(str(target_img), processed_image)
                    elif target_img.resolve() != image_path.resolve():
                        shutil.copy2(image_path, target_img)
                else:
                    target_img = image_path

                if save_debug_images:
                    debug_path = debug_out / f"{doc_id}_boxes.jpg"
                    if processed_image is not None:
                        draw_boxes_on_array(processed_image, nodes, str(debug_path))
                    else:
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
                            "quad": [[px, py] for px, py in (n.quad or ())],
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
            except Exception as exc:
                failed_images.append(f"{image_path.name}\t{exc}")
                logger.error("OCR failed for {}: {}", image_path, exc)
                continue

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
                if failed_images:
                    self._write_text_lines(failed_images_path, failed_images)

        self._write_rows_csv(csv_path, rows)
        if failed_images:
            self._write_text_lines(failed_images_path, failed_images)
        elif failed_images_path.exists():
            failed_images_path.unlink()

        logger.info(
            "Prepared labeling data at: {} (worker {}/{}, images={}, failed={})",
            out_dir,
            worker_index,
            num_workers,
            len(image_paths),
            len(failed_images),
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
            "failed_images_file": str(failed_images_path) if failed_images else "",
            "corrupted_ocr_json_file": str(corrupted_ocr_json_path) if corrupted_ocr_json_files else "",
        }
