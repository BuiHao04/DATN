from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoProcessor

from pipeline.core.schema import OCRNode


class PretrainedInferenceService:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def normalize_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[int]:
        return [
            int(1000 * x1 / max(w, 1)),
            int(1000 * y1 / max(h, 1)),
            int(1000 * x2 / max(w, 1)),
            int(1000 * y2 / max(h, 1)),
        ]

    @staticmethod
    def map_labels_to_invoice(words: list[str], labels: list[str]) -> dict[str, object]:
        grouped: dict[str, list[str]] = {}
        for w, l in zip(words, labels):
            grouped.setdefault(l, []).append(w)

        def join_for(*keys: str) -> str | None:
            vals: list[str] = []
            for k, arr in grouped.items():
                lk = k.lower()
                if any(key in lk for key in keys):
                    vals.extend(arr)
            txt = " ".join(vals).strip()
            return txt if txt else None

        return {
            "date": join_for("date", "invoice_date"),
            "tax_code": join_for("tax", "mst", "tin"),
            "total_amount": join_for("total", "amount_due", "grand_total", "payable"),
            "seller_name": join_for("seller", "vendor", "merchant", "company"),
        }

    def infer(self, image_path: str, nodes: list[OCRNode]) -> dict[str, object]:
        logger.info("Running pretrained baseline with model: {}", self.model_id)
        logger.info("Device: {}", self.device)

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        words = [n.text for n in nodes if n.text.strip()]
        boxes = [
            self.normalize_box(n.x1, n.y1, n.x2, n.y2, img_w, img_h)
            for n in nodes
            if n.text.strip()
        ]

        try:
            processor = AutoProcessor.from_pretrained(self.model_id, apply_ocr=False)
            model = AutoModelForTokenClassification.from_pretrained(self.model_id).to(self.device)
        except Exception as e:
            logger.warning("Online load failed, retrying with local cache only: {}", e)
            processor = AutoProcessor.from_pretrained(
                self.model_id, apply_ocr=False, local_files_only=True
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, local_files_only=True
            ).to(self.device)
        model.eval()

        enc = processor(
            images=[image],
            text=[words],
            boxes=[boxes],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(-1).squeeze(0).tolist()

        token_ids = enc["input_ids"].squeeze(0).tolist()
        id2label = model.config.id2label
        special_ids = set(processor.tokenizer.all_special_ids)

        out_words: list[str] = []
        out_labels: list[str] = []
        for token_id, pred_id in zip(token_ids, pred_ids):
            if token_id in special_ids:
                continue
            token = processor.tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("##"):
                continue
            out_words.append(token)
            out_labels.append(id2label.get(int(pred_id), str(pred_id)))

        invoice = self.map_labels_to_invoice(out_words, out_labels)

        return {
            "model_id": self.model_id,
            "tokens": [{"text": t, "label": l} for t, l in zip(out_words, out_labels)],
            "invoice": invoice,
            "note": "Pretrained baseline only (no fine-tune on your VN invoices).",
        }

    @staticmethod
    def load_model_id_from_env(project_dir: str) -> str:
        load_dotenv(Path(project_dir) / ".env", override=False)
        model_id = os.getenv("MODEL_ID")
        if not model_id:
            raise ValueError("Missing MODEL_ID. Put MODEL_ID=... in .env or export env var MODEL_ID.")
        return model_id

    @staticmethod
    def save_result(result: dict[str, object], output_json_path: str) -> str:
        out = Path(output_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
