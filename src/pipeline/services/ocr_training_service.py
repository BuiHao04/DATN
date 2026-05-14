from __future__ import annotations

import subprocess
from loguru import logger


class OCRTrainingService:
    """
    Wrapper service for OCR training scripts.
    Supports running external command so you can switch between PaddleOCR/Tesseract/TrOCR training pipelines.
    """

    def train(self, command: str, workdir: str | None = None) -> int:
        logger.info("Running OCR training command: {}", command)
        completed = subprocess.run(command, shell=True, cwd=workdir)
        if completed.returncode != 0:
            raise RuntimeError(f"OCR training command failed with code {completed.returncode}")
        return completed.returncode
