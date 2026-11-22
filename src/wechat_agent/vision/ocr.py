"""Independent OCR engine abstraction (default: rapidocr-onnxruntime, extensible to Apple Vision)."""

from __future__ import annotations

import io
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image, UnidentifiedImageError

from wechat_agent.models import Box, OcrBox


class OcrError(RuntimeError):
    """Raised when image decoding or OCR inference fails (isolated from raw library tracebacks)."""


@runtime_checkable
class OcrEngine(Protocol):
    def read_boxes(self, image_bytes: bytes) -> list[OcrBox]: ...


class RapidOcrEngine:
    """Offline RapidOCR + ONNXRuntime CPU backend."""

    def __init__(self) -> None:
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read_boxes(self, image_bytes: bytes) -> list[OcrBox]:
        if not image_bytes:
            return []
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                if width <= 0 or height <= 0:
                    return []
                arr = np.asarray(rgb)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OcrError(f"Failed to decode image bytes for OCR: {exc}") from exc

        try:
            result, _ = self._get_engine()(arr)
        except Exception as exc:
            raise OcrError(f"RapidOCR inference failed: {exc}") from exc
        if not result:
            return []

        boxes: list[OcrBox] = []
        for item in result:
            if not item or len(item) < 3:
                continue
            polygon, text, conf = item[0], item[1], item[2]
            text_str = str(text or "").strip()
            if not text_str:
                continue
            xs = [float(pt[0]) / width for pt in polygon]
            ys = [float(pt[1]) / height for pt in polygon]
            left, right = max(0.0, min(xs)), min(1.0, max(xs))
            top, bottom = max(0.0, min(ys)), min(1.0, max(ys))
            if right <= left or bottom <= top:
                continue
            boxes.append(
                OcrBox(
                    text=text_str,
                    box=Box(left, top, right, bottom),
                    confidence=float(conf),
                )
            )
        boxes.sort(key=lambda b: (round(b.box.top, 3), round(b.box.left, 3)))
        return boxes


class AppleVisionOcrEngine:
    """Extension seam for macOS native Vision framework OCR (to be implemented if needed)."""

    def read_boxes(self, image_bytes: bytes) -> list[OcrBox]:
        raise NotImplementedError(
            "AppleVisionOcrEngine is reserved as a protocol extension; use ocr_backend='rapidocr'."
        )


def create_ocr_engine(name: str = "rapidocr") -> OcrEngine:
    key = (name or "rapidocr").strip().lower()
    if key == "rapidocr":
        return RapidOcrEngine()
    if key in {"apple-vision", "apple_vision", "vision"}:
        return AppleVisionOcrEngine()
    raise ValueError(f"Unsupported OCR backend: {name!r}")
