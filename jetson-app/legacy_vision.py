"""Offline recognition for the legacy dual-camera detector."""

import threading
import time
from dataclasses import dataclass
from math import ceil, floor
from typing import Optional, Tuple

import cv2
import numpy as np

from vision import (
    ColorResult, Detection, LetterboxTransform, classify_hsv, decode_yolo_raw,
    letterbox_rgb, restore_surface_mask,
)


def _center_square_320(frame_bgr: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.size == 0
    ):
        raise ValueError("%s must be a non-empty uint8 HxWx3 BGR array" % name)

    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = frame_bgr[top:top + side, left:left + side]
    return cv2.resize(square, (320, 320), interpolation=cv2.INTER_AREA)


def prepare_legacy_pair(
    left_bgr: np.ndarray, right_bgr: np.ndarray, *, preprocessed=False
) -> Tuple[np.ndarray, np.ndarray, LetterboxTransform]:
    """Return the color stitch, legacy detector tensor, and box transform."""
    if preprocessed:
        for frame in (left_bgr, right_bgr):
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (320, 320, 3):
                raise ValueError('已处理特征拼图必须为 640×320，左右各 320×320')
        stitched = cv2.hconcat([left_bgr, right_bgr])
        tensor, transform = letterbox_rgb(stitched, size=640)
        return stitched, tensor, transform
    left_320 = _center_square_320(left_bgr, "left_bgr")
    right_320 = _center_square_320(right_bgr, "right_bgr")
    stitched = cv2.hconcat([left_320, right_320])
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    tensor, transform = letterbox_rgb(cv2.merge([gray, clahe, edge]), size=640)
    return stitched, tensor, transform


@dataclass(frozen=True)
class LegacyResult:
    detection: Optional[Detection]
    color: Optional[ColorResult]
    mask_status: Tuple[str, str]
    preview_bgr: np.ndarray
    elapsed_ms: float
    error: Optional[str] = None


class LegacyRecognizer:
    """Own the supplied backends on the thread that constructs this recognizer."""

    def __init__(self, detector, segmenter, class_names, config):
        self._detector = detector
        self._segmenter = segmenter
        self._class_names = tuple(class_names)
        self._config = config
        self._owner = threading.current_thread()
        self._closed = False

    def _check_thread(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("legacy recognizer must be used on its owner thread")

    def recognize_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray, preprocessed=False) -> LegacyResult:
        self._check_thread()
        if self._closed:
            raise RuntimeError("legacy recognizer is closed")
        started = time.perf_counter()
        stitched, tensor, transform = prepare_legacy_pair(left_bgr, right_bgr, preprocessed=preprocessed)
        preview = stitched.copy()

        model = self._config.get("model", {})
        try:
            detections = decode_yolo_raw(
                self._detector.infer(tensor),
                transform,
                self._class_names,
                model.get("confidence_threshold", 0.45),
                model.get("iou_threshold", 0.50),
                expected_count=26,
            )
        except Exception as exc:
            return LegacyResult(
                None, None, ("detection_error", "detection_error"), preview,
                (time.perf_counter() - started) * 1000, str(exc) or type(exc).__name__,
            )

        sides = ([], [])
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            # The box height is the same on both sides, so compare widths.
            left_overlap = min(x2, 320) - min(x1, 320)
            right_overlap = (x2 - x1) - left_overlap
            side = int(right_overlap > left_overlap)
            offset = side * 320
            local_box = (max(0, x1 - offset), y1, min(320, x2 - offset), y2)
            sides[side].append((detection, local_box))

        statuses = []
        colors = []
        full_roi = np.ones((320, 320), dtype=bool)
        for side, candidates in enumerate(sides):
            offset = side * 320
            half = stitched[:, offset:offset + 320]
            target = preview[:, offset:offset + 320]
            if preprocessed:
                statuses.append('feature_input' if candidates else 'no_detection')
                continue
            if candidates:
                detection, box = candidates[0]
                colors.append((classify_hsv(half, full_roi, box, self._config), detection.confidence))
            statuses.append(self._overlay(half, target, candidates))

        color = None
        if colors:
            unknown = self._config.get("color", self._config).get(
                "unknown_color", self._config.get("recognition", {}).get("unknown_color", "\u672a\u77e5\u989c\u8272")
            )
            color, _ = max(colors, key=lambda item: (item[0].name != unknown, item[0].confidence, item[1]))

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(
                preview, (floor(x1), floor(y1)), (ceil(x2) - 1, ceil(y2) - 1), (0, 255, 0), 2,
            )
        return LegacyResult(
            detections[0] if detections else None,
            color,
            (statuses[0], statuses[1]),
            preview,
            (time.perf_counter() - started) * 1000,
        )

    def _overlay(self, half, preview, candidates):
        if not candidates:
            return "no_detection"
        if self._segmenter is None:
            return "disabled"
        if len(candidates) != 1:
            return "ambiguous"
        try:
            tensor, transform = letterbox_rgb(half, 320)
            mask = restore_surface_mask(self._segmenter.infer(tensor), transform, threshold=0.5)
        except Exception:
            return "error"
        if not mask.any():
            return "empty"
        x1, y1, x2, y2 = candidates[0][1]
        if not mask[floor(y1):ceil(y2), floor(x1):ceil(x2)].any():
            return "unmatched"

        pixels = preview[mask].astype(np.float32)
        preview[mask] = np.rint(pixels * 0.65 + np.array((0, 255, 0)) * 0.35).astype(np.uint8)
        return "applied"

    def close(self):
        self._check_thread()
        if self._closed:
            return
        self._closed = True
        try:
            self._detector.close()
        finally:
            if self._segmenter is not None and self._segmenter is not self._detector:
                self._segmenter.close()
