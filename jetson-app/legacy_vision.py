"""Offline preprocessing for the legacy dual-camera detector."""

from typing import Tuple

import cv2
import numpy as np

from vision import LetterboxTransform, letterbox_rgb


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
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, LetterboxTransform]:
    """Return the color stitch, legacy detector tensor, and box transform."""
    left_320 = _center_square_320(left_bgr, "left_bgr")
    right_320 = _center_square_320(right_bgr, "right_bgr")
    stitched = cv2.hconcat([left_320, right_320])
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    tensor, transform = letterbox_rgb(cv2.merge([gray, clahe, edge]), size=640)
    return stitched, tensor, transform
