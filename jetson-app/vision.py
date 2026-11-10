"""Pure, offline vision post-processing helpers.

This module deliberately has no GUI, inference-engine, camera, serial, or
hardware dependencies so its geometry and colour behaviour can be tested in
isolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int

    @property
    def scale_x(self) -> float:
        """Actual horizontal resize ratio after integer rounding."""
        return float(self.resized_width) / self.original_width

    @property
    def scale_y(self) -> float:
        """Actual vertical resize ratio after integer rounding."""
        return float(self.resized_height) / self.original_height


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ColorResult:
    name: str
    confidence: float
    valid_pixels: int


def letterbox_rgb(frame_bgr: np.ndarray, size: int = 320) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize a BGR frame into a square RGB NCHW float tensor."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a non-empty HxWx3 array")
    if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
        raise ValueError("frame_bgr must be non-empty")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")

    height, width = frame_bgr.shape[:2]
    scale = min(float(size) / width, float(size) / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        size - resized_height - pad_y,
        pad_x,
        size - resized_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    transform = LetterboxTransform(
        original_width=width,
        original_height=height,
        input_size=size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )
    return tensor, transform


def decode_yolo_raw(
    output: np.ndarray,
    transform: LetterboxTransform,
    class_names: Sequence[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    *,
    expected_count: int = 23,
) -> List[Detection]:
    """Decode YOLO11 [1, 4 + nc, N] output and run class-aware NMS."""
    if expected_count not in (1, 8, 23, 24, 26):
        raise ValueError("expected_count must be 1, 8, 23, 24 or 26")
    if len(class_names) != expected_count:
        raise ValueError("class_names must contain exactly %d entries" % expected_count)
    if not isinstance(output, np.ndarray) or output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("output must have shape [1, 4+nc, N]")
    if output.shape[1] != 4 + len(class_names):
        raise ValueError("output class dimension does not match class_names")
    if not np.isfinite(output).all():
        raise ValueError("output must contain only finite values")
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("thresholds must be in [0, 1]")

    candidates = []
    for index in range(output.shape[2]):
        scores = output[0, 4:, index]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < confidence_threshold:
            continue
        center_x, center_y, width, height = [float(value) for value in output[0, :4, index]]
        if width <= 0.0 or height <= 0.0:
            continue
        x1 = (center_x - width / 2.0 - transform.pad_x) / transform.scale_x
        y1 = (center_y - height / 2.0 - transform.pad_y) / transform.scale_y
        x2 = (center_x + width / 2.0 - transform.pad_x) / transform.scale_x
        y2 = (center_y + height / 2.0 - transform.pad_y) / transform.scale_y
        box = (
            max(0.0, min(float(transform.original_width), x1)),
            max(0.0, min(float(transform.original_height), y1)),
            max(0.0, min(float(transform.original_width), x2)),
            max(0.0, min(float(transform.original_height), y2)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates.append((class_id, confidence, box, index))

    kept = []
    for class_id in range(len(class_names)):
        class_candidates = [item for item in candidates if item[0] == class_id]
        class_candidates.sort(key=lambda item: (-item[1], item[3]))
        while class_candidates:
            current = class_candidates.pop(0)
            kept.append(current)
            class_candidates = [
                candidate
                for candidate in class_candidates
                if _box_iou(current[2], candidate[2]) <= iou_threshold
            ]

    kept.sort(key=lambda item: (-item[1], item[3]))
    return [
        Detection(class_id=item[0], class_name=str(class_names[item[0]]), confidence=item[1], box=item[2])
        for item in kept
    ]


def load_class_names(path, expected_count: int = 23) -> List[str]:
    """Load class names, allowing a training reserve tail for the 23-class model."""
    if expected_count not in (1, 8, 23, 24, 26):
        raise ValueError("expected_count must be 1, 8, 23, 24 or 26")
    try:
        entries = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("could not read class names") from exc
    entries = [entry.strip() for entry in entries]
    if len(entries) < expected_count or any(not entry for entry in entries):
        raise ValueError("class file must contain %d non-empty runtime names" % expected_count)

    runtime_names = entries[:expected_count]
    allowed_reserved = {"reserved_23", "reserved_24", "reserved_25"}
    if any(name in allowed_reserved for name in runtime_names) or (
        expected_count == 26 and any(name.startswith("reserved_") for name in runtime_names)
    ):
        raise ValueError("reserved entries are not runtime classes")
    if len(set(runtime_names)) != expected_count:
        raise ValueError("runtime class names must be unique")
    reserved_tail = entries[expected_count:]
    if expected_count == 26 and reserved_tail:
        raise ValueError("class file must contain exactly 26 entries")
    if reserved_tail and reserved_tail != ["reserved_23", "reserved_24", "reserved_25"]:
        raise ValueError("reserved entries must be the complete ordered tail")
    return runtime_names


def restore_surface_mask(logits: np.ndarray, transform: LetterboxTransform, threshold: float = 0.5) -> np.ndarray:
    """Undo letterboxing for a single-channel segmentation-logit tensor."""
    expected_shape = (1, 1, transform.input_size, transform.input_size)
    if not isinstance(logits, np.ndarray) or logits.shape != expected_shape:
        raise ValueError("logits must have shape [1, 1, input_size, input_size]")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -88.0, 88.0)))
    content = probabilities[
        transform.pad_y:transform.pad_y + transform.resized_height,
        transform.pad_x:transform.pad_x + transform.resized_width,
    ]
    resized = cv2.resize(
        (content > threshold).astype(np.uint8),
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def tighten_box(box, mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Tighten an xyxy box to its foreground pixels, retaining an empty fallback."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    height, width = mask.shape
    x1 = max(0, min(width, int(np.floor(float(box[0])))))
    y1 = max(0, min(height, int(np.floor(float(box[1])))))
    x2 = max(0, min(width, int(np.ceil(float(box[2])))))
    y2 = max(0, min(height, int(np.ceil(float(box[3])))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    foreground_y, foreground_x = np.nonzero(mask[y1:y2, x1:x2])
    if foreground_x.size == 0:
        return tuple(box)
    return (
        int(x1 + foreground_x.min()),
        int(y1 + foreground_y.min()),
        int(x1 + foreground_x.max() + 1),
        int(y1 + foreground_y.max() + 1),
    )


def color_sample_roi(frame_bgr, box, color_config):
    x1, y1, x2, y2 = _safe_box_roi(box, frame_bgr.shape[1], frame_bgr.shape[0])
    if x1 >= x2 or y1 >= y2:
        return x1, y1, x2, y2
    center_ratio = float(color_config.get('bbox_center_ratio', 1.0))
    if not 0 < center_ratio <= 1:
        raise ValueError('bbox_center_ratio must be in (0, 1]')
    inset_x = int((x2 - x1) * (1 - center_ratio) / 2)
    inset_y = int((y2 - y1) * (1 - center_ratio) / 2)
    return x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y


def classify_hsv(frame_bgr: np.ndarray, mask: np.ndarray, box, config) -> ColorResult:
    """Classify foreground pixels in a detection box using configured HSV ranges."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 array")
    if not isinstance(mask, np.ndarray) or mask.shape != frame_bgr.shape[:2]:
        raise ValueError("mask must match frame dimensions")
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    color_config = dict(config.get("color", config))
    recognition_config = config.get("recognition", {})
    if isinstance(recognition_config, dict):
        for key in ("unknown_color", "min_color_confidence"):
            if key not in color_config and key in recognition_config:
                color_config[key] = recognition_config[key]
    unknown_name = color_config.get("unknown_color", "未知颜色")
    x1, y1, x2, y2 = color_sample_roi(frame_bgr, box, color_config)
    if x1 >= x2 or y1 >= y2:
        return ColorResult(unknown_name, 0.0, 0)
    selected = mask[y1:y2, x1:x2].astype(bool)
    valid_pixels = int(selected.sum())
    minimum = int(color_config["min_valid_pixels"])
    if valid_pixels < minimum:
        return ColorResult(unknown_name, 0.0, valid_pixels)

    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    pixels = hsv[selected]
    saturation = pixels[:, 1]
    value = pixels[:, 2]
    achromatic = saturation <= int(color_config["achromatic_saturation_max"])
    counts = {
        "黑色": int(np.count_nonzero(achromatic & (value <= int(color_config["black_value_max"])))),
        "白色": int(np.count_nonzero(achromatic & (value >= int(color_config["white_value_min"])) )),
        "灰色": int(np.count_nonzero(
            achromatic
            & (value > int(color_config["black_value_max"]))
            & (value < int(color_config["white_value_min"]))
        )),
    }
    chromatic = (~achromatic
                 & (saturation >= int(color_config.get('valid_min_saturation', 0)))
                 & (value >= int(color_config.get('valid_min_value', 0)))
                 & (value <= int(color_config.get('valid_max_value', 255))))
    for name, ranges in color_config["hsv_ranges"].items():
        matches = np.zeros(valid_pixels, dtype=bool)
        for hsv_range in ranges:
            lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
            upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
            matches |= np.all((pixels >= lower) & (pixels <= upper), axis=1)
        counts[name] = int(np.count_nonzero(chromatic & matches))

    name, count = max(counts.items(), key=lambda item: item[1])
    confidence = float(count) / valid_pixels
    if confidence < float(color_config["min_color_confidence"]):
        return ColorResult(unknown_name, confidence, valid_pixels)
    return ColorResult(name, confidence, valid_pixels)


def _box_iou(first, second) -> float:
    intersection_left = max(first[0], second[0])
    intersection_top = max(first[1], second[1])
    intersection_right = min(first[2], second[2])
    intersection_bottom = min(first[3], second[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(0.0, intersection_bottom - intersection_top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _safe_box_roi(box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Return a sorted, clipped integer ROI suitable for NumPy slicing."""
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    try:
        left, right = sorted((float(box[0]), float(box[2])))
        top, bottom = sorted((float(box[1]), float(box[3])))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("box coordinates must be numeric") from exc
    if not np.isfinite((left, top, right, bottom)).all():
        raise ValueError("box coordinates must be finite")
    if left >= right or top >= bottom:
        return (0, 0, 0, 0)
    return (
        max(0, min(width, int(np.floor(left)))),
        max(0, min(height, int(np.floor(top)))),
        max(0, min(width, int(np.ceil(right)))),
        max(0, min(height, int(np.ceil(bottom)))),
    )
