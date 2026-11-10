"""Offline recognition for the legacy dual-camera detector."""

import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, floor
from typing import Optional, Tuple

import cv2
import numpy as np

from feature_preprocess import feature_bgr
from vision import (
    ColorResult, Detection, LetterboxTransform, classify_hsv, color_sample_roi, decode_yolo_raw,
    letterbox_rgb, restore_surface_mask, tighten_box,
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
    tensor, transform = letterbox_rgb(feature_bgr(stitched), size=640)
    return stitched, tensor, transform


@dataclass(frozen=True)
class LegacyResult:
    detection: Optional[Detection]
    color: Optional[ColorResult]
    mask_status: Tuple[str, str]
    preview_bgr: np.ndarray
    elapsed_ms: float
    error: Optional[str] = None
    surface_boxes: tuple = (None, None)
    hole_counts: tuple = (None, None)
    color_sources: tuple = ('none', 'none')
    diagnostics: Optional[dict] = None


class LegacyRecognizer:
    """Own the supplied backends on the thread that constructs this recognizer."""

    def __init__(self, detector, segmenter, class_names, config, *, expected_count=26, capture_color_sample=False,
                 per_view_features=False):
        self._detector = detector
        self._segmenter = segmenter
        self._class_names = tuple(class_names)
        self._config = config
        self._expected_count = expected_count
        self._capture_color_sample = capture_color_sample
        self._per_view_features = per_view_features
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
        diagnostics = {
            'pair_bgr': stitched.copy(),
            # Save the exact prepared detector input, without letterbox padding.
            'features_bgr': np.rint(tensor[0, :, 160:480].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8),
            'preprocessed': bool(preprocessed),
            'class_names': list(self._class_names),
            'config': deepcopy(self._config),
            'models': {'detector': deepcopy(getattr(self._detector, 'provenance', None)),
                       'segmenter': deepcopy(getattr(self._segmenter, 'provenance', None))},
        }

        model = self._config.get("model", {})
        try:
            if self._per_view_features:
                if self._expected_count != 24 or len(self._class_names) != 24:
                    raise ValueError('per-view feature detector requires 24 classes')
                outputs = []
                detections = []
                half_transform = LetterboxTransform(320, 320, 320, 1, 0, 0, 320, 320)
                for side in range(2):
                    half_tensor = np.ascontiguousarray(tensor[:, :, 160:480, side * 320:(side + 1) * 320])
                    raw_output = self._detector.infer(half_tensor)
                    if raw_output.shape != (1, 28, 2100):
                        raise ValueError('per-view detector output must be [1,28,2100]')
                    outputs.append(np.array(raw_output, copy=True))
                    local = decode_yolo_raw(raw_output, half_transform, self._class_names,
                                            model.get('confidence_threshold', 0.45),
                                            model.get('iou_threshold', 0.50), expected_count=24)
                    detections.extend(Detection(item.class_id, item.class_name, item.confidence,
                                                (item.box[0] + side * 320, item.box[1],
                                                 item.box[2] + side * 320, item.box[3])) for item in local)
                detections.sort(key=lambda item: -item.confidence)
                diagnostics['raw_output'] = np.stack(outputs)
            else:
                raw_output = self._detector.infer(tensor)
                diagnostics['raw_output'] = np.array(raw_output, copy=True)
                detections = decode_yolo_raw(
                    raw_output, transform, self._class_names,
                    model.get("confidence_threshold", 0.45),
                    model.get("iou_threshold", 0.50), expected_count=self._expected_count)
        except Exception as exc:
            return LegacyResult(
                None, None, ("detection_error", "detection_error"), preview,
                (time.perf_counter() - started) * 1000, str(exc) or type(exc).__name__,
                diagnostics=diagnostics,
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
        surface_boxes, hole_counts, color_sources = [], [], []
        full_roi = np.ones((320, 320), dtype=bool)
        mask_mode = self._config.get('surface_mask', {}).get('input_mode', 'legacy_features')
        for side, candidates in enumerate(sides):
            offset = side * 320
            half = stitched[:, offset:offset + 320]
            target = preview[:, offset:offset + 320]
            # Feature halves are the exact RGB tensors used in training. Slice
            # the already prepared stitch, avoiding a second CLAHE/Canny pass.
            mask_tensor = tensor[:, :, 160:480, offset:offset + 320].copy()
            if mask_mode == 'raw_rgb' and not preprocessed:
                mask_tensor, _ = letterbox_rgb(half, 320)
            if mask_mode not in ('raw_rgb', 'legacy_features') or mask_mode == 'raw_rgb' and preprocessed:
                mask, status = None, 'input_mismatch'
            else:
                mask, status = self._surface(mask_tensor, candidates)
            statuses.append(status)
            box = candidates[0][1] if candidates else None
            surface_box = tighten_box(box, mask) if mask is not None else None
            surface_boxes.append(surface_box)
            hole_counts.append(count_inner_holes(mask) if mask is not None else None)
            if mask is not None:
                pixels = target[mask].astype(np.float32)
                target[mask] = np.rint(pixels * .65 + np.array((0, 255, 0)) * .35).astype(np.uint8)
            source = 'none'
            if (candidates and not preprocessed
                    and (not self._per_view_features or candidates[0][0].class_id == detections[0].class_id)):
                detection = candidates[0][0]
                source = 'mask' if mask is not None else 'box'
                color_roi = surface_box or box
                foreground = mask if mask is not None else full_roi
                sample = None
                color_result = classify_hsv(half, foreground, color_roi, self._config)
                if self._capture_color_sample and color_result.valid_pixels:
                    x1, y1, x2, y2 = color_sample_roi(half, color_roi, self._config.get('color', self._config))
                    sample = half[y1:y2, x1:x2].copy()
                    sample[~foreground[y1:y2, x1:x2]] = 0
                colors.append((color_result, detection.confidence, sample))
            color_sources.append(source)

        color = None
        if colors:
            unknown = self._config.get("color", self._config).get(
                "unknown_color", self._config.get("recognition", {}).get("unknown_color", "\u672a\u77e5\u989c\u8272")
            )
            color, _, sample = max(colors, key=lambda item: (item[0].name != unknown, item[0].confidence, item[1]))
            if sample is not None:
                diagnostics['color_sample_bgr'] = sample

        for side, surface_box in enumerate(surface_boxes):
            if surface_box is not None:
                x1, y1, x2, y2 = surface_box
                cv2.rectangle(preview, (floor(x1) + side * 320, floor(y1)),
                              (ceil(x2) - 1 + side * 320, ceil(y2) - 1), (255, 220, 0), 1)
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
            surface_boxes=tuple(surface_boxes), hole_counts=tuple(hole_counts),
            color_sources=tuple(color_sources),
            diagnostics=diagnostics,
        )

    def _surface(self, tensor, candidates):
        if not candidates:
            return None, "no_detection"
        if self._segmenter is None:
            return None, self._config.get('surface_mask', {}).get('unavailable_status', 'disabled')
        if len(candidates) != 1:
            return None, "ambiguous"
        try:
            # Every mask slot consumes an unpadded 320x320 half-image.
            transform = LetterboxTransform(320, 320, 320, 1, 0, 0, 320, 320)
            mask = restore_surface_mask(self._segmenter.infer(tensor), transform, threshold=0.5)
        except Exception:
            return None, "error"
        if not mask.any():
            return None, "empty"
        if mask.mean() > .9:
            return None, "overfull"
        x1, y1, x2, y2 = candidates[0][1]
        # Select the connected foreground with the largest overlap with the
        # target, then clip to its detector box. No hole-filling is performed.
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        overlap = labels[floor(y1):ceil(y2), floor(x1):ceil(x2)]
        areas = np.bincount(overlap.ravel(), minlength=count)
        areas[0] = 0
        if not areas.any():
            return None, "unmatched"
        component = labels == int(areas.argmax())
        selected = np.zeros_like(mask)
        selected[floor(y1):ceil(y2), floor(x1):ceil(x2)] = component[floor(y1):ceil(y2), floor(x1):ceil(x2)]
        return selected, "applied"

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


def count_inner_holes(mask):
    """Visible enclosed background regions; diagnostic, not a class decision."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    minimum_area = max(8, int(mask.sum()) * .005)
    return sum(1 for contour, relation in zip(contours, hierarchy[0])
               if relation[3] >= 0 and cv2.contourArea(contour) >= minimum_area)


def surface_summary(result):
    names = {'applied': '已参与', 'no_detection': '无YOLO目标', 'disabled': '未开启',
             'missing_model': '模型缺失', 'ambiguous': '多目标未关联', 'empty': '空掩膜',
             'unmatched': '未匹配目标', 'error': '推理失败', 'overfull': '覆盖异常',
             'input_mismatch': '输入不匹配', 'feature_input': '仅特征输入'}
    statuses = getattr(result, 'mask_status', ('unknown', 'unknown'))
    holes = getattr(result, 'hole_counts', (None, None))
    parts = []
    for side, status, count in zip(('左', '右'), statuses, holes):
        text = '%s覆膜：%s' % (side, names.get(status, status))
        if count is not None:
            text += ' / 孔候选%d' % count
        parts.append(text)
    return '；'.join(parts)
