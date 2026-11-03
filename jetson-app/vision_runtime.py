"""Offline orchestration for the detection and surface-mask models."""

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from trt_backend import ThreadOwnershipError, create_backend
from vision import (
    classify_hsv,
    decode_yolo_raw,
    letterbox_rgb,
    load_class_names,
    restore_surface_mask,
    tighten_box,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ViewRecognitionResult:
    """Display-ready result for one camera view."""

    detected: bool
    class_id: Optional[int]
    raw_class_name: Optional[str]
    class_name: Optional[str]
    detection_confidence: Optional[float]
    box: Optional[Tuple[float, float, float, float]]
    color_name: str
    color_confidence: float
    annotated_frame: np.ndarray


@dataclass(frozen=True)
class RecognitionResult:
    """Consensus result across one or two camera views."""

    status: str
    final_class_name: Optional[str]
    final_color_name: str
    detection_confidence: Optional[float]
    views: Tuple[ViewRecognitionResult, ...]


class OfflineRecognizer:
    """Lazily own two native inference backends on their calling thread."""

    def __init__(
        self,
        yolo_engine_path=None,
        mask_engine_path=None,
        class_names_path=None,
        config_path=None,
        backend_factory: Optional[Callable] = None,
        device: int = 0,
    ):
        self.yolo_engine_path = Path(
            yolo_engine_path
            if yolo_engine_path is not None
            else ROOT / "models" / "yolo11n_23cls_raw_fp16.engine"
        )
        self.mask_engine_path = Path(
            mask_engine_path
            if mask_engine_path is not None
            else ROOT / "models" / "surface_mask_fp16.engine"
        )
        self.class_names_path = Path(
            class_names_path
            if class_names_path is not None
            else ROOT / "models" / "classes.txt"
        )
        self.config_path = Path(
            config_path if config_path is not None else ROOT / "vision_config.yaml"
        )
        for label, path in (
            ("YOLO engine", self.yolo_engine_path),
            ("surface-mask engine", self.mask_engine_path),
            ("class names", self.class_names_path),
            ("vision config", self.config_path),
        ):
            if not path.is_file():
                raise FileNotFoundError("%s file not found: %s" % (label, path))

        try:
            with self.config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                "could not read vision config '%s': %s" % (self.config_path, exc)
            ) from exc
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            raise ValueError("vision config must contain a 'model' mapping")

        self._validate_config(config)

        self.config = config
        self.class_names = load_class_names(self.class_names_path)
        self._backend_factory = backend_factory or create_backend
        self._device = int(device)
        self._lifecycle_lock = threading.Lock()
        self._yolo_backend = None
        self._mask_backend = None
        self._owner_thread = None
        self._closed = False

    @staticmethod
    def _validate_config(config):
        model = config["model"]
        required_model_keys = (
            "input_size",
            "class_count",
            "confidence_threshold",
            "iou_threshold",
            "mask_threshold",
        )
        for key in required_model_keys:
            if key not in model:
                raise ValueError("vision config model is missing '%s'" % key)
        if model["input_size"] != 320:
            raise ValueError("vision config model.input_size must be 320")
        if model["class_count"] != 23:
            raise ValueError("vision config model.class_count must be 23")
        for key in ("confidence_threshold", "iou_threshold", "mask_threshold"):
            value = model[key]
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("vision config model.%s must be in [0, 1]" % key)

        recognition = config.get("recognition")
        if not isinstance(recognition, dict):
            raise ValueError("vision config must contain a 'recognition' mapping")
        if (
            not isinstance(recognition.get("unknown_color"), str)
            or not recognition["unknown_color"]
        ):
            raise ValueError("vision config recognition.unknown_color must be non-empty")
        minimum_confidence = recognition.get("min_color_confidence")
        if (
            not isinstance(minimum_confidence, (int, float))
            or not 0.0 <= float(minimum_confidence) <= 1.0
        ):
            raise ValueError(
                "vision config recognition.min_color_confidence must be in [0, 1]"
            )

        color = config.get("color")
        if not isinstance(color, dict):
            raise ValueError("vision config must contain a 'color' mapping")
        for key in (
            "hsv_ranges",
            "min_valid_pixels",
            "black_value_max",
            "white_value_min",
            "achromatic_saturation_max",
        ):
            if key not in color:
                raise ValueError("vision config color is missing '%s'" % key)
        if not isinstance(color["hsv_ranges"], dict) or not color["hsv_ranges"]:
            raise ValueError("vision config color.hsv_ranges must be a non-empty mapping")
        if (
            not isinstance(color["min_valid_pixels"], int)
            or isinstance(color["min_valid_pixels"], bool)
            or color["min_valid_pixels"] < 1
        ):
            raise ValueError("vision config color.min_valid_pixels must be positive")
        for key in (
            "black_value_max",
            "white_value_min",
            "achromatic_saturation_max",
        ):
            value = color[key]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 255
            ):
                raise ValueError("vision config color.%s must be in [0, 255]" % key)
        if color["black_value_max"] >= color["white_value_min"]:
            raise ValueError(
                "vision config color.black_value_max must be below white_value_min"
            )
        for color_name, ranges in color["hsv_ranges"].items():
            if not isinstance(color_name, str) or not color_name.strip():
                raise ValueError("vision config HSV color names must be non-empty")
            if not isinstance(ranges, (list, tuple)) or not ranges:
                raise ValueError(
                    "vision config HSV ranges for '%s' must be a non-empty list"
                    % color_name
                )
            for index, hsv_range in enumerate(ranges):
                if not isinstance(hsv_range, dict):
                    raise ValueError(
                        "vision config HSV range %s[%d] must be a mapping"
                        % (color_name, index)
                    )
                bounds = {}
                for bound_name in ("lower", "upper"):
                    if bound_name not in hsv_range:
                        raise ValueError(
                            "vision config HSV range %s[%d] is missing '%s'"
                            % (color_name, index, bound_name)
                        )
                    bound = hsv_range[bound_name]
                    if not isinstance(bound, (list, tuple)) or len(bound) != 3:
                        raise ValueError(
                            "vision config HSV %s bound must contain three channels"
                            % bound_name
                        )
                    limits = (180, 255, 255)
                    if any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or not 0 <= value <= limit
                        for value, limit in zip(bound, limits)
                    ):
                        raise ValueError(
                            "vision config HSV %s channels are out of range"
                            % bound_name
                        )
                    bounds[bound_name] = bound
                if any(
                    lower > upper
                    for lower, upper in zip(bounds["lower"], bounds["upper"])
                ):
                    raise ValueError(
                        "vision config HSV lower bound must not exceed upper bound"
                    )

        class_name_map = config.get("class_name_map")
        if not isinstance(class_name_map, dict):
            raise ValueError("vision config must contain a 'class_name_map' mapping")
        if any(
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(target, str)
            or not target.strip()
            for source, target in class_name_map.items()
        ):
            raise ValueError(
                "vision config class_name_map keys and values must be non-empty strings"
            )

    def _ensure_backends(self):
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("offline recognizer is closed")
            if self._yolo_backend is not None:
                if threading.get_ident() != self._owner_thread:
                    raise ThreadOwnershipError(
                        "offline recognizer must be used from its owner thread"
                    )
                return
            yolo_backend = self._backend_factory(
                self.yolo_engine_path, "yolo", device=self._device
            )
            try:
                mask_backend = self._backend_factory(
                    self.mask_engine_path, "mask", device=self._device
                )
            except BaseException:
                try:
                    yolo_backend.close()
                except BaseException:
                    pass
                raise
            self._yolo_backend = yolo_backend
            self._mask_backend = mask_backend
            self._owner_thread = threading.get_ident()

    def recognize_view(self, frame_bgr: np.ndarray) -> ViewRecognitionResult:
        model = self.config["model"]
        tensor, transform = letterbox_rgb(frame_bgr, size=int(model["input_size"]))
        self._ensure_backends()
        yolo_output = self._yolo_backend.infer(tensor)
        mask_output = self._mask_backend.infer(tensor)
        if (
            not isinstance(yolo_output, np.ndarray)
            or yolo_output.ndim != 3
            or yolo_output.shape[0] != 1
            or yolo_output.shape[1] != 4 + len(self.class_names)
        ):
            shape = getattr(yolo_output, "shape", None)
            raise ValueError(
                "YOLO output shape %r must be [1, %d, N]"
                % (shape, 4 + len(self.class_names))
            )
        expected_mask_shape = (
            1,
            1,
            int(model["input_size"]),
            int(model["input_size"]),
        )
        if (
            not isinstance(mask_output, np.ndarray)
            or mask_output.shape != expected_mask_shape
        ):
            shape = getattr(mask_output, "shape", None)
            raise ValueError(
                "mask output shape %r must be %r" % (shape, expected_mask_shape)
            )
        detections = decode_yolo_raw(
            yolo_output,
            transform,
            self.class_names,
            confidence_threshold=float(model["confidence_threshold"]),
            iou_threshold=float(model["iou_threshold"]),
        )
        mask = restore_surface_mask(
            mask_output, transform, threshold=float(model["mask_threshold"])
        )
        unknown_color = str(
            self.config.get("recognition", {}).get("unknown_color", "未知颜色")
        )
        if detections:
            detection = detections[0]
            box = tighten_box(detection.box, mask)
            color = classify_hsv(frame_bgr, mask, box, self.config)
            mapped_name = str(
                self.config.get("class_name_map", {}).get(
                    detection.class_name, detection.class_name
                )
            )
            annotated = self._annotate(frame_bgr, mask, detection, box)
            return ViewRecognitionResult(
                detected=True,
                class_id=detection.class_id,
                raw_class_name=detection.class_name,
                class_name=mapped_name,
                detection_confidence=detection.confidence,
                box=box,
                color_name=color.name,
                color_confidence=color.confidence,
                annotated_frame=annotated,
            )
        annotated = self._annotate(frame_bgr, mask, None, None)
        return ViewRecognitionResult(
            detected=False,
            class_id=None,
            raw_class_name=None,
            class_name=None,
            detection_confidence=None,
            box=None,
            color_name=unknown_color,
            color_confidence=0.0,
            annotated_frame=annotated,
        )

    @staticmethod
    def _annotate(frame_bgr, mask, detection, box):
        annotated = np.array(frame_bgr, copy=True, order="C")
        if np.any(mask):
            overlay = annotated.copy()
            overlay[mask] = (0, 180, 255)
            cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0.0, dst=annotated)
        if detection is not None and box is not None:
            left, top, right, bottom = [int(round(float(value))) for value in box]
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
            label = ("%s %.2f" % (detection.class_name, detection.confidence))
            label = label.encode("ascii", "replace").decode("ascii")
            cv2.putText(
                annotated,
                label,
                (max(0, left), max(12, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def recognize_views(self, frames: Sequence[np.ndarray]) -> RecognitionResult:
        if isinstance(frames, np.ndarray) or not isinstance(frames, Sequence):
            raise ValueError("frames must be a sequence containing one or two frames")
        if len(frames) not in (1, 2):
            raise ValueError("frames must contain exactly one or two frames")
        views = tuple(self.recognize_view(frame) for frame in frames)
        detected = tuple(view for view in views if view.detected)
        unknown_color = str(
            self.config.get("recognition", {}).get("unknown_color", "未知颜色")
        )
        if not detected:
            return RecognitionResult(
                status="未识别到物体",
                final_class_name=None,
                final_color_name=unknown_color,
                detection_confidence=None,
                views=views,
            )
        if len(detected) == 1:
            view = detected[0]
            return RecognitionResult(
                status="单路识别",
                final_class_name=view.class_name,
                final_color_name=view.color_name,
                detection_confidence=view.detection_confidence,
                views=views,
            )

        first, second = detected
        first_confidence = float(first.detection_confidence)
        second_confidence = float(second.detection_confidence)
        mean_confidence = (first_confidence + second_confidence) / 2.0
        class_conflict = first.class_id != second.class_id
        if first.color_name == unknown_color:
            final_color = second.color_name
            color_conflict = False
        elif second.color_name == unknown_color:
            final_color = first.color_name
            color_conflict = False
        elif first.color_name == second.color_name:
            final_color = first.color_name
            color_conflict = False
        else:
            final_color = unknown_color
            color_conflict = True

        if class_conflict or color_conflict:
            return RecognitionResult(
                status="结果不确定",
                final_class_name=None,
                final_color_name=final_color,
                detection_confidence=mean_confidence,
                views=views,
            )
        return RecognitionResult(
            status="识别成功",
            final_class_name=first.class_name,
            final_color_name=final_color,
            detection_confidence=mean_confidence,
            views=views,
        )

    def close(self):
        with self._lifecycle_lock:
            if self._closed:
                return
            if (
                self._owner_thread is not None
                and threading.get_ident() != self._owner_thread
            ):
                raise ThreadOwnershipError(
                    "offline recognizer must be closed from its owner thread"
                )
            self._closed = True
            errors = []
            for backend in (self._mask_backend, self._yolo_backend):
                if backend is not None:
                    try:
                        backend.close()
                    except BaseException as exc:
                        errors.append(exc)
            self._mask_backend = None
            self._yolo_backend = None
            self._owner_thread = None
            if errors:
                raise errors[0]


__all__ = [
    "OfflineRecognizer",
    "RecognitionResult",
    "ViewRecognitionResult",
]
