import math

import numpy as np


def _compensation_axes(settings):
    try:
        center_x, center_y = map(float, settings['center'])
        span_x, span_y = map(float, settings['span'])
        left, right, top, bottom = (float(settings[key]) for key in
                                    ('x_left', 'x_right', 'y_top', 'y_bottom'))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid tray position compensation') from error
    if (not all(math.isfinite(value) for value in (center_x, center_y, span_x, span_y,
                                                  left, right, top, bottom))
            or span_x <= 0 or span_y <= 0):
        raise ValueError('tray position compensation requires finite values and positive spans')
    return (center_x, span_x, left, right), (center_y, span_y, top, bottom)


def _axis_correction(position, center, span, lower_side, upper_side):
    distance = position - center
    factor = min(1.0, abs(distance) / span)
    return factor * (lower_side if distance < 0 else upper_side)


class TrayLocator:
    """Select the leftmost part and turn its right edge into a push target.

    Optional compensation uses millimetres throughout. The Windows settings
    expose the center, span and each side's endpoint adjustment. Since the
    compiled calculation is unavailable, the Linux profile uses a linear ramp
    from zero at center to the configured adjustment at one span and beyond.
    The ramp uses the calibrated edge before the fixed tool offsets are added.
    With skip_unsafe, only candidates with safe centers and corrected targets
    enter left-first selection.
    """

    def __init__(self, calibration, detector, roi, offsets=(4,0), confidence=0.35, *,
                 compensation=None, skip_unsafe=False):
        self.calibration, self.detector = calibration, detector
        self.roi, self.offsets, self.confidence = roi, offsets, confidence
        self._compensation = _compensation_axes(compensation) if compensation is not None else None
        self.skip_unsafe = skip_unsafe

    def locate(self, frame):
        x,y,w,h = self.roi
        if any(int(v) != v for v in self.roi) or x < 0 or y < 0 or w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            raise ValueError('invalid ROI')
        candidates = []
        for detection in self.detector.detect(frame[y:y+h,x:x+w]):
            x1,y1,x2,y2,score = map(float,detection)
            if not np.isfinite([x1,y1,x2,y2,score]).all() or score < self.confidence:
                continue
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                continue
            midpoint_y = y + (y1 + y2) / 2
            edge = (x + x2, midpoint_y)
            try:
                center = self.calibration.map_pixel(x + (x1 + x2) / 2, midpoint_y)
                target = self._push_target(edge) if self.skip_unsafe else edge
            except ValueError:
                if self.skip_unsafe:
                    continue
                raise
            candidates.append((center, target))
        if not candidates:
            return None
        _, target = min(candidates, key=lambda item: item[0])
        return target if self.skip_unsafe else self._push_target(target)

    def _push_target(self, pixel):
        edge_x, edge_y = self.calibration.map_pixel(*pixel)
        if self._compensation is not None:
            x_axis, y_axis = self._compensation
            edge_x += _axis_correction(edge_x, *x_axis)
            edge_y += _axis_correction(edge_y, *y_axis)
        result = (edge_x + self.offsets[0], edge_y + self.offsets[1])
        self.calibration.validate_physical(*result)
        return result


class TrayDetector:
    """Decode one-class tray detections from a YOLO backend on an ROI."""
    def __init__(self, backend, confidence=0.35, iou=0.5, *, feature_input=False):
        self.backend, self.confidence, self.iou = backend, confidence, iou
        self.feature_input = feature_input

    def detect(self, frame):
        from vision import letterbox_rgb, decode_yolo_raw
        if self.feature_input:
            from feature_preprocess import feature_bgr
            frame = feature_bgr(frame)
        tensor, transform = letterbox_rgb(frame, size=640)
        detections = decode_yolo_raw(self.backend.infer(tensor), transform, ['object'],
                                     self.confidence, self.iou, expected_count=1)
        return [tuple(d.box) + (d.confidence,) for d in detections]
