import numpy as np

class TrayLocator:
    def __init__(self, calibration, detector, roi, offsets=(4,0), confidence=0.35):
        self.calibration, self.detector = calibration, detector
        self.roi, self.offsets, self.confidence = roi, offsets, confidence

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
            center = self.calibration.map_pixel(x+(x1+x2)/2,y+(y1+y2)/2)
            candidates.append((center, (x+x2,y+(y1+y2)/2)))
        if not candidates: return None
        _, pixel = min(candidates, key=lambda item: item[0])
        position = self.calibration.map_pixel(*pixel)
        result = (position[0]+self.offsets[0], position[1]+self.offsets[1])
        self.calibration.validate_physical(*result)
        return result
class TrayDetector:
    """Decode one-class tray detections from a YOLO backend on an ROI."""
    def __init__(self, backend, confidence=0.35, iou=0.5):
        self.backend, self.confidence, self.iou = backend, confidence, iou

    def detect(self, frame):
        from vision import letterbox_rgb, decode_yolo_raw
        tensor, transform = letterbox_rgb(frame, size=640)
        detections = decode_yolo_raw(self.backend.infer(tensor), transform, ['object'],
                                     self.confidence, self.iou, expected_count=1)
        return [tuple(d.box) + (d.confidence,) for d in detections]
