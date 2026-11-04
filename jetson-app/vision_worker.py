"""Single-owner recognition thread with a thread-safe request queue."""
from pathlib import Path
from queue import Queue

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal


def default_recognizer_factory():
    import yaml
    from legacy_vision import LegacyRecognizer
    from trt_backend import create_backend
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / "legacy_vision.yaml").read_text(encoding="utf8"))
    names = (root / "models/legacy_classes.txt").read_text(encoding="utf8").splitlines()
    detector = create_backend(root / "models/shape_legacy_fp16.engine", model_kind="yolo_legacy")
    # Baseline mask intentionally remains disabled pending new-scene validation.
    return LegacyRecognizer(detector, None, names, config)


class VisionWorker(QThread):
    completed = Signal(int, int, object)
    failed = Signal(int, int, str)

    def __init__(self, recognizer_factory, parent=None):
        super().__init__(parent)
        self.factory = recognizer_factory
        self._requests = Queue()

    def recognize(self, request_id, frame_revision, left_bgr, right_bgr):
        self._requests.put((request_id, frame_revision, left_bgr, right_bgr))

    def shutdown(self):
        self._requests.put(None)

    def run(self):
        recognizer = None
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    break
                request_id, revision, left, right = request
                try:
                    if recognizer is None:
                        recognizer = self.factory()
                    self.completed.emit(request_id, revision, recognizer.recognize_pair(left, right))
                except Exception as exc:
                    self.failed.emit(request_id, revision, str(exc) or type(exc).__name__)
        finally:
            if recognizer is not None:
                try:
                    recognizer.close()
                except Exception as exc:
                    self.failed.emit(-1, -1, str(exc) or type(exc).__name__)
