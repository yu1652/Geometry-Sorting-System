"""Single-owner recognition thread with serialized model transactions."""
from pathlib import Path
from queue import Queue
from model_store import ModelStore

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal

ROOT = Path(__file__).resolve().parent


def load_model_set(releases):
    import yaml
    from legacy_vision import LegacyRecognizer
    from trt_backend import create_backend
    config = yaml.safe_load((ROOT / "legacy_vision.yaml").read_text(encoding="utf8"))
    names = (ROOT / "models/legacy_classes.txt").read_text(encoding="utf8").splitlines()
    detector = segmenter = None
    try:
        detector_path = releases["side_detector"].engine_path if "side_detector" in releases else ROOT / "models/shape_legacy_fp16.engine"
        detector = create_backend(detector_path, model_kind="yolo_legacy")
        if "surface_mask" in releases:
            segmenter = create_backend(releases["surface_mask"].engine_path, model_kind="mask")
        return LegacyRecognizer(detector, segmenter, names, config)
    except Exception:
        for resource in (segmenter, detector):
            if resource is not None:
                try: resource.close()
                except Exception: pass
        raise


def default_recognizer_factory():
    store = ModelStore(ROOT / "data/model_store")
    return load_model_set(store.resolve_active(store.read_active()["active"]))


class VisionWorker(QThread):
    completed = Signal(int, int, object)
    failed = Signal(int, int, str)
    model_trial_ready = Signal(object)
    model_changed = Signal(object)
    model_failed = Signal(str)

    def __init__(self, recognizer_factory=default_recognizer_factory, parent=None, store=None, loader=None):
        super().__init__(parent)
        self.factory = recognizer_factory
        self._store = store or ModelStore(ROOT / "data/model_store")
        self._loader = loader or load_model_set
        self._requests = Queue()
        self._preview = None

    def trial_candidate(self, manifest_path, slot, fixture_pair):
        self._requests.put(("candidate", str(manifest_path), slot, fixture_pair))

    activate_candidate = trial_candidate

    def confirm_candidate(self): self._requests.put(("confirm",))
    def cancel_candidate(self): self._requests.put(("cancel",))
    def rollback(self, slot, fixture_pair): self._requests.put(("rollback", slot, fixture_pair))
    def recognize(self, request_id, frame_revision, left_bgr, right_bgr, preprocessed=False):
        self._requests.put((request_id, frame_revision, left_bgr, right_bgr, preprocessed))
    def shutdown(self): self._requests.put(None)

    @staticmethod
    def _close(obj):
        if obj is not None:
            try: obj.close()
            except Exception: pass

    def _cancel(self):
        if self._preview is not None:
            self._close(self._preview[2])
            self._preview = None

    def _trial(self, active, fixture, mask_required=False):
        recognizer = self._loader(self._store.resolve_active(active))
        try:
            result = recognizer.recognize_pair(*fixture)
            if result.error or "error" in result.mask_status:
                raise ValueError("trial inference failed: " + str(result.error or result.mask_status))
            if mask_required and "applied" not in result.mask_status:
                raise ValueError("trial produced no applied surface mask")
            return recognizer, result
        except Exception:
            self._close(recognizer)
            raise

    def run(self):
        recognizer = None
        try:
            while True:
                request = self._requests.get()
                if request is None: break
                kind = request[0]
                if kind in ("candidate", "confirm", "cancel", "rollback"):
                    try:
                        if kind == "cancel":
                            self._cancel()
                        elif kind == "candidate":
                            self._cancel()
                            _, path, slot, fixture = request
                            release = self._store.validate_manifest(path, slot)
                            staged = self._store.install(release)
                            release = self._store.validate_manifest(staged, slot)
                            state = self._store.read_active()
                            active = dict(state["active"])
                            active[slot] = {"manifest_path": str(release.manifest_path), "version": release.version}
                            candidate, result = self._trial(active, fixture, slot == "surface_mask")
                            history = dict(state["previous"])
                            history[slot] = state["active"].get(slot)
                            self._preview = (active, history, candidate, result)
                            self.model_trial_ready.emit({"slot": slot, "version": release.version, "result": result})
                        elif kind == "confirm":
                            if self._preview is None: raise ValueError("没有待确认的模型候选")
                            active, previous, candidate, result = self._preview
                            self._store.commit_active(active, previous)
                            old, recognizer = recognizer, candidate
                            self._preview = None
                            self._close(old)
                            self.model_changed.emit(active)
                        else:
                            self._cancel()
                            state = self._store.read_active()
                            slot = request[1]
                            if slot not in self._store.SLOTS or slot not in state["previous"]:
                                raise ValueError("没有可回退的模型版本")
                            prior = dict(state["active"])
                            record = state["previous"][slot]
                            if record is None: prior.pop(slot, None)
                            else: prior[slot] = record
                            history = dict(state["previous"])
                            history[slot] = state["active"].get(slot)
                            candidate, result = self._trial(prior, request[2], slot == "surface_mask" and record is not None)
                            try: self._store.commit_active(prior, history)
                            except Exception:
                                self._close(candidate)
                                raise
                            old, recognizer = recognizer, candidate
                            self._close(old)
                            self.model_changed.emit(prior)
                    except Exception as exc:
                        if kind == "confirm": self._cancel()
                        self.model_failed.emit(str(exc) or type(exc).__name__)
                    continue
                request_id, revision, left, right, preprocessed = request
                try:
                    if recognizer is None: recognizer = self.factory()
                    if preprocessed:
                        result = recognizer.recognize_pair(left, right, preprocessed=True)
                    else:
                        result = recognizer.recognize_pair(left, right)
                    self.completed.emit(request_id, revision, result)
                except Exception as exc:
                    self.failed.emit(request_id, revision, str(exc) or type(exc).__name__)
        finally:
            self._cancel()
            self._close(recognizer)
