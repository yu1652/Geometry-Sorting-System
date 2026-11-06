"""Three physical cameras and TensorRT models owned by one worker thread."""
import math
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event

# Load Qt before OpenCV to avoid Jetson's libGLdispatch static TLS conflict.
from vision_worker import QThread, Signal, default_recognizer_factory

import cv2
import numpy as np

from camera_service import CAMERA_ROLES, CameraError, CameraService
from tray_locator import TrayDetector, TrayLocator


ROOT = Path(__file__).resolve().parent


def open_live_capture(path, report=None):
    """Request a bounded-bandwidth format from the Linux V4L2 driver."""
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            raise CameraError('Cannot open camera ' + str(path))
        mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        settings = (
            (cv2.CAP_PROP_FOURCC, mjpg),
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        )
        for property_id, value in settings:
            if not capture.set(property_id, value):
                raise CameraError('Camera rejected setting {}={} on {}'.format(property_id, value, path))
        # Frame-rate controls are optional on UVC cameras. MJPG and actual
        # frame dimensions remain mandatory, even when the FPS hint is ignored.
        try:
            capture.set(cv2.CAP_PROP_FPS, 15)
        except cv2.error:
            pass
        if round(capture.get(cv2.CAP_PROP_FOURCC)) != mjpg:
            raise CameraError('Camera did not negotiate MJPG: ' + str(path))
        try:
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
        except cv2.error:
            actual_fps = float('nan')
        fps_text = format(actual_fps, 'g') if np.isfinite(actual_fps) and actual_fps > 0 else '未知'
        if report is not None:
            report('相机 %s：MJPG 640×480，驱动报告 %s FPS（请求 15 FPS）' % (path, fps_text))
        # This driver hint is optional. Jobs also drain fresh frame batches.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return capture
    except Exception:
        capture.release()
        raise


def _default_tray_factory():
    from trt_backend import create_backend
    return create_backend(ROOT / 'models/tray_object_detector_fp16.engine', model_kind='tray')


class LiveVisionWorker(QThread):
    """Raw-frame preview and fresh-frame jobs, without any motion authority.

    `left` is the recognition overhead view, `right` the recognition side view,
    and `top` the independent tray view. `tray_factory` returns an owned backend
    with infer/close methods. Job methods return False when busy or shutting down.
    """

    frames_ready = Signal(object)
    recognition_ready = Signal(int, object)
    location_ready = Signal(int, object)
    empty_reference_ready = Signal(int, object)
    failed = Signal(int, str)
    notice = Signal(str)

    def __init__(self, sources, parent=None, capture_factory=None,
                 recognizer_factory=default_recognizer_factory, tray_factory=None,
                 scan_settle_seconds=0.8, drain_batches=4, clock=time.monotonic):
        super().__init__(parent)
        if set(sources) != set(CAMERA_ROLES):
            raise ValueError('All three physical camera roles must be configured')
        self.sources = {role: str(sources[role]) for role in CAMERA_ROLES}
        if len(set(self.sources.values())) != 3:
            raise ValueError('Camera roles must use three distinct devices')
        if (isinstance(scan_settle_seconds, bool)
                or not isinstance(scan_settle_seconds, (int, float))
                or not math.isfinite(scan_settle_seconds)
                or not 0 <= scan_settle_seconds <= 30):
            raise ValueError('Scan settling time must be finite and between 0 and 30 seconds')
        if (isinstance(drain_batches, bool) or not isinstance(drain_batches, int)
                or not 1 <= drain_batches <= 100):
            raise ValueError('Drain batches must be an integer between 1 and 100')
        if not callable(clock):
            raise ValueError('Capture clock must be callable')
        self.scan_settle_seconds = float(scan_settle_seconds)
        self.drain_batches = drain_batches
        self._clock = clock
        self.recognizer_factory = recognizer_factory or default_recognizer_factory
        self.tray_factory = tray_factory or _default_tray_factory
        factory = capture_factory or (lambda path: open_live_capture(path, self.notice.emit))
        self._cameras = CameraService(self.sources, factory)
        self._requests = Queue(maxsize=1)
        self._stopping = Event()

    def _submit(self, token, job):
        if self._stopping.is_set():
            self.failed.emit(token, 'Camera worker is shutting down')
            return False
        try:
            self._requests.put_nowait(job)
        except Full:
            self.failed.emit(token, 'A vision request is already queued')
            return False
        return True

    def recognize(self, token, settle=False):
        return self._submit(token, ('recognize', token, bool(settle), self._clock()))

    def locate(self, token, calibration, roi, offsets):
        return self._submit(token, ('locate', token, calibration, tuple(roi), tuple(offsets)))

    def capture_empty(self, token):
        return self._submit(token, ('empty_reference', token))

    def shutdown(self):
        self._stopping.set()

    def _read_frames(self):
        frames = {}
        for role in CAMERA_ROLES:
            if self._stopping.is_set():
                return None
            frame = self._cameras.read(role)
            if frame is None:
                raise CameraError('No fresh frame from ' + role)
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise CameraError('{} requires uint8 640x480 BGR frames, got {}'.format(role, frame.shape))
            frames[role] = frame
        return frames

    def _fresh_frames(self, settle_seconds):
        """Drain actual captures after model loading; publish previews while settling."""
        started = self._clock()
        batches = 0
        while not self._stopping.is_set():
            frames = self._read_frames()
            if frames is None:
                return None
            self.frames_ready.emit(frames)
            batches += 1
            if self._stopping.is_set():
                return None
            if batches >= self.drain_batches and self._clock() - started >= settle_seconds:
                return frames
            # Native camera reads pace real devices. Bound CPU use for recordings
            # and fast capture backends, and let shutdown interrupt every wait.
            self._stopping.wait(0.005)
        return None

    def run(self):
        recognizer = tray_backend = None
        if self._stopping.is_set():
            return
        try:
            for role in CAMERA_ROLES:
                if self._stopping.is_set():
                    return
                self._cameras.open(role)
            for _ in range(3):
                frames = self._read_frames()
                if frames is None:
                    return
            self.frames_ready.emit(frames)

            while not self._stopping.is_set():
                try:
                    request = self._requests.get_nowait()
                except Empty:
                    frames = self._read_frames()
                    if frames is not None:
                        self.frames_ready.emit(frames)
                    self._stopping.wait(0.005)
                    continue
                if self._stopping.is_set():
                    break
                # Check cameras before expensive lazy loading. These frames are
                # preview/preflight only and must never be used for inference.
                frames = self._read_frames()
                if frames is None:
                    break
                self.frames_ready.emit(frames)
                kind, token = request[:2]
                if self._stopping.is_set():
                    break
                try:
                    if kind == 'recognize' and recognizer is None:
                        recognizer = self.recognizer_factory()
                    elif kind == 'locate' and tray_backend is None:
                        tray_backend = self.tray_factory()
                except Exception as exc:
                    if not self._stopping.is_set():
                        self.failed.emit(token, str(exc) or type(exc).__name__)
                    continue
                # Initialization can leave the driver's queue full of old frames.
                # Always drain after loading, and give automatic SCANs time to settle.
                settle_requested = (kind == 'empty_reference' or kind == 'recognize' and request[2])
                settle_seconds = self.scan_settle_seconds if settle_requested else 0
                frames = self._fresh_frames(settle_seconds)
                if frames is None:
                    break
                try:
                    if kind == 'empty_reference':
                        reference = {role: frame.copy() for role, frame in frames.items()}
                        if not self._stopping.is_set():
                            self.empty_reference_ready.emit(token, reference)
                    elif kind == 'recognize':
                        capture_completed = self._clock()
                        captures = {role: frame.copy() for role, frame in frames.items()}
                        result = recognizer.recognize_pair(frames['left'], frames['right'])
                        inference_completed = self._clock()
                        diagnostics = getattr(result, 'diagnostics', None)
                        if isinstance(diagnostics, dict):
                            diagnostics['captures'] = captures
                            diagnostics['capture'] = {
                                'settle_requested': request[2],
                                'settle_seconds': settle_seconds,
                                'drain_batches': self.drain_batches,
                                'request_received_monotonic': request[3],
                                'capture_completed_monotonic': capture_completed,
                                'inference_completed_monotonic': inference_completed,
                            }
                        if not self._stopping.is_set():
                            self.recognition_ready.emit(token, result)
                    else:
                        _, _, calibration, roi, offsets = request
                        locator = TrayLocator(calibration, TrayDetector(tray_backend), roi, offsets)
                        target = locator.locate(frames['top'])
                        if not self._stopping.is_set():
                            self.location_ready.emit(token, target)
                except Exception as exc:
                    if not self._stopping.is_set():
                        self.failed.emit(token, str(exc) or type(exc).__name__)
        except Exception as exc:
            self.failed.emit(-1, str(exc) or type(exc).__name__)
        finally:
            resources = [self._cameras, recognizer, tray_backend]
            for resource in resources:
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.failed.emit(-1, 'Vision cleanup failed: ' + str(exc))
