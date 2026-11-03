"""Read real photos or video without opening cameras."""
from pathlib import Path
import cv2
import numpy as np


class MediaSource:
    def __init__(self):
        self.path = None
        self.capture = None
        self.image = None
        self.fps = 25.0
        self.frame_index = 0

    @property
    def is_video(self):
        return self.capture is not None

    def open(self, path):
        self.close()
        self.path = Path(path)
        if self.path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            self.image = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.image is None:
                raise ValueError('图片损坏或不支持该图片格式')
        else:
            self.capture = cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                self.close()
                raise ValueError('无法打开录像，请检查格式和文件路径')
            rate = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = rate if 1 <= rate <= 120 else 25.0

    def read(self):
        if self.image is not None:
            return self.image.copy()
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if ok:
            self.frame_index += 1
            return frame
        return None

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture, self.image, self.path = None, None, None
        self.frame_index = 0
