"""Check the three butu-24 TensorRT engines without opening a camera or serial port."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from butu_20261110 import make_recognizer
from linux_20261107 import make_tray_backend
from tray_locator import TrayDetector


def main():
    path = ROOT / 'samples/butu_24_half_cylinder.png'
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (320, 640, 3):
        raise ValueError('预检样本必须为 640×320 双视角特征拼图')
    recognizer = make_recognizer()
    try:
        result = recognizer.recognize_pair(image[:, :320], image[:, 320:], preprocessed=True)
        if result.error:
            raise RuntimeError(result.error)
        if result.detection is None or result.detection.class_name != 'ban_yuan_zhu':
            raise RuntimeError('预检样本应识别为半圆柱')
        if 'applied' not in result.mask_status:
            raise RuntimeError('预检样本未生成有效表面掩膜：' + ', '.join(result.mask_status))
    finally:
        recognizer.close()
    tray = make_tray_backend()
    try:
        TrayDetector(tray, feature_input=True).detect(np.zeros((480, 640, 3), dtype=np.uint8))
    finally:
        tray.close()
    print('TensorRT 预检通过。形状为半圆柱，表面掩膜和托盘模型已推理。')


if __name__ == '__main__':
    main()
