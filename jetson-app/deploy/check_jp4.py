"""Runtime check only; no camera or serial device is opened."""
import ctypes
import json
from pathlib import Path
import sys

from PyQt5 import QtCore
import cv2
import numpy
import yaml
import dataclasses
import serial
import tensorrt

assert sys.version_info[:2] == (3, 6), sys.version
assert tensorrt.__version__.startswith('8.2.'), tensorrt.__version__
cuda = ctypes.CDLL('libcudart.so')
count = ctypes.c_int()
code = cuda.cudaGetDeviceCount(ctypes.byref(count))
assert code == 0 and count.value > 0, (code, count.value)
report = {'python': sys.version, 'qt': QtCore.QT_VERSION_STR,
          'numpy': numpy.__version__, 'opencv': cv2.__version__,
          'tensorrt': tensorrt.__version__, 'cuda_devices': count.value}
print(json.dumps(report, indent=2))
work = Path.home() / 'ylers-jp4-work'
work.mkdir(exist_ok=True)
(work / 'runtime.json').write_text(json.dumps(report, indent=2), encoding='utf8')
