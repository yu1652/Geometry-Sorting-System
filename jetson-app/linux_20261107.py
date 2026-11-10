"""Native Xavier settings reconstructed from the 20261107 release files."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models/linux_20261107'
SHAPES = {
    'wlz': '五棱柱', 'llz': '六棱柱', 'zft': '正方体', 'slz': '三棱柱',
    'yzhu': '圆柱', 'yzhui': '圆锥', 'smt': '四面体', 'qiu': '球',
}
CLASS_NAMES = tuple(SHAPES)
TRAY_ROI = (16, 0, 519, 479)
PICK_REGION = (43.563, 0.125, 226.331, 164.027)
PUSH_OFFSET = (1.0, 0.0)
TRAY_COMPENSATION = {'center': (80., 80.), 'span': (80., 80.),
                     'x_left': 0., 'x_right': -4., 'y_top': 6., 'y_bottom': -6.}
SCAN_ATTEMPTS = 3
SCAN_TIMEOUT = 5.0
ROUND_SIZE = 12


def make_recognizer():
    import yaml
    from legacy_vision import LegacyRecognizer
    from trt_backend import create_backend

    config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
    config['surface_mask']['enabled'] = False
    config['color'].update(bbox_center_ratio=.6, valid_min_value=40,
                           valid_max_value=245, valid_min_saturation=40)
    backend = create_backend(MODEL_DIR / 'shape_detector_fp16.engine', model_kind='yolo_win8')
    try:
        return LegacyRecognizer(backend, None, CLASS_NAMES, config, expected_count=len(CLASS_NAMES))
    except Exception:
        backend.close()
        raise


def make_tray_backend():
    from trt_backend import create_backend

    return create_backend(MODEL_DIR / 'tray_object_detector_fp16.engine', model_kind='tray')
