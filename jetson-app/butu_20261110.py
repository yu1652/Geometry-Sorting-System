"""24-class per-view native recognizer for the 2026-11-10 butu dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models/butu_20261110'
CLASS_NAMES = tuple((MODEL_DIR / 'classes.txt').read_text(encoding='utf8').splitlines())
SHAPES = {
    'ba_leng_huan_zhu': '八棱环柱', 'ba_leng_zhu': '八棱柱',
    'ban_qiu': '半球', 'ban_qiu_ke': '半球壳',
    'liu_leng_huan_zhu': '六棱环柱', 'liu_leng_zhu': '六棱柱',
    'qi_leng_huan_zhu': '七棱环柱', 'qi_leng_zhu': '七棱柱',
    'san_leng_zhu': '三棱柱', 'shi_leng_huan_zhu': '十棱环柱',
    'shi_leng_zhu': '十棱柱', 'si_leng_huan_zhu': '四棱环柱',
    'si_leng_tai': '四棱台', 'si_leng_zhu': '四棱柱',
    'si_leng_zhu_jia': '四棱柱架', 'si_leng_zhui': '四棱锥',
    'wu_leng_huan_zhu': '五棱环柱', 'wu_leng_tai': '五棱台',
    'wu_leng_zhu': '五棱柱', 'yuan_huan_zhu': '圆环柱',
    'yuan_tai': '圆台', 'yuan_zhu': '圆柱',
    'yuan_zhui': '圆锥', 'ban_yuan_zhu': '半圆柱',
}
if CLASS_NAMES != tuple(SHAPES):
    raise ValueError('butu_20261110 classes.txt does not match the 24-class model order')


def make_recognizer():
    import yaml
    from legacy_vision import LegacyRecognizer
    from trt_backend import create_backend

    config = yaml.safe_load((ROOT / 'legacy_vision.yaml').read_text(encoding='utf8'))
    config['surface_mask']['enabled'] = True
    config['surface_mask']['input_mode'] = 'legacy_features'
    detector = create_backend(MODEL_DIR / 'shape_detector_fp16.engine', 'yolo_butu24')
    try:
        segmenter = create_backend(MODEL_DIR / 'surface_mask_fp16.engine', 'mask')
    except Exception:
        detector.close()
        raise
    try:
        return LegacyRecognizer(detector, segmenter, CLASS_NAMES, config,
                                expected_count=24, capture_color_sample=True,
                                per_view_features=True)
    except Exception:
        segmenter.close()
        detector.close()
        raise
