import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


import cv2
import numpy as np

try:
    from pycocotools import mask as mask_api
except ImportError:
    mask_api = None


@unittest.skipIf(mask_api is None, "pycocotools is required for annotation tests")
class LegacyLabelTests(unittest.TestCase):
    def setUp(self):
        from tools.check_legacy_labels import validate_pair
        self.validate_pair = validate_pair
        self.image = np.zeros((320, 640, 3), dtype=np.uint8)
        self.image[:, :320] = (10, 20, 30)
        self.image[:, 320:] = (40, 50, 60)
        left = np.zeros((320, 640), dtype=np.uint8)
        left[40:80, 50:90] = 1
        right = np.zeros_like(left)
        right[90:130, 420:460] = 1
        right[105:115, 435:445] = 0
        self.annotations = [self.annotation(1, left), self.annotation(2, right)]
        self.rows = ["0 0.109375 0.1875 0.0625 0.125", "0 0.6875 0.34375 0.0625 0.125"]
        self.class_map = {7: 0}

    def annotation(self, annotation_id, mask):
        return {"id": annotation_id, "image_id": 8, "category_id": 7,
                "segmentation": mask_api.encode(np.asfortranarray(mask))}

    def validate(self, rows, annotations):
        return self.validate_pair(self.image, rows, annotations, self.class_map)

    def test_crops_views_and_preserves_right_hole(self):
        left, right = self.validate(self.rows, self.annotations)
        self.assertEqual((left.side, right.side), ("left", "right"))
        for sample, offset in ((left, 0), (right, 320)):
            np.testing.assert_array_equal(sample.image, self.image[:, offset:offset + 320])
            self.assertEqual(sample.mask.shape, (320, 320))
            self.assertEqual(sample.mask.dtype, np.uint8)
        self.assertEqual(left.mask[45, 55], 255)
        self.assertEqual(right.mask[95, 105], 255)
        self.assertEqual(right.mask[110, 120], 0)
        self.assertEqual(right.area, 1500)
        self.assertEqual(right.bbox, (420, 90, 460, 130))

    def test_rejects_missing_or_wrong_class(self):
        with self.assertRaisesRegex(ValueError, "annotation 2"):
            self.validate(self.rows[:1], self.annotations)
        with self.assertRaisesRegex(ValueError, "annotation 1"):
            self.validate(["1" + self.rows[0][1:], self.rows[1]], self.annotations)
        with self.assertRaisesRegex(ValueError, "row 2"):
            self.validate(self.rows, self.annotations[:1])

    def test_rejects_empty_crossing_and_malformed_masks(self):
        full = np.zeros((320, 640), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "annotation 9.*empty"):
            self.validate(self.rows, [self.annotation(9, full)])
        full[95, 319:321] = 1
        with self.assertRaisesRegex(ValueError, "annotation 9.*seam"):
            self.validate(self.rows, [self.annotation(9, full)])
        malformed = self.annotation(9, full)
        malformed["segmentation"]["counts"] = [204801]
        with self.assertRaisesRegex(ValueError, "annotation 9.*RLE"):
            self.validate(self.rows, [malformed])

    def test_rejects_crossing_box_and_allows_seam_endpoint(self):
        with self.assertRaisesRegex(ValueError, "row 1.*seam"):
            self.validate(["0 0.5 0.1875 0.0625 0.125"], self.annotations)
        self.assertEqual(len(self.validate(
            ["0 0.28125 0.1875 0.4375 0.125"], self.annotations[:1])), 1)

    def test_rejects_ambiguous_matches(self):
        duplicate = dict(self.annotations[0], id=3)
        with self.assertRaisesRegex(ValueError, "annotation.*ambiguous"):
            self.validate(self.rows, self.annotations + [duplicate])
        with self.assertRaisesRegex(ValueError, "annotation 1.*ambiguous"):
            self.validate(self.rows + self.rows[:1], self.annotations)

    def test_cli_exports_unicode_positive_and_explicit_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, output = (root / name for name in ("images", "labels", "output"))
            images.mkdir()
            labels.mkdir()
            name = "\u5de6\u53f3"
            for stem, rows in ((name, self.rows), ("negative", [])):
                (images / (stem + ".png")).write_bytes(cv2.imencode(".png", self.image)[1].tobytes())
                (labels / (stem + ".txt")).write_text("\n".join(rows), encoding="utf-8")
            coco = root / "coco.json"
            class_map = root / "map.json"
            for annotation in self.annotations:
                annotation["segmentation"]["counts"] = annotation["segmentation"]["counts"].decode("ascii")
            data = {"images": [
                {"id": 8, "file_name": name + ".png", "width": 640, "height": 320},
                {"id": 9, "file_name": "negative.png", "width": 640, "height": 320},
            ], "annotations": self.annotations}
            coco.write_text(json.dumps(data), encoding="utf-8")
            class_map.write_text('{"7": 0}', encoding="utf-8")
            sources = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            cmd = [sys.executable, "-X", "utf8", "-B", "tools/check_legacy_labels.py",
                   "--images", str(images), "--labels", str(labels),
                   "--coco", str(coco), "--class-map", str(class_map), "--output", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(output.rglob("*.png"))), 8)
            right = cv2.imread(str(output / "masks" / "8_2_right.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual((right[95, 105], right[110, 120]), (255, 0))
            for side in ("left", "right"):
                mask = cv2.imread(str(output / "masks" / ("9_negative_" + side + ".png")), 0)
                self.assertEqual(mask.shape, (320, 320))
                self.assertFalse(mask.any())
            self.assertEqual({path: path.read_bytes() for path in sources}, sources)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

            cmd[-1] = str(root / "invalid")
            for kind, expected in (("missing", "negative.png"), ("orphan", "orphan.txt"),
                                   ("dimensions", name + ".png"), ("unsafe", "../escape.png")):
                with self.subTest(kind=kind):
                    negative = labels / "negative.txt"
                    orphan = labels / "orphan.txt"
                    if kind == "missing":
                        negative.unlink()
                    elif kind == "orphan":
                        orphan.touch()
                    elif kind == "dimensions":
                        data["images"][0]["width"] = 320
                    else:
                        data["images"][0]["file_name"] = "../escape.png"
                    coco.write_text(json.dumps(data), encoding="utf-8")
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse(Path(cmd[-1]).exists())
                    negative.touch()
                    if orphan.exists():
                        orphan.unlink()
                    data["images"][0].update(width=640, file_name=name + ".png")
