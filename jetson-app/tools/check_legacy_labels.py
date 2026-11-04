"""Validate stitched legacy labels and export 320 x 320 training samples."""

import argparse
from dataclasses import dataclass
import json
from math import isfinite
import ntpath
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import cv2
import numpy as np


HEIGHT, SEAM, WIDTH = 320, 320, 640
SIDES = ("left", "right")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_id: Optional[int]
    annotation_id: Optional[int]
    side: str
    image: np.ndarray
    mask: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]] = None
    area: int = 0


def _runs(counts):
    if isinstance(counts, list):
        yield from counts
        return
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    if not isinstance(counts, bytes):
        raise ValueError("counts must be a string or list")

    # COCO stores signed five-bit chunks and delta-codes runs from the fourth run onward.
    previous = []
    pos = 0
    while pos < len(counts):
        run = shift = 0
        while True:
            if pos == len(counts) or shift > 20:
                raise ValueError("invalid compressed count")
            chunk = counts[pos] - 48
            pos += 1
            if not 0 <= chunk <= 63:
                raise ValueError("invalid compressed count")
            run |= (chunk & 31) << shift
            shift += 5
            if not chunk & 32:
                if chunk & 16:
                    run -= 1 << shift
                break
        if len(previous) >= 3:
            run += previous[-2]
        previous.append(run)
        yield run


def _decode(segmentation, mask_api):
    size = segmentation.get("size") if isinstance(segmentation, dict) else None
    if size != [HEIGHT, WIDTH] or any(type(value) is not int for value in size):
        raise ValueError("RLE size must be [320, 640]")
    total = 0
    try:
        counts = segmentation.get("counts")
        for run in _runs(counts):
            if type(run) is not int or run < 0 or run > HEIGHT * WIDTH - total:
                raise ValueError("count is outside the image")
            total += run
        if total != HEIGHT * WIDTH:
            raise ValueError("counts do not cover the image")
        if isinstance(counts, list):
            segmentation = mask_api.frPyObjects(segmentation, HEIGHT, WIDTH)
        return mask_api.decode(segmentation)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid RLE: %s" % exc) from exc


def _box(row, row_number):
    fields = row.split() if isinstance(row, str) else row
    try:
        class_id, cx, cy, width, height = map(float, fields)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("YOLO row %s: expected class cx cy width height" % row_number) from exc
    if not (class_id.is_integer() and 0 <= class_id < 26):
        raise ValueError("YOLO row %s: class must be in 0..25" % row_number)
    if not all(isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)):
        raise ValueError("YOLO row %s: coordinates must be normalized" % row_number)
    x1, x2 = (cx - width / 2) * WIDTH, (cx + width / 2) * WIDTH
    y1, y2 = (cy - height / 2) * HEIGHT, (cy + height / 2) * HEIGHT
    # Permit round-off at exact image and seam endpoints.
    epsilon = 1e-9
    if not (-epsilon <= x1 < x2 <= WIDTH + epsilon and -epsilon <= y1 < y2 <= HEIGHT + epsilon):
        raise ValueError("YOLO row %s: invalid or out-of-bounds rectangle" % row_number)
    if x1 < SEAM - epsilon and x2 > SEAM + epsilon:
        raise ValueError("YOLO row %s: rectangle crosses the seam" % row_number)
    return int(class_id), int(cx >= 0.5), (x1, y1, x2, y2)


def _overlap(box, other):
    x1, y1, x2, y2 = box
    a1, b1, a2, b2 = other
    return min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1)


def validate_pair(bgr, yolo_rows, coco_annotations, coco_to_yolo) -> list:
    """Validate a stitch and return per-instance samples with absolute half-open boxes.

    Rows may be YOLO text lines or five-value sequences. For an empty pair, the
    caller is responsible for confirming the explicit source image and TXT.
    """
    from pycocotools import mask as mask_api

    if not isinstance(bgr, np.ndarray) or bgr.shape != (HEIGHT, WIDTH, 3) or bgr.dtype != np.uint8:
        raise ValueError("expected a 640 x 320 uint8 BGR image")
    views = (bgr[:, :SEAM], bgr[:, SEAM:])
    boxes = {number: _box(row, number) for number, row in enumerate(yolo_rows, 1)
             if not isinstance(row, str) or row.strip()}
    matches = {}
    samples = []
    for annotation in coco_annotations:
        annotation_id = annotation["id"]
        try:
            class_id = coco_to_yolo[annotation["category_id"]]
            mask = _decode(annotation.get("segmentation"), mask_api)
            ys, xs = np.nonzero(mask)
            if not xs.size:
                raise ValueError("empty mask")
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if bbox[0] < SEAM < bbox[2]:
                raise ValueError("mask crosses the seam")
            side = int(bbox[0] >= SEAM)
            candidates = [
                row for row, (row_class, row_side, row_box) in boxes.items()
                if row_class == class_id and row_side == side and _overlap(bbox, row_box)
            ]
            if not candidates:
                raise ValueError("no overlapping YOLO rectangle of the same class and side")
            if len(candidates) > 1:
                raise ValueError("ambiguous YOLO rows %s" % candidates)
            row = candidates[0]
            if row in matches:
                raise ValueError("ambiguous YOLO row %s also matches annotation %s" % (row, matches[row]))
            matches[row] = annotation_id
        except KeyError as exc:
            raise ValueError("annotation %s: missing class mapping or field %s" % (annotation_id, exc)) from exc
        except ValueError as exc:
            raise ValueError("annotation %s: %s" % (annotation_id, exc)) from exc

        start = side * SEAM
        samples.append(Sample(annotation["image_id"], annotation_id, SIDES[side], views[side],
                              mask[:, start:start + SEAM] * 255, bbox, int(xs.size)))

    for row in sorted(boxes.keys() - matches.keys()):
        raise ValueError("YOLO row %s: no matching annotation" % row)
    return samples or [Sample(None, None, side, view, np.zeros((HEIGHT, SEAM), np.uint8))
                       for side, view in zip(SIDES, views)]


def _safe_relative(root, name):
    path = PurePosixPath(name)
    reserved = ntpath.isreserved(name) if hasattr(ntpath, "isreserved") else PureWindowsPath(name).is_reserved()
    if (not name or any(char in name for char in '\\:<>"|?*\x00') or path.is_absolute() or reserved
            or any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in name.split("/"))):
        raise ValueError("unsafe relative filename %r" % name)
    relative = Path(*path.parts)
    try:
        (root / relative).resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("unsafe relative filename %r" % name) from exc
    return relative


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _class_map(path):
    mapping = _read_json(path)
    if not isinstance(mapping, dict):
        raise ValueError("class map must be a JSON object")
    result = {}
    for key, value in mapping.items():
        category = int(key)
        if category in result or type(value) is not int or not 0 <= value < 26:
            raise ValueError("invalid class map entry %r: %r" % (key, value))
        result[category] = value
    return result


def _read_coco(path):
    coco = _read_json(path)
    if not isinstance(coco, dict) or any(not isinstance(coco.get(key), list)
                                         for key in ("images", "annotations")):
        raise ValueError("COCO must contain images and annotations arrays")
    images = {}
    annotations = {}
    ids = set()
    for record in coco["images"]:
        if not isinstance(record, dict) or type(record.get("id")) is not int or record["id"] in images:
            raise ValueError("invalid or duplicate COCO image: %r" % record)
        image_id = record["id"]
        images[image_id] = record
        annotations[image_id] = []
    for record in coco["annotations"]:
        if not isinstance(record, dict):
            raise ValueError("invalid COCO annotation: %r" % record)
        image_id = record.get("image_id")
        annotation_id = record.get("id")
        context = "image %r annotation %r" % (image_id, annotation_id)
        if type(image_id) is not int or image_id not in images:
            raise ValueError("%s: image_id does not identify a COCO image" % context)
        if type(annotation_id) is not int or annotation_id in ids:
            raise ValueError("%s: invalid or duplicate annotation ID" % context)
        if type(record.get("category_id")) is not int:
            raise ValueError("%s: invalid category_id" % context)
        ids.add(annotation_id)
        annotations[image_id].append(record)
    return images, annotations


def _check_orphans(root, expected, suffixes, stage):
    for path in root.rglob("*"):
        if stage in path.parents:
            continue
        if path.is_file() and path.suffix.lower() in suffixes and path.relative_to(root) not in expected:
            raise ValueError("%s: no corresponding COCO image record" % path)


def _write_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("cannot encode %s" % path)
    path.write_bytes(encoded.tobytes())


def _export(args):
    images, labels = args.images.resolve(), args.labels.resolve()
    if not images.is_dir() or not labels.is_dir():
        raise ValueError("--images and --labels must be existing directories")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory: %s" % output)
    mapping = _class_map(args.class_map)
    records, annotations = _read_coco(args.coco)
    expected_images, expected_labels = set(), set()

    with TemporaryDirectory(prefix=".legacy-labels-", dir=output.parent) as temporary:
        stage = Path(temporary).resolve() / "export"
        (stage / "images").mkdir(parents=True)
        (stage / "masks").mkdir()
        for image_id, record in records.items():
            filename = record.get("file_name")
            try:
                if not isinstance(filename, str):
                    raise ValueError("COCO file_name must be a relative filename")
                relative = _safe_relative(images, filename)
                txt = _safe_relative(labels, relative.with_suffix(".txt").as_posix())
                if relative in expected_images or txt in expected_labels:
                    raise ValueError("duplicate image filename or YOLO TXT")
                if (record.get("height"), record.get("width")) != (HEIGHT, WIDTH):
                    raise ValueError("COCO dimensions must be 640 x 320")
                raw = np.frombuffer((images / relative).read_bytes(), dtype=np.uint8)
                if not raw.size:
                    raise ValueError("image file is empty")
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                rows = (labels / txt).read_text(encoding="utf-8-sig").splitlines()
                for sample in validate_pair(image, rows, annotations[image_id], mapping):
                    instance = "negative" if sample.annotation_id is None else sample.annotation_id
                    name = "%s_%s_%s.png" % (image_id, instance, sample.side)
                    _write_png(stage / "images" / name, sample.image)
                    _write_png(stage / "masks" / name, sample.mask)
                expected_images.add(relative)
                expected_labels.add(txt)
            except (OSError, ValueError, cv2.error) as exc:
                raise ValueError("image %s (%s): %s" % (image_id, filename, exc)) from exc
        _check_orphans(images, expected_images, IMAGE_SUFFIXES, stage)
        _check_orphans(labels, expected_labels, {".txt"}, stage)
        if output.exists() or output.is_symlink():
            raise ValueError("output must be a new directory: %s" % output)
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "labels", "coco", "class-map", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        # Report the optional dependency even for empty datasets.
        from pycocotools import mask  # noqa: F401
        _export(args)
    except ImportError as exc:
        print("error: pycocotools is required for label export (%s)" % exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
