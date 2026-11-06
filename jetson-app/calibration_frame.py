"""Draft-only image rectangle for seeding the nine calibration pixels."""
import math

from app import ImageView, Qt


def validate_frame(box):
    try:
        x, y, width, height = map(float, box)
    except (TypeError, ValueError) as exc:
        raise ValueError('标定框须填写 X、Y、宽、高') from exc
    if (not all(math.isfinite(v) for v in (x, y, width, height))
            or x < 0 or y < 0 or width < 2 or height < 2
            or x + width > 639 or y + height > 479):
        raise ValueError('标定框须在 640×480 画面内，宽高至少 2 像素')
    return x, y, width, height


def frame_grid(box, inset_percent=15):
    x, y, width, height = validate_frame(box)
    inset = float(inset_percent) / 100
    if not math.isfinite(inset) or not 0 <= inset <= .45:
        raise ValueError('九点内缩须在 0–45% 之间')
    fractions = (inset, .5, 1 - inset)
    return [(x + width * fx, y + height * fy) for fy in fractions for fx in fractions]


class CalibrationFrameView(ImageView):
    """Drag inside to move, corners to resize, outside to draw a new frame."""
    def __init__(self, box, changed):
        super().__init__('顶部画面')
        self.box = validate_frame(box)
        self.changed = changed
        self._drag = None
        self.setToolTip('框内拖动平移；拖动四角缩放；框外按住拖动重新画框。')

    def image_position(self, x, y, clamp=False):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        width, height = pixmap.width(), pixmap.height()
        left, top = (self.width() - width) / 2, (self.height() - height) / 2
        if not clamp and not (left <= x < left + width and top <= y < top + height):
            return None
        return (max(0, min(639, (x - left) * 640 / width)),
                max(0, min(479, (y - top) * 480 / height)))

    def _position(self, event, clamp=False):
        point = event.position() if hasattr(event, 'position') else event.localPos()
        return self.image_position(point.x(), point.y(), clamp)

    def mousePressEvent(self, event):
        left = getattr(Qt, 'MouseButton', Qt).LeftButton
        point = self._position(event)
        if event.button() != left or point is None:
            super().mousePressEvent(event)
            return
        px, py = point
        x, y, width, height = self.box
        corners = ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
        radius = 10 * 640 / self.pixmap().width()
        for i, (cx, cy) in enumerate(corners):
            if abs(px - cx) <= radius and abs(py - cy) <= radius:
                self._drag = ('resize', corners[3 - i], self.box)
                break
        else:
            mode = 'move' if x <= px <= x + width and y <= py <= y + height else 'resize'
            self._drag = (mode, point, self.box)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        px, py = self._position(event, clamp=True)
        mode, (ax, ay), (x, y, width, height) = self._drag
        if mode == 'move':
            box = (max(0, min(639 - width, x + px - ax)),
                   max(0, min(479 - height, y + py - ay)), width, height)
        else:
            if abs(px - ax) < 2 or abs(py - ay) < 2:
                return
            box = (min(px, ax), min(py, ay), abs(px - ax), abs(py - ay))
        self.changed(box)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self.mouseMoveEvent(event)
            self._drag = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
