"""Exercise real Qt widgets, media decoding and archival without hardware."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as operator_app
from app import Window
from media import MediaSource
import cv2
import numpy as np

app = operator_app.QApplication([])
window = Window()
window.show()
app.processEvents()
assert window.width() == 1024 and window.height() == 600
for name, button in window.buttons.items():
    point = button.mapTo(window, button.rect().bottomRight())
    assert point.x() < 1024 and point.y() < 600, (name, point)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    frame = np.zeros((320, 640, 3), np.uint8)
    frame[:, :320] = (50, 100, 150)
    frame[:, 320:] = (180, 150, 90)
    image = tmp/'双摄照片.png'
    cv2.imencode('.png', frame)[1].tofile(image)
    assert window.load_file(image)
    assert window.left_view.frame.shape == (320, 320, 3)
    assert window.right_view.frame[0, 0].tolist() == [180, 150, 90]
    window.layout_mode = '单视图（显示在左侧）'
    window.display(frame)
    assert window.left_view.frame.shape == (320, 640, 3)
    assert window.right_view.frame is None
    window.layout_mode = '双视图拼接（中线拆分）'
    window.display(frame)
    window.save()
    assert window.last_saved
    saved = window.last_saved
    assert cv2.imdecode(np.fromfile(saved/'source.png', dtype=np.uint8), 1).shape == frame.shape
    assert json.loads((saved/'metadata.json').read_text(encoding='utf8'))['hardware_enabled'] is False
    video = tmp/'replay.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 320))
    assert writer.isOpened()
    for n in range(5):
        writer.write(np.full((320, 640, 3), n*40, np.uint8))
    writer.release()
    assert window.load_file(video)
    window.play()
    assert window.timer.isActive()
    window.pause()
    assert not window.timer.isActive()
    window.step()
    assert window.source.frame_index == 2
    for _ in range(6):
        window.next_frame()
    assert not window.timer.isActive()
    window.play()
    assert window.source.frame_index == 1  # EOF restarts from beginning.
    window.stop()
    assert window.state.status == 'stopped'
    # Keep the smoke self-contained: generated pair is used instead of an
    # external training-audit image that is not present in every checkout.
    assert window.frame.shape == frame.shape
    app.processEvents()
    window.source.close()
window.close()
assert window.source.capture is None
print('PASS: 1024x600 controls, generated dual-view image, two layouts, video pause/step/EOF, saved PNG+metadata, release.')
