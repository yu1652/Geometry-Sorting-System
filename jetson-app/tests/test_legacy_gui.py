import threading, time, unittest
from pathlib import Path
import sys, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import cv2, numpy as np
from app import Window
from core import save_sample
from types import SimpleNamespace
APP=app.QApplication.instance() or app.QApplication([])
class FakeRecognizer:
 def __init__(self): self.calls=[]; self.infer_thread_id=None; self.close_thread_id=None; self.started=threading.Event(); self.release=threading.Event(); self.release.set()
 def recognize_pair(self,l,r):
  self.infer_thread_id=threading.get_ident(); self.calls.append((l.copy(),r.copy())); self.started.set(); self.release.wait(3)
  return SimpleNamespace(detection=SimpleNamespace(shape='圆环',confidence=.88,class_id=14),color=SimpleNamespace(name='蓝色',confidence=.91),mask_status=('applied','disabled'),preview_bgr=np.full((320,640,3),77,np.uint8),error=None,elapsed_ms=2)
 def close(self): self.close_thread_id=threading.get_ident()
class LegacyGuiTests(unittest.TestCase):
 def wait(self,p,timeout=3):
  end=time.time()+timeout
  while time.time()<end:
   APP.processEvents()
   if p(): return True
   time.sleep(.01)
  return p()
 def tearDownWindow(self,w):
  if w is not None:
   w.close(); self.wait(lambda:w._thread is None or not w._thread.isRunning())
 def test_manual_request_uses_raw_pair_shows_result_and_closes_on_worker(self):
  fake=FakeRecognizer(); w=Window(lambda:fake); self.addCleanup(self.tearDownWindow,w)
  with tempfile.TemporaryDirectory() as d:
   frame=np.zeros((320,640,3),np.uint8); frame[:,:320]=(10,20,30); frame[:,320:]=(40,50,60)
   p=Path(d)/'pair.png'; cv2.imwrite(str(p),frame); self.assertTrue(w.load_file(p)); original=w.frame.copy()
   w.recognize(); w.recognize(); self.assertTrue(self.wait(lambda:w.result_label.text()=='圆环')); self.assertEqual(len(fake.calls),1)
   np.testing.assert_array_equal(fake.calls[0][0],original[:,:320]); np.testing.assert_array_equal(fake.calls[0][1],original[:,320:])
   self.assertIn('applied',w.mask_status_label.text()); np.testing.assert_array_equal(w.frame,original)
   w.save(); saved=cv2.imdecode(np.fromfile(w.last_saved/'source.png',dtype=np.uint8),1); np.testing.assert_array_equal(saved,original)
  self.tearDownWindow(w); self.assertEqual(fake.infer_thread_id,fake.close_thread_id)
 def test_stale_completion_discarded_after_source_change(self):
  fake=FakeRecognizer(); w=Window(lambda:fake); self.addCleanup(self.tearDownWindow,w)
  frame=np.zeros((320,640,3),np.uint8); w.display(frame); w.recognize(); old_id=w.active_request_id; old_revision=w.frame_revision
  fake.release.clear(); self.assertTrue(fake.started.wait(2)); w.display(np.ones_like(frame)); fake.release.set()
  self.assertTrue(self.wait(lambda:w.active_request_id is None)); self.assertEqual(w.result_label.text(),'等待识别')
  self.assertEqual(len(fake.calls),1); self.assertNotEqual(old_revision,w.frame_revision)
 def test_frame_change_clears_preview_and_stale_failure_is_ignored(self):
  fake=FakeRecognizer(); w=Window(lambda:fake); self.addCleanup(self.tearDownWindow,w)
  w.display(np.zeros((320,640,3),np.uint8)); w.result_label.setText('圆环'); w.preview_view.show_frame(np.full((320,640,3),77,np.uint8))
  request_id=42; old_revision=w.frame_revision; w.active_request_id=request_id
  w.display(np.ones((320,640,3),np.uint8))
  self.assertIsNone(w.preview_view.frame); self.assertEqual(w.result_label.text(),'等待识别')
  w._on_recognition_failed(request_id,old_revision,'old source error')
  self.assertEqual(w.result_label.text(),'等待识别'); self.assertNotIn('old source error',w.status_label.text())
 def test_close_during_inflight_waits_for_owner_thread_cleanup(self):
  fake=FakeRecognizer(); w=Window(lambda:fake); self.addCleanup(self.tearDownWindow,w)
  fake.release.clear(); w.display(np.zeros((320,640,3),np.uint8)); w.recognize(); self.assertTrue(fake.started.wait(2))
  w.close(); self.assertTrue(w.closing); self.assertTrue(w._thread.isRunning())
  fake.release.set(); self.assertTrue(self.wait(lambda:w._thread is None))
  self.assertIsNotNone(fake.close_thread_id); self.assertEqual(fake.infer_thread_id,fake.close_thread_id)
if __name__=='__main__': unittest.main()
