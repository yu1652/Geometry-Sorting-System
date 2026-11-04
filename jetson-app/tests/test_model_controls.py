import unittest
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import numpy as np
from types import SimpleNamespace
try:
 from PySide6.QtTest import QTest
 KEY_ESCAPE = app.Qt.Key.Key_Escape
except ImportError:
 from PyQt5.QtTest import QTest
 KEY_ESCAPE = app.Qt.Key_Escape
APP = app.QApplication.instance() or app.QApplication([])

class ModelControlsTests(unittest.TestCase):
 def make_window(self):
  w=app.Window(lambda: None)
  self.addCleanup(self.cleanup_window,w)
  return w
 def cleanup_window(self,w):
  if w.model_dialog is not None and w.model_dialog.busy: w.model_dialog.on_failed('test cleanup')
  if w.model_dialog is not None: w.model_dialog.reject()
  w.close()
  deadline=time.monotonic()+3
  while w._thread is not None and time.monotonic()<deadline:
   APP.processEvents(); time.sleep(.005)
  self.assertIsNone(w._thread, 'worker did not shut down')

 def test_trial_requires_explicit_confirmation_and_close_discards_late_result(self):
  w=self.make_window()
  w.display(np.zeros((320,640,3),dtype=np.uint8))
  d=app.ModelDialog(w)
  calls=[]
  w._thread.trial_candidate=lambda *args: calls.append(('trial',args))
  w._thread.confirm_candidate=lambda: calls.append(('confirm',))
  w._thread.cancel_candidate=lambda: calls.append(('cancel',))
  d.paths['side_detector']='candidate.json'
  d.start_trial('side_detector')
  self.assertEqual([x[0] for x in calls],['trial'])
  self.assertFalse(d.confirm_button.isEnabled())
  result=SimpleNamespace(error=None,mask_status=('disabled','disabled'),elapsed_ms=2,preview_bgr=np.zeros((320,640,3),dtype=np.uint8))
  d.on_trial({'slot':'side_detector','version':'v1','result':result})
  self.assertTrue(d.confirm_button.isEnabled())
  d.reject()
  d.on_trial({'slot':'side_detector','version':'v1','result':result})
  self.assertFalse(d.confirm_button.isEnabled())
  self.assertNotIn(('confirm',),calls)
 def test_closed_inflight_trial_keeps_gate_until_result(self):
  w=self.make_window()
  w.display(np.zeros((320,640,3),dtype=np.uint8))
  d=app.ModelDialog(w)
  w._thread.trial_candidate=lambda *args: None
  w._thread.cancel_candidate=lambda: None
  d.paths['surface_mask']='candidate.json'
  d.start_trial('surface_mask')
  d.reject()
  self.assertTrue(w.model_busy)
  w.recognize()
  self.assertIsNone(w.active_request_id)
  w._discard_closed_model_operation({})
  self.assertFalse(w.model_busy)
 def test_confirm_blocks_dialog_and_parent_close_until_acknowledged(self):
  w=self.make_window(); w.show()
  d=app.ModelDialog(w); d.show()
  w._thread.confirm_candidate=lambda: None
  d.pending=True; d.confirm()
  QTest.keyClick(d, KEY_ESCAPE); self.assertFalse(d.closed); self.assertTrue(d.isVisible())
  d.close(); self.assertFalse(d.closed); self.assertTrue(d.isVisible())
  w.close(); self.assertFalse(w.closing); self.assertTrue(w.isVisible())
  w.preview_view.show_frame(np.ones((2,2,3),dtype=np.uint8))
  w.result_label.setText('old result')
  d.on_changed({})
  self.assertIsNone(w.preview_view.frame)
  self.assertNotEqual(w.result_label.text(),'old result')
  d.reject(); self.assertTrue(d.closed)
 def test_rollback_blocks_close_until_failure(self):
  w=self.make_window(); w.show()
  w.display(np.zeros((320,640,3),dtype=np.uint8))
  d=app.ModelDialog(w); d.show()
  w._thread.rollback=lambda slot, fixture: None
  d.rollback('side_detector'); d.reject(); d.close(); w.close()
  self.assertFalse(d.closed); self.assertFalse(w.closing)
  d.on_failed('failure')
  self.assertIn('failure',d.status.text())
  d.reject(); self.assertTrue(d.closed)
