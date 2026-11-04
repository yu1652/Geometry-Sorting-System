import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import numpy as np
from types import SimpleNamespace
APP = app.QApplication.instance() or app.QApplication([])

class ModelControlsTests(unittest.TestCase):
 def test_trial_requires_explicit_confirmation_and_close_discards_late_result(self):
  w=app.Window(lambda: None)
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
  w.close()
  while w._thread is not None:
   APP.processEvents()
 def test_closed_inflight_trial_keeps_gate_until_result(self):
  w=app.Window(lambda: None)
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
  w.close()
  while w._thread is not None: APP.processEvents()
