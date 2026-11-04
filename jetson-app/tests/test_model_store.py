import hashlib, json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_store import ModelStore

class ModelStoreTests(unittest.TestCase):
 def package(self, root, slot='side_detector', version='v1', names=None, engine=b'engine'):
  folder=Path(root)/('candidate-'+version); folder.mkdir()
  (folder/'weights.engine').write_bytes(engine)
  value={'schema':1,'slot':slot,'version':version,'engine':'weights.engine','sha256':hashlib.sha256(engine).hexdigest(),'input_shape':[1,3,640,640] if slot=='side_detector' else [1,3,320,320],'output_shape':[1,30,8400] if slot=='side_detector' else [1,1,320,320],'preprocess':'legacy_stitched_v1' if slot=='side_detector' else 'surface_rgb320_v1'}
  if slot=='side_detector': value['class_names']=names or (Path(__file__).parents[1]/'models/legacy_classes.txt').read_text(encoding='utf8').splitlines()
  manifest=folder/'manifest.json'; manifest.write_text(json.dumps(value),encoding='utf8'); return manifest
 def test_rejects_bad_hash_without_changing_active_pointer(self):
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(d); store.commit_active({'side_detector':{'version':'old'}},{})
   before=store.active_path.read_bytes(); m=self.package(d); data=json.loads(m.read_text()); data['sha256']='0'*64; m.write_text(json.dumps(data))
   with self.assertRaises(ValueError): store.validate_manifest(m,'side_detector')
   self.assertEqual(store.active_path.read_bytes(),before)
 def test_rejects_class_order_and_path_escape(self):
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(d); m=self.package(d,names=['wrong']*26)
   with self.assertRaises(ValueError): store.validate_manifest(m,'side_detector')
   data=json.loads(m.read_text()); data['engine']='../outside.engine'; m.write_text(json.dumps(data))
   with self.assertRaises(ValueError): store.validate_manifest(m,'side_detector')
 def test_commit_and_rollback_pointer_are_restartable(self):
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(d); store.commit_active({'surface_mask':{'version':'m1'}},{})
   store.commit_active({'surface_mask':{'version':'m2'}},{'surface_mask':{'version':'m1'}})
   store.commit_active({'surface_mask':{'version':'m1'}},{'surface_mask':{'version':'m2'}})
   restarted=ModelStore(d); self.assertEqual(restarted.read_active()['active']['surface_mask']['version'],'m1')
   self.assertEqual(restarted.read_active()['previous']['surface_mask']['version'],'m2')
if __name__=='__main__': unittest.main()

class WorkerTransactionTests(ModelStoreTests):
 def test_independent_rollback_and_failure(self):
  from vision_worker import VisionWorker
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(Path(d)/'store'); failures=[]
   class Fake:
    def __init__(self, releases): pass
    def recognize_pair(self,*args): return SimpleNamespace(error='bad' if failures else None, mask_status=('applied','disabled'))
    def close(self): pass
   def run(operation):
    worker=VisionWorker(store=store,loader=Fake); operation(worker); worker.shutdown(); worker.run()
   def install(slot,version):
    path=self.package(d,slot=slot,version=version)
    run(lambda w:(w.trial_candidate(path,slot,(None,None)),w.confirm_candidate()))
   install('side_detector','A'); install('side_detector','B'); install('surface_mask','M'); install('surface_mask','N')
   before=store.active_path.read_bytes(); failures.append(True)
   run(lambda w:w.rollback('side_detector',(None,None)))
   self.assertEqual(before,store.active_path.read_bytes()); failures.clear()
   run(lambda w:w.rollback('side_detector',(None,None)))
   state=ModelStore(Path(d)/'store').read_active()
   self.assertEqual(state['active']['side_detector']['version'],'A')
   self.assertEqual(state['active']['surface_mask']['version'],'N')
   self.assertEqual(state['previous']['surface_mask']['version'],'M')
   run(lambda w:w.rollback('surface_mask',(None,None)))
   self.assertEqual(store.read_active()['active']['surface_mask']['version'],'M')
   store.commit_active(state['active'], {'side_detector':None})
   run(lambda w:w.rollback('side_detector',(None,None)))
   self.assertNotIn('side_detector',store.read_active()['active'])
 def test_legacy_pointer_migration(self):
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(d)
   store.active_path.write_text(json.dumps({'active':{'side_detector':{'version':'B'},'surface_mask':{'version':'M'}},'previous':{'side_detector':{'version':'A'},'surface_mask':{'version':'M'}}}))
   state=store.read_active(); self.assertEqual(state['previous'],{'side_detector':{'version':'A'}})
   self.assertEqual(state['active']['surface_mask']['version'],'M')
   store.active_path.write_text(json.dumps({'active':{'side_detector':{'version':'A'}},'previous':{}}))
   self.assertIsNone(store.read_active()['previous']['side_detector'])
 def test_failed_trial_cancel_and_failed_commit_preserve_active(self):
  from vision_worker import VisionWorker
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as d:
   store=ModelStore(Path(d)/'store'); made=[]
   class Fake:
    def __init__(self, releases): self.closed=False; made.append(self)
    def recognize_pair(self,*args): return SimpleNamespace(error='bad' if len(made)==1 else None, mask_status=('disabled','disabled'))
    def close(self): self.closed=True
   path=self.package(d)
   w=VisionWorker(store=store,loader=Fake)
   w.trial_candidate(path,'side_detector',(None,None)); w.trial_candidate(path,'side_detector',(None,None)); w.cancel_candidate(); w.shutdown(); w.run()
   self.assertTrue(all(x.closed for x in made)); self.assertFalse(store.active_path.exists())
   def fail(*args): raise OSError('disk failure')
   store.commit_active=fail
   w=VisionWorker(store=store,loader=Fake); w.trial_candidate(path,'side_detector',(None,None)); w.confirm_candidate(); w.shutdown(); w.run()
   self.assertTrue(made[-1].closed); self.assertFalse(store.active_path.exists())
