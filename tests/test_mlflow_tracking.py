import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlops.tracking import RunTracker,training_tracking


@unittest.skipUnless(importlib.util.find_spec('mlflow'),'Install MLflow to test the tracking backend')
class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.uri='sqlite:///'+(self.root/'tracking.db').as_posix()
        self.root_patch=patch('mlops.tracking.ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.close_database)

    def close_database(self):
        # MLflow caches SQLAlchemy pools; Windows cannot delete an open SQLite file.
        from mlflow.tracking._tracking_service.utils import _get_store
        _get_store(self.uri).engine.dispose()

    def test_reimport_and_resume_do_not_duplicate_epoch_metrics(self):
        params={'dataset_manifest_sha256':'dataset-a','parameters':12}
        record={'epoch':1,'train_loss':.5,'val_loss':.6,'seconds':1.,'lr':.001}
        first=RunTracker(self.root/'run',params,uri=self.uri)
        first.epoch(record);first.finish()
        second=RunTracker(self.root/'run',params,uri=self.uri)
        self.assertEqual(first.run_id,second.run_id)
        second.epoch(record);second.epoch(dict(record,epoch=2,val_loss=.55));second.finish()
        values=second.client.get_metric_history(second.run_id,'val_loss')
        self.assertEqual([(m.step,m.value) for m in values],[(1,.6),(2,.55)])
        self.assertAlmostEqual(second.client.get_run(second.run_id).data.metrics['generalization_gap'],.05)

    def test_dataset_identity_cannot_change_inside_a_run(self):
        first=RunTracker(self.root/'run',{'dataset_manifest_sha256':'a'},uri=self.uri);first.finish()
        with self.assertRaisesRegex(ValueError,'Immutable MLflow parameter'):
            RunTracker(self.root/'run',{'dataset_manifest_sha256':'b'},uri=self.uri)

    def test_failure_is_recorded_and_sensitive_artifacts_are_excluded(self):
        with patch.dict('os.environ',{'MLFLOW_TRACKING_URI':self.uri}):
            with self.assertRaisesRegex(RuntimeError,'training failed'):
                with training_tracking(True,self.root/'run',{'parameters':12}) as tracker:
                    secret=self.root/'.env';secret.write_text('DO_NOT_UPLOAD=secret')
                    weights=self.root/'model.pt';weights.write_bytes(b'weights')
                    tracker.artifact(secret);tracker.artifact(weights)
                    raise RuntimeError('training failed')
        self.assertEqual(tracker.client.get_run(tracker.run_id).info.status,'FAILED')
        self.assertEqual(tracker.client.list_artifacts(tracker.run_id),[])


if __name__=='__main__':unittest.main()
