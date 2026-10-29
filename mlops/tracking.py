"""Optional MLflow tracking; model weights and secrets are never uploaded here."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='Pixel Landscape Training'


def local_tracking_uri():
    directory=ROOT/'artifacts/mlflow';directory.mkdir(parents=True,exist_ok=True)
    return 'sqlite:///'+(directory/'tracking.db').as_posix()


def tracking_uri():
    return os.getenv('MLFLOW_TRACKING_URI') or local_tracking_uri()


class RunTracker:
    def __init__(self,run_dir,params,tags=None,uri=None,experiment=EXPERIMENT):
        from mlflow import MlflowClient
        self.directory=Path(run_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.uri=uri or tracking_uri();self.client=MlflowClient(tracking_uri=self.uri)
        existing=self.client.get_experiment_by_name(experiment)
        if existing is None:
            artifact_dir=ROOT/'artifacts/mlflow/files';artifact_dir.mkdir(parents=True,exist_ok=True)
            experiment_id=self.client.create_experiment(experiment,artifact_location=artifact_dir.as_uri())
        else:experiment_id=existing.experiment_id
        mapping=self.directory/'mlflow.json'
        saved=json.loads(mapping.read_text()) if mapping.exists() else {}
        if saved.get('tracking_uri')==self.uri:
            self.run_id=saved['run_id']
            prior=self.client.get_run(self.run_id)
            if prior.info.experiment_id!=experiment_id:raise ValueError('MLflow experiment changed for this run')
        else:
            name=(tags or {}).get('mlflow.runName',self.directory.name)
            self.run_id=self.client.create_run(experiment_id,tags={'mlflow.runName':name}).info.run_id
            prior=self.client.get_run(self.run_id)
        for key,value in params.items():
            if value is None:continue
            value=str(value)
            if key in prior.data.params and prior.data.params[key]!=value:
                raise ValueError(f'Immutable MLflow parameter changed: {key}')
            if key not in prior.data.params:self.client.log_param(self.run_id,key,value)
        for key,value in (tags or {}).items():self.client.set_tag(self.run_id,key,str(value))
        self.client.update_run(self.run_id,status='RUNNING')
        mapping.write_text(json.dumps({'tracking_uri':self.uri,'run_id':self.run_id},indent=2))
        self._known={}

    def metrics(self,values,step=0):
        for key,value in values.items():
            if value is None:continue
            value=float(value)
            if not math.isfinite(value):raise ValueError(f'Nonfinite MLflow metric: {key}')
            if key not in self._known:
                self._known[key]={(m.step,m.value) for m in self.client.get_metric_history(self.run_id,key)}
            if (step,value) not in self._known[key]:
                self.client.log_metric(self.run_id,key,value,step=step)
                self._known[key].add((step,value))

    def epoch(self,record):
        self.metrics({k:record[k] for k in ('train_loss','val_loss','lr','seconds') if k in record},step=record['epoch'])
        if record.get('train_loss') is not None and record.get('val_loss') is not None:
            self.metrics({'generalization_gap':record['val_loss']-record['train_loss']},step=record['epoch'])

    def artifact(self,path,folder='metadata'):
        path=Path(path)
        # An explicit allowlist prevents accidental .env, weight or token uploads.
        if path.is_file() and path.suffix.lower() in ('.json','.png','.md'):
            self.client.log_artifact(self.run_id,str(path),artifact_path=folder)

    def finish(self,status='FINISHED'):
        self.client.set_terminated(self.run_id,status=status)


class DisabledTracker:
    def metrics(self,*args,**kwargs):pass
    def epoch(self,*args,**kwargs):pass
    def artifact(self,*args,**kwargs):pass


@contextmanager
def training_tracking(enabled,run_dir,params,tags=None):
    tracker=RunTracker(run_dir,params,tags) if enabled else DisabledTracker()
    try:
        yield tracker
    except BaseException as error:
        if enabled:tracker.finish('KILLED' if isinstance(error,KeyboardInterrupt) else 'FAILED')
        raise
    else:
        if enabled:tracker.finish()
