"""Import actual historical curves and existing images; does not retrain models."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlops.tracking import ROOT,RunTracker

RUNS={'landscape-small-v1':'runs/pixel_patch_training',
      'landscape-large-v1':'runs/pixel_patch_larger',
      'landscape-clean-candidate-v1':'runs/clean_finetune'}


def main():
    registry=json.loads((ROOT/'models/registry.json').read_text())
    output={}
    for version,directory in RUNS.items():
        record=registry['releases'][version];run=ROOT/directory
        history=json.loads((run/'history.json').read_text())
        params={k:record[k] for k in ('parameters','dataset_manifest_sha256','tokenizer_sha256')}
        params.update({k:record['config'][k] for k in ('n_layer','n_head','n_embd','block_size','vocab_size')})
        tracker=RunTracker(run,params,tags={'mlflow.runName':version,'recording_mode':'historical_import',
            'model_version':version,'model_status':record['status'],
            'active_deployment':str(version==registry['active_version']).lower(),
            'checkpoint_sha256':record['checkpoint_sha256'],
            'mlflow.note.content':'Historical import of real saved metrics. MLflow run timestamps reflect import, '
                'not original training dates. Compare validation loss only for the same dataset_manifest_sha256. '
                'Images are qualitative evidence; loss and duplicate counts are not image-quality scores.'})
        try:
            baseline=run/'baseline.json'
            if baseline.exists():
                data=json.loads(baseline.read_text());tracker.metrics({'baseline_val_loss':data['val_loss_on_candidate']},step=0)
            for row in history:tracker.epoch(row)
            tracker.metrics({'best_val_loss':record['best_val_loss'],'best_epoch':record['epoch'],
                'completed_epochs':history[-1]['epoch'],
                'training_seconds_total':sum(r['seconds'] for r in history)},step=history[-1]['epoch'])
            tracker.artifact(ROOT/'models/registry.json')
            for name in ('history.json','status.json','baseline.json','tokenizer_check.json'):
                tracker.artifact(run/name)
            for image in sorted((run/'samples').rglob('*.png')):
                tracker.artifact(image,'samples/'+image.parent.name)
            if version in ('landscape-small-v1','landscape-large-v1'):
                tag='small' if version=='landscape-small-v1' else 'large'
                for name in ('pixel_model_comparison','pixel_model_comparison_extra'):
                    source=ROOT/'runs'/name;report=json.loads((source/'report.json').read_text())
                    model=report['models'][tag]
                    if model['sha256']!=record['checkpoint_sha256']:raise ValueError('Evaluation/checkpoint mismatch')
                    tracker.metrics({name+'_exact_training_token_matches':model['exact_training_token_matches'],
                                     name+'_sample_count':len(model['nearest_tokens'])})
                    tracker.artifact(source/'report.json','evaluations/'+name)
                    tracker.artifact(source/'CONCLUSION.md','evaluations/'+name)
                    for image in source.glob('comparison_*.png'):tracker.artifact(image,'evaluations/'+name)
            tracker.finish();output[version]=tracker.run_id
        except BaseException:
            tracker.finish('FAILED');raise
        print('Imported',version,len(history),'epochs',flush=True)
    (ROOT/'runs/mlflow_import.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
