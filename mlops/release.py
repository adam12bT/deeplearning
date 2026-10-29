"""Export immutable inference releases; publishing never promotes a model."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'release.json').read_text())
    for name,digest in metadata['files'].items():
        if name not in ('model.pt','tokenizer.pt'):raise ValueError('Unexpected artifact name')
        if sha(directory/name)!=digest:raise ValueError(f'Corrupt artifact: {name}')
    if set(metadata['files'])!={'model.pt','tokenizer.pt'}:raise ValueError('Incomplete release')
    return metadata


def export(version,checkpoint_path):
    import torch
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,63}',version):raise ValueError('Invalid version')
    target=ROOT/'artifacts/releases'/version
    if target.exists():raise ValueError('Release exists; use a new version instead of overwriting weights')
    checkpoint_path=Path(checkpoint_path)
    c=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    codec=Path(c['tokenizer']['checkpoint'])
    if sha(codec)!=c['tokenizer']['checkpoint_sha256']:raise ValueError('Tokenizer mismatch')
    target.mkdir(parents=True)
    bundle={k:c[k] for k in ('config','model_state','conditions_metadata','tokenizer','epoch')}
    bundle['source_checkpoint_sha256']=sha(checkpoint_path)
    torch.save(bundle,target/'model.pt');shutil.copy2(codec,target/'tokenizer.pt')
    record=dict(version=version,created_utc=datetime.now(timezone.utc).isoformat(),
        epoch=c['epoch'],parameters=sum(x.numel() for x in c['model_state'].values()),
        checkpoint_sha256=sha(checkpoint_path),tokenizer_sha256=sha(codec),
        dataset_manifest_sha256=c['tokenizer']['manifest_sha256'],config=c['config'],
        best_val_loss=c['best_val_loss'],files={name:sha(target/name) for name in ('model.pt','tokenizer.pt')})
    (target/'release.json').write_text(json.dumps(record,indent=2)+'\n')
    verify_release(target)
    return target,record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--publish-repo',help='Optional Hugging Face model repo; publishes but never deploys')
    args=parser.parse_args()
    directory=ROOT/'artifacts/releases'/args.version
    if args.checkpoint:directory,record=export(args.version,args.checkpoint)
    else:record=verify_release(directory)
    registry_path=ROOT/'models/registry.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else dict(active_version=None,releases={})
    prior=registry['releases'].get(args.version)
    if prior and prior['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('Version already identifies different weights')
    if prior:record.update({k:v for k,v in prior.items() if k in ('remote','status','local_checkpoint')})
    if args.checkpoint:record['local_checkpoint']=args.checkpoint
    record.setdefault('status','candidate')
    if args.publish_repo:
        from huggingface_hub import HfApi
        api=HfApi();api.create_repo(args.publish_repo,repo_type='model',exist_ok=True)
        existing=api.list_repo_files(args.publish_repo,repo_type='model')
        if any(x.startswith(args.version+'/') for x in existing):
            raise ValueError('Remote version exists; immutable releases cannot be overwritten')
        commit=api.upload_folder(repo_id=args.publish_repo,repo_type='model',folder_path=str(directory),
            path_in_repo=args.version,commit_message=f'Publish immutable release {args.version}')
        api.create_tag(args.publish_repo,tag=args.version,revision=commit.oid,repo_type='model')
        record['remote']=dict(repo_id=args.publish_repo,revision=commit.oid,path=args.version)
    registry['releases'][args.version]=record
    registry_path.parent.mkdir(exist_ok=True)
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(dict(version=args.version,epoch=record['epoch'],parameters=record['parameters'],
                         remote=record.get('remote')),indent=2))


if __name__=='__main__':main()
