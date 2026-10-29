"""Deploy a registered, pinned model release; never reads training data."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlops.release import ROOT,verify_release


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',required=True)
    parser.add_argument('--space',default='adamnouassida5/deeplearning')
    args=parser.parse_args()
    registry=json.loads((ROOT/'models/registry.json').read_text())
    if args.version!=registry['active_version']:raise ValueError('Select the approved active version in the registry first')
    record=registry['releases'][args.version]
    if record['status']!='approved':raise ValueError('Only approved releases can deploy')
    remote=record['remote']
    if len(remote['revision'])!=40:raise ValueError('Use an immutable full commit SHA')
    from huggingface_hub import HfApi,snapshot_download
    snapshot=Path(snapshot_download(remote['repo_id'],repo_type='model',revision=remote['revision'],
                  allow_patterns=[remote['path']+'/*']))/remote['path']
    metadata=verify_release(snapshot)
    if metadata['checkpoint_sha256']!=record['checkpoint_sha256'] or metadata['files']!=record['files']:
        raise ValueError('Remote release differs from registry')
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)
        shutil.copytree(ROOT/'deployment/hf_space',target,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.pt','__pycache__','*.pyc'))
        shutil.copytree(ROOT/'backend',target/'backend',dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc'))
        (target/'models').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'models/registry.json',target/'models/registry.json')
        for name in ('model.pt','tokenizer.pt','release.json'):shutil.copy2(snapshot/name,target/name)
        commit=HfApi().upload_folder(repo_id=args.space,repo_type='space',folder_path=str(target),
                                     commit_message=f'Deploy approved release {args.version}')
        print(commit.commit_url)


if __name__=='__main__':main()
