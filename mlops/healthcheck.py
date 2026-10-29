"""Wait for the deployed app, then verify weights without consuming GPU quota."""
import argparse
import json
import time
from huggingface_hub import HfApi,hf_hub_download
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--space',default='adamnouassida5/deeplearning')
parser.add_argument('--version',required=True)
args=parser.parse_args()
registry=json.loads(Path('models/registry.json').read_text())
release=registry['releases'][args.version]
api=HfApi()
for _ in range(90):
    stage=str(api.get_space_runtime(args.space).stage)
    if stage=='RUNNING':break
    if stage in ('BUILD_ERROR','RUNTIME_ERROR','CONFIG_ERROR'):raise RuntimeError(f'Space failed: {stage}')
    time.sleep(10)
else:raise TimeoutError('Space did not become ready')
path=hf_hub_download(args.space,'release.json',repo_type='space',force_download=True)
actual=json.loads(Path(path).read_text())
assert actual['version']==args.version and actual['files']==release['files']
files={s.rfilename:s for s in api.space_info(args.space,files_metadata=True).siblings}
for name,digest in release['files'].items():
    assert files[name].lfs and files[name].lfs.sha256==digest
print('Space running with verified release',args.version)
