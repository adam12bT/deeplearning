"""Validate release selection in CI without downloading model weights."""
import json
from pathlib import Path
import re


def check_registry(path=Path('models/registry.json')):
    registry=json.loads(Path(path).read_text())
    assert registry['active_version'] in registry['releases']
    assert registry['releases'][registry['active_version']]['status']=='approved'
    for version,release in registry['releases'].items():
        assert version==release['version']
        for key in ('checkpoint_sha256','tokenizer_sha256','dataset_manifest_sha256'):
            assert re.fullmatch('[0-9a-f]{64}',release[key])
        assert release['parameters']>0 and release['epoch']>=0
        if release.get('remote'):
            assert re.fullmatch('[0-9a-f]{40}',release['remote']['revision'])
            assert release['remote']['path']==version
    assert registry['releases'][registry['active_version']].get('remote'),'Publish active release before deploying'
    print('Registry valid; active release:',registry['active_version'])


if __name__=='__main__':check_registry()
