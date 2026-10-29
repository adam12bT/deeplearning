"""Build an isolated, balanced old/new candidate without touching active data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.validate_compositional import validate
from datasets.generate_compositional import reserved


def main():
    root=Path('dataset_clean_candidate')
    source=Path('dataset/manifest.json')
    protected=hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run([sys.executable,'-B','datasets/generate_compositional.py','--output-dir',str(root),
                    '--per-combination','16','--seed','193031','--clean-edges'],check=True)
    entries=json.loads((root/'manifest.json').read_text())
    for e in entries:e['source_version']='clean_candidate'
    counts=Counter()
    for entry in json.loads(source.read_text()):
        key=(entry['mood'],entry['split'])
        limit=16 if reserved(entry) else {'train':12,'val':2,'test':2}[entry['split']]
        if counts[key]>=limit:continue
        counts[key]+=1
        target=root/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('dataset')/entry['path'],target)
        entries.append(dict(entry,source_version='original'))
    schema=json.loads((root/'schema.json').read_text())
    schema.update(per_combination=32,mixture='16 original + 16 clean per full combination',
                  original_manifest_sha256=protected)
    (root/'schema.json').write_text(json.dumps(schema,indent=2))
    (root/'manifest.json').write_text(json.dumps(entries,indent=2))
    result=validate(root)
    assert protected==hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps({k:v for k,v in result.items() if k!='feature_counts'},indent=2),flush=True)
    (root/'README.md').write_text('Candidate only: balanced original/clean drawings. '
        'Original split membership is preserved. Reserved rules remain test-only. '
        'Preview files show clean drawings, not model generations.\n')


if __name__=='__main__':main()
