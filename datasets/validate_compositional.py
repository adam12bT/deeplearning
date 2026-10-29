"""Full-file validation before activating or deleting a previous dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image


def validate(root,write_report=True):
    root=Path(root).resolve()
    schema=json.loads((root/'schema.json').read_text())
    entries=json.loads((root/'manifest.json').read_text())
    assert schema['schema_version']==2
    fields=schema['condition_fields']
    names=schema['condition_names']
    splits=Counter(); combinations=Counter(); filenames=set(); seeds=set()
    hashes={}; feature_counts={s:{f:Counter() for f in fields} for s in ('train','val','test')}
    heldout=Counter(); objects=Counter(); all_paths=set()
    for e in entries:
        assert e['filename'] not in filenames
        assert e['seed'] not in seeds
        filenames.add(e['filename']); seeds.add(e['seed'])
        assert e['split'] in feature_counts
        assert all(e[f] in names[f] for f in fields)
        reserved=any(all(e[k]==v for k,v in rule.items()) for rule in schema['heldout_rules'])
        assert not reserved or e['split']=='test', 'Held-out feature rule leaked into training/validation'
        if reserved: heldout[e['split']]+=1
        assert (e['visible_objects']==0)==(e['objects']=='none')
        assert (e['water_pixels']==0)==(e['water']=='none')
        path=(root/e['path']).resolve(); path.relative_to(root)
        assert path.name==e['filename'] and path.parts[len(root.parts)]==e['split']
        assert path not in all_paths
        all_paths.add(path)
        with Image.open(path) as im:
            im.load(); assert im.mode=='RGB' and im.size==(64,64)
            raw=im.tobytes()
            assert all(any(raw[i:i+3]) for i in range(0,len(raw),3)), 'Unpainted black pixels'
            digest=hashlib.sha256(raw).hexdigest()
        assert digest not in hashes, f'Duplicate image content: {e["filename"]} / {hashes.get(digest)}'
        hashes[digest]=(e['filename'],e['split'])
        splits[e['split']]+=1
        combination=tuple(e[f] for f in fields)
        combinations[combination]+=1
        for f in fields: feature_counts[e['split']][f][e[f]]+=1
        objects[e['objects']]+=1
    expected=1
    for values in names.values(): expected*=len(values)
    assert len(combinations)==expected
    assert set(combinations.values())=={schema['per_combination']}
    assert len(entries)==expected*schema['per_combination']
    actual={p.resolve() for split in ('train','val','test') for p in (root/split).rglob('*.png')}
    assert actual==all_paths, 'Manifest/file mismatch'
    for split in ('train','val'):
        for field,values in names.items():
            assert set(feature_counts[split][field])==set(values), f'Missing individual feature: {split}/{field}'
    result={'passed':True,'schema_version':2,'images':len(entries),'unique_pixel_hashes':len(hashes),
            'combinations':len(combinations),'splits':dict(splits),'heldout_images':dict(heldout),
            'feature_counts':feature_counts,'objects':dict(objects),
            'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    if write_report:
        (root/'validation.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',nargs='?',default='dataset')
    result=validate(parser.parse_args().root)
    print(json.dumps({k:v for k,v in result.items() if k!='feature_counts'},indent=2))
