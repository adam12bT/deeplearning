"""Build, validate and tokenize an isolated mixture; fine-tune a candidate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'runs/clean_finetune';run.mkdir(exist_ok=True)
    protected=['dataset/manifest.json','checkpoints/model_pixel_patches/best.pt',
               'checkpoints/model_pixel_patches_larger/best.pt',
               'checkpoints/pixel_patches_candidate/tokenizer.pt']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in protected}
    (run/'protected_hashes.json').write_text(json.dumps(hashes,indent=2))
    stages=[('build_and_validate',['datasets/build_clean_candidate.py']),
            ('check_tokenizer',['evaluation/check_clean_candidate.py']),
            ('extract_tokens',['training/extract_tokens.py','--manifest','dataset_clean_candidate/manifest.json',
                '--search-root','dataset_clean_candidate','--checkpoint','checkpoints/pixel_patches_candidate/tokenizer.pt',
                '--output-dir','checkpoints/tokens_clean_candidate']),
            ('fine_tune',['training/run_compositional_training.py','--epochs','8','--lr','5e-5',
                '--warmup-steps','100','--patience','3','--init-from','checkpoints/model_pixel_patches/best.pt',
                '--tokens-dir','checkpoints/tokens_clean_candidate','--out-dir','checkpoints/model_clean_candidate',
                '--run-dir','runs/clean_finetune'])]
    for stage,command in stages:
        (run/'pipeline_status.json').write_text(json.dumps(dict(status='running',stage=stage),indent=2))
        print('STAGE:',stage,flush=True)
        completed=subprocess.run([sys.executable,'-B','-u',*command],cwd=root)
        if completed.returncode:
            (run/'pipeline_status.json').write_text(json.dumps(dict(status='failed',stage=stage,exit_code=completed.returncode),indent=2))
            raise SystemExit(completed.returncode)
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
    (run/'pipeline_status.json').write_text(json.dumps(dict(status='completed',protected_artifacts_unchanged=True),indent=2))


if __name__=='__main__':main()
