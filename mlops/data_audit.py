import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.validate_compositional import validate

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
result=validate(args.root,write_report=False)
result.pop('feature_counts')
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print(out)
