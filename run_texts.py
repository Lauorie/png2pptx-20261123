"""Recognize line-level text and append editable text to frozen native PPTs."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read,Runtime
from shape_integration.texts.runner import run
from png2pptx.api.relay_client import load_bashrc_credentials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True);p.add_argument('--ids',nargs='+');p.add_argument('--config',type=Path)
    p.add_argument('--from-run',type=Path,help='Completed shape/arrow run; append text to frozen PPTs')
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):raise ValueError('Run name must be one path component')
    config=read(a.config or ROOT/'config.json');items=read(ROOT/'inputs/manifest.json')['images']
    if a.ids:
        if not set(a.ids)<={i['id'] for i in items}:raise ValueError('Unknown image IDs')
        items=[i for i in items if i['id'] in a.ids]
    base,key,_=load_bashrc_credentials()
    if base!=config['expected_boyue_base_url']:raise ValueError('Expected Boyue relay')
    result=run(ROOT,ROOT/'runs'/a.run_name,config,items,Runtime(config,base,key),
               source=(ROOT/a.from_run).resolve() if a.from_run else None)
    return int(any(r['state']=='failed' for r in result['cases'].values()))


if __name__=='__main__':raise SystemExit(main())
