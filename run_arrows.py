"""Run independent native-arrow recovery, optionally appending to frozen shape PPTs."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read,Runtime
from shape_integration.arrows.runner import run
from png2pptx.api.relay_client import load_bashrc_credentials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True);p.add_argument('--ids',nargs='+');p.add_argument('--config',type=Path)
    sources=p.add_mutually_exclusive_group()
    sources.add_argument('--from-styling',type=Path,help='Completed style run; append arrows to its frozen native PPTs')
    sources.add_argument('--from-arrows',type=Path,help='Completed arrow run; freeze its detection and geometry and recover styles again')
    p.add_argument('--retry-missing-geometry',action='store_true',help='With --from-arrows, retry only objects without usable geometry in a full-image view; preserve all existing geometry')
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):raise ValueError('Run name must be a single component')
    if a.retry_missing_geometry and not a.from_arrows:raise ValueError('--retry-missing-geometry requires --from-arrows')
    config=read(a.config or ROOT/'config.json');items=read(ROOT/'inputs/manifest.json')['images']
    if a.ids:
        if not set(a.ids)<={i['id'] for i in items}:raise ValueError('Unknown image IDs')
        items=[i for i in items if i['id'] in a.ids]
    base,key,_=load_bashrc_credentials()
    if base!=config['expected_boyue_base_url']:raise ValueError('Expected Boyue relay')
    result=run(ROOT,ROOT/'runs'/a.run_name,config,items,Runtime(config,base,key),
               style_source=(ROOT/a.from_styling).resolve() if a.from_styling else None,
               arrow_source=(ROOT/a.from_arrows).resolve() if a.from_arrows else None,retry_missing_geometry=a.retry_missing_geometry)
    return int(any(r['state']=='failed' for r in result['cases'].values()))


if __name__=='__main__':raise SystemExit(main())
