"""Style-only recovery from a frozen, terminal geometry run (no geometry API calls)."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import Runtime,read
from shape_integration.styling.runner import run
from png2pptx.api.relay_client import load_bashrc_credentials


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True);p.add_argument('--from-fitting',type=Path,required=True)
    p.add_argument('--ids',nargs='+');p.add_argument('--selection',type=Path,help='JSON case ID -> object ID list; testing only')
    p.add_argument('--semantic-seed',type=Path,help='Reuse terminal GLM styles under ROOT/cases/<id>/styling; numerical repair only, no API calls')
    p.add_argument('--sam-evidence',type=Path,help='Matching native crop cache: ROOT/<case>/line_evidence.json; no cache uses CLEAN only')
    p.add_argument('--config',type=Path);p.add_argument('--rules-only',action='store_true',help='No API calls; includes real PPT rendering')
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):raise ValueError('Run name must be one component')
    config=read(a.config or ROOT/'config.json');items=read(ROOT/'inputs/manifest.json')['images']
    if a.sam_evidence:
        evidence=a.sam_evidence.resolve()
        if not evidence.is_dir():raise ValueError('SAM evidence directory is missing')
        config.setdefault('styling',{})['sam_evidence_root']=str(evidence)
    if a.semantic_seed:
        config.setdefault('styling',{})['semantic_seed_root']=str(a.semantic_seed.resolve())
    selected=read(a.selection) if a.selection else None
    ids=set(a.ids or (selected.keys() if selected else [i['id'] for i in items]))
    if not ids<={i['id'] for i in items}:raise ValueError('Unknown case ID')
    if selected and (not ids<=set(selected) or any(not isinstance(v,list) or not v for v in selected.values())):
        raise ValueError('Selection must cover every requested case with nonempty object IDs')
    items=[i for i in items if i['id'] in ids];runtime=None
    if not a.rules_only and not a.semantic_seed:
        base,key,_=load_bashrc_credentials()
        if base!=config['expected_boyue_base_url']:raise ValueError('Expected Boyue endpoint')
        runtime=Runtime(config,base,key)
    result=run(ROOT,ROOT/'runs'/a.run_name,config,items,runtime,source=(ROOT/a.from_fitting).resolve(),selected=selected,rules_only=a.rules_only)
    return int(any(c['state']=='failed' for c in result['cases'].values()))


if __name__=='__main__':raise SystemExit(main())
