"""Isolated imports for frozen discovery and SAM; one case per process."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Standalone frozen runners load the same verified copy implementation.
for _copy_root in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[1]/'geometry_code'):
    if (_copy_root/'checked_copy.py').is_file():
        sys.path.insert(0,str(_copy_root));break
from checked_copy import copy_file

def read(p):
    return json.loads(p.read_text())

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix('.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    t.replace(p)

def baseline_rows(handoff, result):
    regions={r['id']:r for r in result['regions']}
    return [dict(id=f'B{i:04d}',bbox=t['bbox_native'],label=t['identity_hint'],
                 render_type=t['render_type'],grouping=t['grouping'],
                 extent=regions[t['region_id']]['extent'],
                 original_target_id=t['target_id'],original_region_id=t['region_id'])
            for i,t in enumerate(handoff['tasks'],1)]

def run(root, cid, stage):
    if stage=='localization':
        d=root/'localization'
        sys.path.insert(0,str(d/'code'))
        import run as L
        import presentation
        from transport.model import Model
        L.publish=lambda *a:None
        presentation.publish=lambda *a:None
        cfg=read(d/'config.json')
        result=L.run_case(d,cid,Model(cfg,run_dir=d,system=L.Draft.DISCOVERY_SYSTEM),cfg)
        if result['state'] not in ('completed','completed_with_issues'):
            raise RuntimeError('Localization did not complete')
        return
    code=root/'geometry_code/checkpoint/upstream/sam_supplement/code'
    sys.path.insert(0,str(code))
    import experiment as E
    from request_budget import FileSlots
    E.publish=lambda *a:None
    loc=root/'localization/cases'/cid
    upstream=read(loc/'result.json')
    inp=root/'sam/inputs'/cid;inp.mkdir(parents=True,exist_ok=True)
    original=root/'localization/inputs'/cid/'source.png'
    copy_file(original,inp/'source.png')
    write(inp/'baseline_native.json',baseline_rows(read(loc/'reconstruction_handoff.json'),upstream))
    write(inp/'input_provenance.json',dict(localization=str(loc),source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),prior_run_reused=False))
    missing=root/'sam/missing'
    (missing/'cases'/cid).mkdir(parents=True,exist_ok=True)
    cfg=read(missing/'config.json')
    cfg['sam_existing_coverage_threshold']=.9
    write(missing/'cases'/cid/'status.json',dict(state='running',stage='sam_scan_only'))
    try:
        _, candidates, errors = E.scan(missing,cid,cfg,FileSlots(root/'sam_slots',cfg['sam_workers']))
        write(missing/'cases'/cid/'status.json',dict(state='completed_with_issues' if errors else 'completed',
              stage='sam_scan_only',candidates=len(candidates),issues=errors,glm_missing_calls=0))
    except Exception as exc:
        write(missing/'cases'/cid/'status.json',dict(state='failed',stage='sam_scan_only',error=str(exc)))
        # Initial geometry and the independent global omission check can still run.

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--case',required=True)
    ap.add_argument('--stage',choices=['localization','sam_scan'],required=True)
    a=ap.parse_args()
    run(a.root.resolve(),a.case,a.stage)
