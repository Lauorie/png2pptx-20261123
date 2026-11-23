"""One synthetic complete queue run with real PPT rendering and no API calls."""
import argparse
import copy
import sys
from pathlib import Path
from unittest.mock import patch
import run_geometry_dedup24 as Q


def run(work,output,base_config):
    if output.exists():raise FileExistsError(output)
    sys.path[:0]=[str(work/'geometry_code'),str(work/'geometry_code/lib')]
    from PIL import Image
    root=output/'run';source=output/'source';root.mkdir(parents=True)
    (root/'geometry_code').symlink_to(work/'geometry_code',target_is_directory=True)
    case=source/'cases/synthetic';case.mkdir(parents=True);(source/'inputs/synthetic').mkdir(parents=True)
    Image.new('RGB',(100,80),(40,90,120)).save(source/'inputs/synthetic/source.png')
    Image.open(source/'inputs/synthetic/source.png').save(case/'current.png')
    tasks=[dict(target_id=f'R{i}',display_id=i,region_id=f'B{i}',bbox_native=[20,20,80,60],
        identity_hint='visible arrow',render_type='linework',kind='single',grouping='single',render_owner=True,
        readonly_identities=[]) for i in [1,2]]
    Q.write(case/'reconstruction_handoff.json',dict(case_id='synthetic',tasks=tasks))
    Q.write(case/'result.json',dict(state='completed',issues=[]))
    Q.write(case/'status.json',dict(state='completed'))
    cfg=copy.deepcopy(Q.read(base_config));cfg.update(source_localization=str(source),source_controller_run=str(source),
        case_workers=1,task_workers=2,cpu_workers=2,sam_workers=1,api_concurrency=1,
        geometry_critic_enabled=False,reconcile_enabled=True,reconcile_max_paths=64,
        preview_interval_seconds=10,geometry_preview_width_original_px=2.0)
    Q.write(root/'config.json',cfg);Q.write(root/'cases.json',['synthetic']);Q.write(root/'manifest.json',dict(files={}))
    calls=[]
    class Fake:
        def __init__(self,*a,**k):pass
        def ask(self,folder,prompt,images,validate,**kwargs):
            if prompt.get('task','').startswith('Reconcile'):
                calls.append('dedup');return validate({'decision':'accept'})
            calls.append('geometry');box=prompt['targets'][0]['bbox'];y=(box[1]+box[3])/2
            return validate(dict(status='draw',nodes=[['a',box[0],y],['b',box[2],y]],
                routes=[dict(id='R1',paths=[dict(id='p',heads=['none','triangle'],segments=[dict(kind='line',nodes=['a','b'])])])]))
    with patch('transport.model.Model',Fake):Q.run(root)
    out=root/'cases/synthetic';summary=Q.read(out/'completed.json')
    assert summary['state']=='completed',summary
    assert calls.count('geometry')==2 and calls.count('dedup')==1,calls
    assert summary['paths_before']==summary['paths_after']==2
    assert Q.read(out/'reconciliation/audit.json')['nonedited_rows_preserved']
    for name in ['arrows.pptx','before_dedup/arrows.pptx','result/overlap.png','before_dedup/result/overlap.png']:
        assert (out/name).is_file(),name
    assert Image.open(out/'result/overlap.png').convert('RGB').getpixel((0,79))==(40,90,120)
    result=dict(api_calls=0,geometry_calls=2,dedup_calls=1,native_ppt_before_after=True,
        source_background_preserved=True,execution_state=Q.read(root/'execution.json')['state'])
    Q.write(output/'checks.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-config',type=Path,required=True)
    a=p.parse_args();run(a.work.resolve(),a.output.resolve(),a.base_config.resolve())
