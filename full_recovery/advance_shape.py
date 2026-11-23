"""Advance unfinished checkpoints in place using a separately frozen code snapshot.

Completed deliveries and completed object observations are preserved. The old
writer must have released both locks before activation changes any work file.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from .compose import sha


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def prepare(project,base,capsule,concurrency,case_workers=24,repair_selection=None):
    import run_shapes
    manifest=read(base/'source_manifest.json');old=read(base/'settings.json');dataset=Path(old['input_root'])
    run_shapes.prepare(SimpleNamespace(run_dir=capsule/'shape',input_root=dataset,ids=None,max_stroke_pt=None))
    settings={**old,'shape_work_root':str(base/'shape'),
        'shape_base_run':str(base),'workflow_scope':'shape','api_global_concurrency':concurrency,
        'glm_concurrency':concurrency,'case_workers':case_workers,'fitting_workers':96,'styling_workers':64,
        'resume_finished_geometry':True,'resume_transport_cases_once':True,'sam_attempt_id':capsule.name,
        'worker_max_seconds':345600,'maximum_stroke_width_pt':None}
    latest=project/'runs/LATEST_SHAPE_API.json'
    pointer=read(latest) if latest.exists() else {}
    settings['previous_shape_snapshot']=pointer.get('active_snapshot',str(base)) if pointer.get('run_dir')==str(base) else str(base)
    settings.pop('api_relays',None)
    if repair_selection is not None:
        save(capsule/'repair_selection.json',read(repair_selection))
        settings['shape_repair_selection']=str(capsule/'repair_selection.json')
    save(capsule/'settings.json',settings);save(capsule/'shape/settings.json',settings)
    config=read(base/'shape/pipeline_config.json')
    config.update(case_workers=case_workers,api_concurrency=concurrency,
        request_timeout_seconds=settings['glm_request_timeout_seconds'])
    config['fitting']['source_frame_refinement']=True
    config['styling']['maximum_stroke_width_pt']=None
    save(capsule/'shape/pipeline_config.json',config)
    frozen_config=capsule/'shape/code/config.json'
    frozen=read(frozen_config)
    frozen.update(case_workers=case_workers,api_concurrency=concurrency,
        request_timeout_seconds=settings['glm_request_timeout_seconds'])
    save(frozen_config,frozen)
    code_hashes=read(capsule/'shape/code_sha256.json')
    code_hashes['config.json']=sha(frozen_config)
    save(capsule/'shape/code_sha256.json',code_hashes)
    save(capsule/'source_manifest.json',manifest)
    code=capsule/'code';code.mkdir()
    shutil.copytree(project/'full_recovery',code/'full_recovery',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(project/'scripts/lo_guard.py',code/'lo_guard.py')
    for name in ['test_full_composition.py','test_full_api.py','test_full_line_dedup.py','test_full_text_occlusion.py']:
        shutil.copy2(project/'tests'/name,code/name)
    backend=capsule/'backends';backend.mkdir()
    (backend/'reference_fonts').symlink_to(base/'backends/reference_fonts',target_is_directory=True)
    (capsule/'renderer_runtime').symlink_to(base/'renderer_runtime',target_is_directory=True)
    for name in ['jobs','logs','render_locks']:(capsule/name).mkdir(exist_ok=True)
    identities={str(p.relative_to(capsule)):sha(p) for p in code.rglob('*') if p.is_file()}
    for p in [capsule/'settings.json',capsule/'shape/settings.json',capsule/'shape/pipeline_config.json',capsule/'source_manifest.json']:
        identities[str(p.relative_to(capsule))]=sha(p)
    if repair_selection is not None:identities['repair_selection.json']=sha(capsule/'repair_selection.json')
    save(capsule/'code_manifest.json',identities)
    save(capsule/'status.json',{'state':'prepared','workflow_scope':'shape','same_output_root':str(base/'shape'),'at':time.time()})
    print('PREPARED_SAME_DIRECTORY_SHAPE_CONTINUATION',capsule,'API',concurrency,flush=True)


def activate(capsule):
    settings=read(capsule/'settings.json');base=Path(settings['shape_base_run']);work=base/'shape'
    marker=capsule/'activation.json'
    if marker.exists():return read(marker)
    previous=Path(settings.get('previous_shape_snapshot',str(base)))
    repairs=read(settings['shape_repair_selection']).get('cases',{}) if settings.get('shape_repair_selection') else {}
    requested={item['id'] for item in read(base/'source_manifest.json')['images']}
    if not set(repairs)<=requested:raise ValueError('Repair selection contains an unknown case')
    with ExitStack() as stack:
        for path in dict.fromkeys([base/'worker.lock',previous/'worker.lock',work/'.pipeline.lock']):
            handle=stack.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        archive=capsule/'previous_work';preserved={};moved=[];objects_kept=0
        def move(path):
            if not path.exists():return
            dest=archive/path.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True)
            path.rename(dest);moved.append(str(path.relative_to(base)))
        # Validate the complete selection before moving any production file.
        for cid,spec in repairs.items():
            case=work/'geometry/cases'/cid;scene=read(case/'fitting/scene.json')
            ids=spec.get('objects',[])
            if not ids or len(set(ids))!=len(ids) or not set(ids)<=scene['objects'].keys():
                raise ValueError('Repair selection requires existing unique object IDs: '+cid)
            if scene['source_sha256']!=spec['source_sha256'] or sha(case/'fitting/scene.json')!=spec['scene_sha256']:
                raise ValueError('Repair selection source/scene changed: '+cid)
            if not {r['object_id'] for r in spec.get('repair_existing',[])}<=set(ids):
                raise ValueError('Repair requests must reference the selected objects: '+cid)
        for item in read(base/'source_manifest.json')['images']:
            case=work/'geometry/cases'/item['id'];receipt=case/'complete.json'
            if item['id'] in repairs:
                fit=case/'fitting';scene=read(fit/'scene.json');spec=repairs[item['id']]
                candidates=[o['candidate'] for o in scene['objects'].values()]
                seeded={r['object_id'] for r in spec.get('repair_existing',[])}
                for oid in spec['objects']:
                    path=fit/'objects'/oid
                    if oid in seeded:
                        dest=archive/path.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copytree(path,dest,ignore=shutil.ignore_patterns('profile','__pycache__'))
                    else:move(path)
                for path in list(fit.iterdir()):
                    if path.name!='objects':move(path)
                for path in (receipt,case/'failure.json',case/'styling',work/'delivery'/item['id'],
                             work/'sam_styles'/item['id']/'line_evidence.json'):move(path)
                save(fit/'resume_candidates.json',{'source_sha256':item['source_sha256'],
                    'localization_sha256':sha(case/'geometry.json'),'objects':candidates,
                    'repair_object_ids':spec['objects'],'repair_existing':spec.get('repair_existing',[]),
                    'previous_scene':str(archive/(fit/'scene.json').relative_to(base))})
                objects_kept+=len(scene['objects'])-len(spec['objects'])+len(seeded)
                continue
            if receipt.exists():
                record=read(receipt)
                if sha(record['pptx'])!=record['pptx_sha256']:raise ValueError('Completed shape changed: '+item['id'])
                preserved[item['id']]={'pptx_sha256':record['pptx_sha256'],'geometry_sha256':sha(case/'fitting/scene.json')}
                continue
            move(case/'failure.json')
            local=work/'localization/cases'/item['id']
            if not (local/'summary.json').exists():move(local)
            fit=case/'fitting';scene=read(fit/'scene.json') if (fit/'scene.json').exists() else None
            finished=(scene is not None and (fit/'summary.json').exists()
                and read(fit/'summary.json').get('state') in ('completed','completed_with_unresolved')
                and not any(o.get('stop_reason') in ('observation_transport_failed','check_transport_failed') for o in scene['objects'].values()))
            if finished:
                # An unfinished style stage belongs to the previous snapshot's
                # code and settings. Reuse geometry and SAM evidence, then
                # restart styling with the current snapshot's identity.
                move(case/'styling')
                continue
            initial=read(case/'geometry.json')['objects'] if (case/'geometry.json').exists() else []
            known={o['id']:o for o in initial}
            for obj in (fit/'objects').glob('*'):
                done=read(obj/'done.json').get('record',{}) if (obj/'done.json').exists() else None
                if (done is not None and obj.name in known
                        and done.get('candidate')==known[obj.name]
                        and done.get('stop_reason') not in ('observation_transport_failed','check_transport_failed')):
                    objects_kept+=1;continue
                move(obj)
            for folder in fit.glob('coverage_*'):move(folder)
            move(case/'styling')
            # Keep source-bound tile/mask caches. Only the aggregate scene-bound
            # index must be rebuilt when an unfinished scene changes.
            move(work/'sam_styles'/item['id']/'line_evidence.json')
        for source,dest in [(capsule/'shape/settings.json',work/'settings.json'),
                            (capsule/'shape/pipeline_config.json',work/'pipeline_config.json')]:
            move(dest);shutil.copy2(source,dest)
        result={'completed_cases_preserved':preserved,'completed_objects_preserved':objects_kept,
                'selectively_repaired_cases':{k:v['objects'] for k,v in repairs.items()},
                'moved_incomplete_work':moved,'same_output_root':str(work),'api_global_concurrency':settings['api_global_concurrency'],
                'code_snapshot':str(capsule/'shape/code'),'at':time.time()}
        save(marker,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-run',type=Path,required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--api-concurrency',type=int,default=96);p.add_argument('--case-workers',type=int,default=24)
    p.add_argument('--repair-selection',type=Path);a=p.parse_args()
    base=a.base_run.resolve();capsule=base/'updates'/a.snapshot
    if capsule.exists():raise ValueError('Snapshot already exists')
    if not a.snapshot.replace('_','').isalnum():raise ValueError('Unsafe snapshot name')
    prepare(Path(__file__).resolve().parents[1],base,capsule,a.api_concurrency,a.case_workers,a.repair_selection)


if __name__=='__main__':main()
