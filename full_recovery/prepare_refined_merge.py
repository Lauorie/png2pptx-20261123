"""Freeze a source-preserving replay against a supplied completed arrow run."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import shutil
from .merge_existing import read,save,sha,prepare
from .service_endpoints import IMAGE_EDIT_BASE_URL


def collect(project, target):
    baseline=project/read(project/'runs/LATEST_SHAPE_API.json')['run_dir']
    manifest=read(baseline/'source_manifest.json');items={r['id']:r for r in manifest['images']}
    options={cid:[] for cid in items}
    for run in sorted((project/'runs').iterdir()):
        if not run.is_dir() or not (run/'sources.json').is_file():continue
        for row in read(run/'sources.json').get('ready',[]):
            cid=row['item']['id'];marker=run/'cases'/cid/'completed.json'
            if cid not in options or not marker.is_file():continue
            done=read(marker)
            if row['item']['source_sha256']!=items[cid]['source_sha256']:continue
            options[cid].append((float(done.get('at',0)),run,row))
    origins={}
    for cid,item in items.items():
        selected=None;origin=None
        for stamp,run,row in sorted(options[cid],key=lambda v:v[0],reverse=True):
            keys=['shape_scene','shape_styles','shape_complete','shape_pptx','shape_preview','image_registry']
            if all(Path(row['paths'][k]).is_file() and sha(row['paths'][k])==row['sha256'][k] for k in keys):
                selected={k:Path(row['paths'][k]) for k in keys};origin=str(run);break
        if selected is None:
            case=baseline/'shape/geometry/cases'/cid
            selected={'shape_scene':case/'fitting/scene.json','shape_styles':case/'styling/styles.json',
                'shape_complete':case/'complete.json','shape_pptx':case/'styling/styled.pptx',
                'shape_preview':case/'styling/native.png',
                'image_registry':baseline/'shape/localization/cases'/cid/'dedup/image_boxes_original.json'}
            origin=str(baseline)
        case=target/'shape/geometry/cases'/cid;case.mkdir(parents=True,exist_ok=True)
        (case/'fitting').symlink_to(selected['shape_scene'].parent,target_is_directory=True)
        (case/'styling').symlink_to(selected['shape_styles'].parent,target_is_directory=True)
        (case/'complete.json').symlink_to(selected['shape_complete'])
        registry=target/'shape/localization/cases'/cid/'dedup/image_boxes_original.json';registry.parent.mkdir(parents=True,exist_ok=True)
        registry.symlink_to(selected['image_registry'])
        origins[cid]={'published_composition':origin,'paths':{k:str(v) for k,v in selected.items()},
                      'source_sha256':item['source_sha256']}
    save(target/'source_manifest.json',manifest)
    save(target/'selection.json',{'policy':'latest_completed_composition_inputs_with_matching_hashes; complete_shape_baseline_for_uncomposed_cases',
                                'read_only_symlinks':True,'cases':origins})
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--arrow-run',type=Path,required=True);p.add_argument('--ids',nargs='+')
    p.add_argument('--text-run',type=Path,required=True,
        help='Explicit historical text product run; never inherit a fixed previous run')
    p.add_argument('--source-collection',type=Path,required=True,
        help='Explicit historical product collection; use run_ppt.py for new recognition from PNGs')
    p.add_argument('--case-workers',type=int,default=4);p.add_argument('--api-concurrency',type=int,default=16)
    p.add_argument('--timeout-experiment-inputs',type=Path)
    p.add_argument('--reuse-requests-from',type=Path)
    p.add_argument('--image-edit-url',default=IMAGE_EDIT_BASE_URL)
    p.add_argument('--image-edit-model',default='SenseNova-U1.5-8B-MoT')
    args=p.parse_args();project=Path(__file__).resolve().parents[1];run=args.run_dir.resolve()
    collection=args.source_collection.resolve()
    if not (collection/'selection.json').is_file():
        p.error('Expected an explicit frozen selection.json; automatic historical latest-input selection is disabled')
    prepare(project,SimpleNamespace(run_dir=run,shape_run=collection,text_run=args.text_run.resolve(),
        arrow_run=args.arrow_run,ids=args.ids,case_workers=args.case_workers,api_concurrency=args.api_concurrency,
        require_all=True,resume_from=None,reuse_assets_from=None,image_edit_url=args.image_edit_url,
        image_edit_model=args.image_edit_model,image_edit_concurrency=2))
    shutil.copy2(run/'sources.json',run/'replay_sources.json')
    (run/'code/merge_existing.py').write_text('from pathlib import Path\nfrom full_recovery.shape_replay import main\n'
        'from full_recovery.timeout_experiment import run_experiment\nif __name__ == "__main__":\n'
        '    run_experiment(Path(__file__).resolve().parents[1])\n    main()\n')
    settings=read(run/'settings.json');settings.update(source_scope_workers=2,mapping_workers=2,
        shape_replay_mode='native_source_order_only',worker_max_seconds=345600)
    if args.reuse_requests_from:
        settings['reuse_request_roots']=[str(args.reuse_requests_from.resolve())]
    save(run/'settings.json',settings)
    save(run/'code_manifest.json',{str(f.relative_to(run)):sha(f) for f in (run/'code').rglob('*') if f.is_file()})
    if args.timeout_experiment_inputs:
        shutil.copy2(args.timeout_experiment_inputs,run/'timeout_experiment_inputs.json')
        frozen=read(run/'code_manifest.json');frozen['timeout_experiment_inputs.json']=sha(run/'timeout_experiment_inputs.json')
        save(run/'code_manifest.json',frozen)
    save(run/'experiment.json',{'source_collection':str(collection),'arrow_run':str(args.arrow_run.resolve()),
        'cases':[r['id'] for r in read(run/'source_manifest.json')['images']],
        'original_inputs_reused_in_place':True,'arrow_inference_enabled':False,'new_semantic_style_calls':0,
        'image_generation_after_source_ownership':True,'source_coverage_deletion_guards':True})
    print('REFINED_MERGE_PREPARED',run,settings['requested'],flush=True)


if __name__=='__main__':main()
