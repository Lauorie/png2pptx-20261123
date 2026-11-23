"""Paired forward reconstruction with optional SAM evidence, no scene postpatch."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# A frozen workload supplies SHAPE_REFERENCE_ROOT. Direct use of this checked-in
# entry point must load this project's implementation, never the historical
# external reference selected by the old core.py default.
_HERE = Path(__file__).resolve().parent
_DEFAULT_SOURCE = _HERE/'reference' if (_HERE/'reference/src').is_dir() else _HERE.parents[1]
_SOURCE_ROOT = Path(os.environ.get('SHAPE_REFERENCE_ROOT', _DEFAULT_SOURCE)).resolve()
if not (_SOURCE_ROOT/'src/shape_integration').is_dir():
    raise ValueError('Invalid SHAPE_REFERENCE_ROOT: '+str(_SOURCE_ROOT))
os.environ['SHAPE_REFERENCE_ROOT'] = str(_SOURCE_ROOT)
sys.path[:0] = [str(_SOURCE_ROOT/'src'), str(_SOURCE_ROOT/'vendor'), str(_HERE)]
from core import read, save, sha, public_proposal
# core.py can insert its own defaults; the explicit source stays authoritative.
sys.path[:0] = [str(_SOURCE_ROOT/'src'), str(_SOURCE_ROOT/'vendor')]
from models import Models
from runtime import ForwardRuntime
from evidence import frame_crop, normalize_predictions, group_duplicates
from shape_integration.sam_evidence import SamEvidence
from shape_integration.sam_guidance import GuidedSamEvidence
from shape_integration.sam_fixed import FixedSamEvidence
from shape_integration.shape import run_shape
from shape_integration.image_conflicts import run_filter
from shape_integration.common import geometry, usage
from shape_integration.pipeline import run_case, source_manifest

if Path(sys.modules['shape_integration'].__file__).resolve().parent != (_SOURCE_ROOT/'src/shape_integration').resolve():
    raise ValueError('A different shape implementation was already imported; start an isolated worker.')


def sam_index(item, config, models, folder):
    receipt = folder/'index.json'
    if receipt.exists():
        index = read(receipt)
        if index['source_sha256'] != item['source_sha256']:
            raise ValueError('SAM source changed')
        return index
    sources=config.get('reuse_sam_indices') or []
    if isinstance(sources,str):sources=[sources]
    for reuse in sources:
        old=Path(reuse)/item['id'];cached=old/'index.json';metadata=old/'full/masks/predictions.json'
        if cached.exists() and metadata.exists():
            index=read(cached)
            best=read(Path(config['training_run'])/'checkpoints/best.json')
            if (index['source_sha256']==item['source_sha256'] and index['candidate_threshold']==config['sam_threshold']
                    and read(metadata)['best_checkpoint']==best
                    and all(Path(g['representative']['mask_path']).is_file() for g in index['groups'])):
                index={**index,'cache_source':str(cached),'producer_elapsed_seconds':index.get('producer_elapsed_seconds',index['elapsed_seconds']),'elapsed_seconds':0.}
                save(receipt,index);return index
    if config.get('sam_cache_only'):
        raise ValueError('No validated SAM cache for '+item['id'])
    base = Path(config['baseline_root']); old = base/'runs'/config['baseline_run']/'cases'/item['id']
    with Image.open(base/item['original']) as im:
        source = im.convert('RGB')
    plan = read(old/'shapes/plan.json')
    views = [('full', [0, 0, source.width-1, source.height-1])]
    for region in plan['regions']:
        views.append((region['id'], [v*item['source_size'][i%2]/item['input_size'][i%2]
                                      for i,v in enumerate(region['bbox_2d'])]))
    def infer(view):
        name, box = view; target = folder/name
        image, frame = frame_crop(source, box, target)
        result = models.sam(image, target/'masks')
        return normalize_predictions(result, frame, name, source.size), result.get('seconds',0.)
    started = time.monotonic(); predictions = []; inference_seconds = 0.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows, seconds in pool.map(infer, views):
            predictions.extend(rows); inference_seconds += seconds
    groups = group_duplicates(predictions)
    index = {'source_sha256': item['source_sha256'], 'candidate_threshold': config['sam_threshold'],
             'mask_pixel_threshold': .5, 'group_count': len(groups), 'predictions': len(predictions),
             'elapsed_seconds': time.monotonic()-started, 'summed_service_seconds': inference_seconds,
             'groups': [{'id':g['id'],'representative':public_proposal(g['representative'])} for g in groups]}
    save(receipt, index); return index


def case_arm(item, arm, root, config, pipeline_config, models, index, fit_pool):
    output = root/arm; output.mkdir(parents=True,exist_ok=True)
    receipt = output/'cases'/item['id']/'forward_complete.json'
    if receipt.exists():
        previous = read(receipt)
        if sha(receipt.parent/'fitting/scene.json') != previous['scene_sha256']:
            raise ValueError('Completed scene changed')
        return previous
    base = Path(config['baseline_root']); old = base/'runs'/config['baseline_run']/'cases'/item['id']
    local = output/'localization'; folder = local/'cases'/item['id']
    assets_data = read(old/'dedup/image_boxes_original.json'); assets = assets_data['assets']
    mode = config.get('arm_modes', {}).get(arm)
    provider = (FixedSamEvidence(index['groups'], assets, mode=mode,
                precheck=config.get('arm_precheck', {}).get(arm, True), **config['evidence'])
                if mode and config.get('sam_schedule') == 'fixed' else
                GuidedSamEvidence(index['groups'], assets, mode=mode, **config['evidence']) if mode
                else SamEvidence(index['groups'], assets, **config['evidence']) if arm == 'sam_forward' else None)
    runtime = ForwardRuntime(models, pipeline_config, old, provider)
    started = time.monotonic()
    if not (folder/'localization_complete.json').exists():
        shapes = run_shape(base/item['original'], base/item['input'], folder/'shapes', runtime)
        source_identity = {'path': str(base/item['original']), 'sha256': item['source_sha256']}
        filtered = run_filter(shapes['objects'], assets, source_identity, folder/'image_conflicts', runtime, pipeline_config)
        save(folder/'dedup/image_boxes_original.json', assets_data)
        save(folder/'dedup/decisions.json',filtered['decisions']);save(folder/'dedup/removed.json',filtered['removed'])
        save(folder/'geometry.json',geometry(filtered['kept'],item['source_size'],shapes['uncertain']))
        summary = {**read(old/'summary.json'), 'final_shapes': len(filtered['kept']),
                   'local_failures': read(folder/'shapes/summary.json')['local_failures'],
                   'image_conflict_deferred_ids': filtered['deferred_ids'], 'arm': arm,
                   'initial_discovery_replayed': True, 'fitted_geometry_reused': False}
        save(folder/'summary.json',summary);save(folder/'localization_complete.json',summary)
    candidates = read(folder/'geometry.json')['objects']
    if provider:
        provider.set_inventory(candidates)
    provenance = source_manifest(local, [item])
    result = run_case(base, output, item, runtime, pipeline_config, fit_pool, local, provenance, False)
    scene = receipt.parent/'fitting/scene.json'
    observations = list((receipt.parent/'fitting/objects').glob('*/revision_*/sam_evidence.json'))
    evidence_used = sum(bool(read(p)['shown']) for p in observations)
    elapsed = time.monotonic()-started
    receipt_data = {'id':item['id'], 'arm':arm, 'scene_sha256':sha(scene), 'summary':result,
                    'elapsed_seconds':elapsed, 'sam_observations':evidence_used,
                    'sam_seconds':index['elapsed_seconds'] if provider else 0.,
                    'completed_at':time.time()}
    save(receipt,receipt_data)
    print('FORWARD_CASE',json.dumps(receipt_data,ensure_ascii=False),flush=True)
    return receipt_data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--ids',nargs='*')
    args=parser.parse_args();root=args.run_dir;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=read(args.config);base=Path(config['baseline_root'])
    if config.get('geometry_guard_enabled', True) and not (_SOURCE_ROOT/'src/shape_integration/fitting/geometry_guard.py').is_file():
        raise ValueError('Selected reference has no geometry guard; freeze the updated project into a new run.')
    items=read(base/'inputs/manifest.json')['images']
    if args.ids:items=[i for i in items if i['id'] in args.ids]
    pipe=read(base/'runs'/config['baseline_run']/'manifest.json')['config']
    pipe=copy.deepcopy(pipe);pipe.update(case_workers=config['case_workers'],api_concurrency=config['glm_concurrency'],local_workers=2)
    pipe['fitting'].update(max_revisions=config['max_revisions'],workers=config['fit_workers'],
                           recover_clipped_partial=config.get('recover_clipped_partial', False),
                           geometry_guard_enabled=config.get('geometry_guard_enabled', True))
    for key in ['styling_enabled','texts_enabled','arrows_enabled','image_export_enabled']:pipe[key]=False
    implementation = {str(path.relative_to(_SOURCE_ROOT)): sha(path)
                      for path in sorted((_SOURCE_ROOT/'src/shape_integration').rglob('*.py'))}
    identity={'config':config,'pipeline_config':pipe,'images':items,'implementation_sha256':implementation}
    if (root/'manifest.json').exists() and read(root/'manifest.json')!=identity:raise ValueError('Forward manifest changed')
    save(root/'manifest.json',identity)
    arms=config.get('arms',['glm_control','sam_forward'])
    for arm in arms:
        save(root/arm/'localization/manifest.json',{'images':items})
    models=Models(config);results=[]
    def pair(item):
        index=sam_index(item,config,models,root/'sam_cache'/item['id'])
        # Alternate order by manifest index, not output quality; both arms retain raw results.
        offset=0 if config.get('fixed_arm_order') else items.index(item)%len(arms)
        order=arms[offset:]+arms[:offset]
        output=[]
        for arm in order:output.append(case_arm(item,arm,root,config,pipe,models,index,fit_pool))
        return output
    with ThreadPoolExecutor(max_workers=config['fit_workers']) as fit_pool:
        pilots=[i for i in items if i['id'] in config.get('pilot_ids',[])]
        for item in pilots:results.extend(pair(item))
        if pilots:
            save(root/'pilot_complete.json',{'results':results,'completed_at':time.time()})
            print('PILOT_READY_FOR_INSPECTION',flush=True)
            deadline=time.monotonic()+config.get('pilot_review_timeout_seconds',1200)
            while not (root/'pilot_decision.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('Pilot inspection deadline reached; checkpoints retained')
                time.sleep(10)
            if read(root/'pilot_decision.json').get('action')!='continue':raise RuntimeError('Pilot stopped after inspection')
        errors = []
        save(root/'forward_errors.json', errors)
        with ThreadPoolExecutor(max_workers=config['case_workers']) as pool:
            pending={pool.submit(pair,i):i for i in items if i not in pilots}
            for future in as_completed(pending):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    error = {'id': pending[future]['id'], 'type': type(exc).__name__, 'error': str(exc)}
                    errors.append(error)
                    save(root/'forward_errors.json', errors)
                    print('FORWARD_CASE_FAILED', json.dumps(error), flush=True)
                save(root/'progress.json',{'cases_finished':len(results)//len(arms),'cases_failed':len(errors),
                    'cases_total':len(items),'results':results,'updated_at':time.time()})
        if errors:
            raise RuntimeError(f'{len(errors)} forward cases failed; see forward_errors.json. Completed cases retained.')
    save(root/'completion.json',{'cases':len(items),'arms':len(arms),'results':results,'completed_at':time.time()})


if __name__=='__main__':main()
