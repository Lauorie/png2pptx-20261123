"""Review every saved object with a shared object pool across and within images."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
from pathlib import Path
import threading
import time

from . import batch
from .common import now, read, save, sha
from .fitting.control import process_object, inspection_images
from .fitting.seed import revision_seed
from .fitting.scene import assemble, export
from .pipeline import source_manifest, import_localization, case_report, report, audit, validate_candidates, append_styling


def run(workspace, root, config, items, runtime, *, source, prepare_only=False):
    """Reuse selected geometry; run fresh critic and bounded refinement/recropping.

    Existing discovered candidates are included. Coverage and localization remain
    frozen, so this compares critic/refinement on the same object population.
    """
    root.mkdir(parents=True, exist_ok=True)
    provenance = source_manifest(source, items)
    config = copy.deepcopy(config)
    config['fitting_recheck_source'] = str(source.resolve())
    config['fitting']['coverage_rounds'] = 0
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.freeze(workspace, root, config, items, provenance)
        states = {}; seeds = {}; pending = []
        for item in items:
            case = item['id']; folder = root/'cases'/case; fitted = folder/'fitting'
            for name, key in [('original', 'source_sha256'), ('input', 'input_sha256')]:
                if sha(workspace/item[name]) != item[key]:
                    raise ValueError('Source image changed: '+case)
            upstream = import_localization(folder, source, item, provenance)
            old = source/'cases'/case/'fitting'
            candidates_path = old/'candidates.json'
            candidates = read(candidates_path)['objects']
            validate_candidates({'objects': candidates, 'image_size': item['source_size'],
                                 'coordinate_system': 'image_pixel_centers'}, item)
            save(fitted/'candidates.json', {'objects': candidates, 'source': str(candidates_path),
                                          'source_sha256': sha(candidates_path)})
            states[case] = {'item': item, 'folder': fitted, 'candidates': candidates, 'records': {},
                            'upstream': upstream, 'last_checkpoint': 0.,
                            'image_assets': read(folder/'dedup/image_boxes_original.json')['assets']}
            seeds[case] = {c['id']: revision_seed(old/'objects'/c['id']) for c in candidates}
        manifest = {'source': str(source.resolve()), 'seeds': seeds,
                    'scope': 'All final source candidates, fresh critic plus bounded refine/recrop; '
                             'objects lacking usable geometry receive fresh observation. No new coverage scan.'}
        if (root/'recheck_manifest.json').exists():
            if read(root/'recheck_manifest.json') != manifest:
                raise ValueError('Recheck source changed; use a new run')
        else:
            save(root/'recheck_manifest.json', manifest)
        # Round-robin submission avoids putting an entire large image ahead of
        # every other image. All objects share one worker pool and API limiter.
        for index in range(max(len(s['candidates']) for s in states.values())):
            for case, state in states.items():
                if index < len(state['candidates']):
                    pending.append((case, state['candidates'][index]))
        summaries = {}; last_report = 0.; completed = 0
        activity_lock = threading.Lock(); active = {case: 0 for case in states}; peak = active.copy()
        def process(case, candidate):
            with activity_lock:
                active[case] += 1; peak[case] = max(peak[case], active[case])
            try:
                state = states[case]; obj = state['folder']/'objects'/candidate['id']
                record = process_object(obj, candidate, state['item'], workspace, runtime,
                    config['fitting'], namespace=root.name+'/'+case, prepare_only=prepare_only,
                    seed=seeds[case][candidate['id']], image_assets=state['image_assets'])
                if prepare_only and (obj/'revision_00/fit_local.json').exists():
                    inspection_images(obj/'revision_00')
                return record
            finally:
                with activity_lock:
                    active[case] -= 1
        def checkpoint(case, finished):
            state = states[case]; stage = ('prepared' if prepare_only else 'completed') if finished else 'running'
            scene = assemble(state['folder'], state['item'], state['candidates'], state['records'], state['upstream'], [])
            if finished and not prepare_only:
                export(scene, state['folder'], workspace/state['item']['original'])
                if scene['counts']['unresolved_objects']:
                    stage = 'completed_with_unresolved'
            summary = case_report(state['folder'], scene, state['records'], stage)
            state['last_checkpoint'] = time.monotonic()
            if finished:
                summaries[case] = summary
        def progress():
            with activity_lock:
                activity = {'active_objects_by_image': active.copy(), 'peak_objects_by_image': peak.copy()}
            save(root/'progress.json', {'updated_utc': now(), 'completed': completed, 'total': len(pending),
                 'worker_limit': config['fitting']['workers'], **activity,
                 'api': runtime.slots.snapshot() if runtime is not None else None,
                 'cases': {case: {'completed': len(s['records']), 'total': len(s['candidates'])} for case, s in states.items()}})
        progress()
        with ThreadPoolExecutor(max_workers=config['fitting']['workers']) as pool:
            futures = {pool.submit(process, case, candidate): (case, candidate) for case, candidate in pending}
            for future in as_completed(futures):
                case, candidate = futures[future]; oid = candidate['id']; state = states[case]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {'object_id': oid, 'state': 'unresolved', 'error': str(exc),
                              'selected_revision': None, 'history': []}
                    save(state['folder']/'objects'/oid/'runner_error.json', record)
                state['records'][oid] = record; completed += 1
                finished = len(state['records']) == len(state['candidates'])
                if finished or time.monotonic()-state['last_checkpoint'] >= 10:
                    checkpoint(case, finished)
                if finished or time.monotonic()-last_report >= 10:
                    report(root, items, summaries, prepare_only); last_report = time.monotonic()
                progress()
                print(f'{case}/{oid}: {record["state"]}; {completed}/{len(pending)}', flush=True)
        if config.get('styling_enabled') and not prepare_only:
            renders=threading.BoundedSemaphore(config.get('styling',{}).get('render_workers',2))
            pixels=threading.BoundedSemaphore(config.get('styling',{}).get('pixel_workers',4))
            with ThreadPoolExecutor(max_workers=config.get('styling',{}).get('workers',128)) as style_pool, ThreadPoolExecutor(max_workers=config['case_workers']) as case_pool:
                futures={case_pool.submit(append_styling,workspace,state['folder'].parent,state['item'],
                          read(state['folder']/'scene.json'),summaries[case],runtime,config,style_pool,renders,pixels):case for case,state in states.items()}
                for f in as_completed(futures):summaries[futures[f]]=f.result()
        if config.get('arrows_enabled') and not prepare_only:
            from .arrows.runner import append as append_arrows
            renders=threading.BoundedSemaphore(config.get('arrows',{}).get('render_workers',2))
            with ThreadPoolExecutor(max_workers=config.get('arrows',{}).get('workers',128)) as arrow_pool, ThreadPoolExecutor(max_workers=config['case_workers']) as case_pool:
                futures={case_pool.submit(append_arrows,workspace,state['folder'].parent,state['item'],summaries[case],runtime,config,arrow_pool,renders):case for case,state in states.items()}
                for f in as_completed(futures):summaries[futures[f]]=f.result()
        if config.get('image_export_enabled') and not prepare_only:
            from .composition import append as append_images
            with ThreadPoolExecutor(max_workers=config['case_workers']) as case_pool:
                renders=threading.BoundedSemaphore(2)
                futures={case_pool.submit(append_images,workspace,state['folder'].parent,state['item'],summaries[case],config,renders):case for case,state in states.items()}
                for f in as_completed(futures):summaries[futures[f]]=f.result()
        if config.get('texts_enabled') and not prepare_only:
            from .texts.runner import append as append_texts
            from .texts import options as text_options
            opt=text_options(config)
            renders=threading.BoundedSemaphore(opt['render_workers'])
            with ThreadPoolExecutor(max_workers=opt['workers']) as text_pool, ThreadPoolExecutor(max_workers=config['case_workers']) as case_pool:
                futures={case_pool.submit(append_texts,workspace,state['folder'].parent,state['item'],summaries[case],runtime,config,text_pool,renders):case for case,state in states.items()}
                for f in as_completed(futures):summaries[futures[f]]=f.result()
        save(root/'audit.json', audit(root, config))
        progress()
        return report(root, items, summaries, prepare_only)
