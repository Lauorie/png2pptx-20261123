"""Unified experimental localization -> fitting -> checks -> scene/export runner."""
import copy
import fcntl
import json
import shutil
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from . import batch, image_bridge
from .common import read, save, sha, now, usage, valid_box
from .fitting.control import process_object, safe_id
from .fitting.context import identity_context
from .fitting.coverage import review as coverage_review
from .fitting.scene import assemble, export
from .fitting.coordinates import coordinate_mode, NORM1000_CONVENTION

IMPORT_FILES = ['geometry.json', 'summary.json', 'dedup/image_boxes_original.json',
                'dedup/decisions.json', 'dedup/removed.json', 'shapes/history.json',
                'shapes/geometry.json', 'shapes/global_overlay.png', 'shapes/enhanced_overlay.png',
                'shapes/summary.json', 'images/status.json', 'images/inventory.json',
                'final_overlay.png', 'image_overlay.png', 'comparison.png']


def source_manifest(source, items):
    if source is None:
        return None
    m = read(source/'manifest.json'); old = {x['id']: x for x in m['images']}
    hashes = {'manifest.json': sha(source/'manifest.json')}
    for item in items:
        if old.get(item['id']) != item:
            raise ValueError('Localization input provenance changed: '+item['id'])
        case = source/'cases'/item['id']
        for name in ['geometry.json', 'summary.json', 'dedup/image_boxes_original.json']:
            if not (case/name).is_file():
                raise ValueError('Localization missing required artifact: '+str(case/name))
        for name in IMPORT_FILES:
            p = case/name
            if p.is_file():
                hashes[str(p.relative_to(source))] = sha(p)
    return {'path': str(source.resolve()), 'sha256': hashes, 'reused_stages': ['shape_localization', 'image_detection', 'image_iou_dedup']}


def import_localization(folder, source, item, provenance):
    folder.mkdir(parents=True, exist_ok=True)
    prefix = 'cases/'+item['id']+'/'
    for relative, digest in provenance['sha256'].items():
        if not relative.startswith(prefix):
            continue
        src = source/relative; dst = folder/relative[len(prefix):]
        if sha(src) != digest:
            raise ValueError('Localization changed during import')
        if dst.exists():
            if sha(dst) != digest:
                raise ValueError('Imported localization artifact changed')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    save(folder/'localization_source.json', {'path': str(source/'cases'/item['id']),
        'usage_note': 'Original localization usage stays in source summary; current usage counts new calls only.'})
    return read(folder/'summary.json')


def validate_candidates(data, item):
    if data['image_size'] != item['source_size'] or data.get('coordinate_system') != 'image_pixel_centers':
        raise ValueError('Candidate coordinate frame mismatch')
    ids = [safe_id(o['id']) for o in data['objects']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate ID')
    if any(not valid_box(o['bbox_2d'], item['source_size']) for o in data['objects']):
        raise ValueError('Invalid source candidate bbox')


def case_report(folder, scene, records, state):
    used = usage(folder); save(folder/'usage.json', used)
    summary = {'id': scene['case_id'], 'state': state, **scene['counts'], 'usage': used['totals'],
               'coverage': scene['coverage'], 'visual_accuracy_verified': False}
    save(folder/'summary.json', summary)
    lines = ['# 逐图拟合结果', '', f'状态：`{state}`。模型通过与数值拟合成功均不等于人工视觉验收。', '',
             '[Scene](scene.json) · [最终 PNG](geometry_final.png) · [最终 SVG](geometry_final.svg) · [最终 PPTX](geometry_final.pptx) · [仅模型通过 PPTX](geometry_accepted.pptx)', '',
             '最终结果保留每个对象最后一次有可用几何的修订；预算耗尽、未决或拒绝状态不影响收录，原状态与停止原因继续保留。', '',
             '![整图拟合叠加](fit_overlay.png)', '', '| 对象 | 状态 | 当前几何修订 | 查看 |', '| --- | --- | --- | --- |']
    for oid, obj in scene['objects'].items():
        rev = obj.get('selected_revision'); link = f'[记录](objects/{oid}/record.json)'
        if rev:
            link += f' · [四图对照](objects/{oid}/{rev}/comparison.png)'
        lines.append(f'| {oid} | {obj["state"]} | {rev or "—"} | {link} |')
    for oid, obj in scene['objects'].items():
        rev = obj.get('selected_revision')
        if rev:
            d = f'objects/{oid}/{rev}'
            status_path = folder/d/'status.json'
            call = read(status_path).get('observation_call') if status_path.exists() else None
            call = call or 'calls/observe_00'
            raw_link = f'{d}/{call}/content.txt'
            if (folder/d/'reused_revision.json').exists() and read(folder/d/'reused_revision.json')['fit_reused']:
                origin = Path(read(folder/d/'reused_revision.json')['path'])
                raw_link = os.path.relpath(origin/call/'content.txt', folder)
            lines += ['', f'## {oid}', '', f'[Prompt]({d}/prompt.json) · [原始输出]({raw_link}) · [原图坐标拟合]({d}/fit_original.json)', '',
                      f'![{oid} 局部拟合]({d}/comparison.png)']
    (folder/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return summary


def run_case(workspace, root, item, runtime, config, fit_pool, source, provenance, prepare_only, style_render_slots=None, style_pixel_slots=None):
    folder = root/'cases'/safe_id(item['id']); fitted = folder/'fitting'; fitted.mkdir(parents=True, exist_ok=True)
    for name, key in [('original', 'source_sha256'), ('input', 'input_sha256')]:
        if sha(workspace/item[name]) != item[key]:
            raise ValueError('Input image changed')
    upstream = (import_localization(folder, source, item, provenance) if source is not None
                else batch.case(root, item, workspace, runtime))
    data = read(folder/'geometry.json'); validate_candidates(data, item)
    candidates = copy.deepcopy(data['objects'])
    localized = copy.deepcopy(candidates)
    localization_history = read(folder/'shapes/history.json') if (folder/'shapes/history.json').exists() else []
    assets = read(folder/'dedup/image_boxes_original.json')['assets'] if (folder/'dedup/image_boxes_original.json').exists() else []
    selection = config.get('fit_selection')
    if selection is not None:
        ids = {o['object_id'] for o in selection if o['case_id'] == item['id']}
        if not ids <= {o['id'] for o in candidates}:
            raise ValueError('Selected candidate missing after localization')
        candidates = [o for o in candidates if o['id'] in ids]
    options = config['fitting']; records = {}; coverage = []
    stage_runtime = runtime.for_stage(shape_schema_retries=options.get('context_retries', 1)) if runtime is not None else None
    def checkpoint(state):
        save(fitted/'candidates.json', {'objects': candidates, 'upstream_geometry_sha256': sha(folder/'geometry.json')})
        scene = assemble(fitted, item, candidates, records, upstream, coverage)
        if state != 'running':
            export(scene, fitted, workspace/item['original'])
        return scene, case_report(fitted, scene, records, state)
    pending = candidates.copy(); last_checkpoint = 0.
    for index in range(options['coverage_rounds']+1):
        jobs = {fit_pool.submit(process_object, fitted/'objects'/c['id'], c, item, workspace, stage_runtime, options,
                                namespace=root.name+'/'+item['id'], prepare_only=prepare_only,
                                image_assets=assets,
                                input_context=identity_context(c, localized + [o for o in candidates if o['id'] not in {q['id'] for q in localized}],
                                                               assets, localization_history, options['crop_padding'],
                                                               exclude_shape_relations=options.get('exclude_shape_neighbor_relations', ()))): c for c in pending}
        for future in as_completed(jobs):
            candidate = jobs[future]; oid = candidate['id']
            try:
                record = future.result()
            except Exception as exc:
                record = {'object_id': oid, 'state': 'unresolved', 'error': str(exc), 'selected_revision': None, 'history': []}
                save(fitted/'objects'/oid/'runner_error.json', record)
            records[oid] = record
            if time.monotonic()-last_checkpoint >= 10:
                checkpoint('running'); last_checkpoint = time.monotonic()
            print(json.dumps({'case': item['id'], 'object': oid, 'state': record['state'], 'processed': len(records), 'candidates': len(candidates)}, ensure_ascii=False), flush=True)
        scene, summary = checkpoint('prepared' if prepare_only else 'fitted')
        if prepare_only or index == options['coverage_rounds']:
            break
        try:
            assets = read(folder/'dedup/image_boxes_original.json')['assets']
            review = coverage_review(scene, item, workspace, fitted/f'coverage_{index:02d}', stage_runtime,
                                     options, candidates, assets, config['iou_threshold'], index)
            coverage.append(review); pending = review['new_objects']; candidates.extend(pending)
            if not pending:
                break
        except Exception as exc:
            error = {'round': index, 'error': str(exc), 'status': 'unresolved'}
            save(fitted/f'coverage_{index:02d}'/'runner_error.json', error); coverage.append(error); break
    state = 'prepared' if prepare_only else 'completed'
    # Execution completion does not erase fitting, upstream or coverage uncertainty.
    if not prepare_only and (scene['counts']['unresolved_objects'] or upstream.get('errors') or upstream.get('local_failures')
            or upstream.get('image_conflict_deferred_ids') or data.get('dedup_provenance',{}).get('deferred_ids')
            or upstream.get('image_unresolved_ids') or upstream.get('image_coverage_status') != 'complete'
            or any(r.get('error') or r.get('image_conflict_deferred_ids') or r.get('review', {}).get('uncertain') or r.get('review', {}).get('possible_duplicates')
                   or r.get('review', {}).get('status') == 'uncertain' or r.get('new_objects') for r in coverage)):
        state = 'completed_with_unresolved'
    scene, summary = checkpoint(state)
    if config.get('styling_enabled') and not prepare_only:
        summary = append_styling(workspace, folder, item, scene, summary, runtime, config, fit_pool,
                                 style_render_slots or threading.BoundedSemaphore(1),style_pixel_slots)
    if config.get('arrows_enabled') and not prepare_only:
        from .arrows.runner import append
        summary=append(workspace,folder,item,summary,runtime,config,fit_pool,
                       style_render_slots or threading.BoundedSemaphore(1))
    if config.get('image_export_enabled') and not prepare_only:
        from .composition import append as append_images
        summary=append_images(workspace,folder,item,summary,config,style_render_slots or threading.BoundedSemaphore(1))
    if config.get('texts_enabled') and not prepare_only:
        from .texts.runner import append as append_texts
        summary=append_texts(workspace,folder,item,summary,runtime,config,fit_pool,
                             style_render_slots or threading.BoundedSemaphore(1))
    return summary


def append_styling(workspace, folder, item, scene, summary, runtime, config, pool, render_slots,pixel_slots=None):
    """A style failure cannot erase completed geometry or its review states."""
    from .styling import options
    from .styling.controller import run_case as style_case
    assets_path=folder/'dedup/image_boxes_original.json'
    assets=read(assets_path)['assets'] if assets_path.exists() else []
    try:
        result=style_case(scene,workspace/item['original'],folder/'styling',runtime,options(config),pool,render_slots,assets=assets,pixel_slots=pixel_slots)
    except Exception as exc:
        result={'state':'failed','error':str(exc)};save(folder/'styling/runner_error.json',result)
    summary={**summary,'styling':result}
    save(folder/'fitting/summary.json',summary)
    return summary


def report(root, items, summaries, prepare_only):
    rows = [summaries.get(i['id'], {'id': i['id'], 'state': 'pending'}) for i in items]
    used = usage(root/'cases'); save(root/'usage.json', used)
    state = 'running' if len(summaries) < len(items) else 'prepared' if prepare_only else 'completed'
    if len(summaries) == len(items) and any(r['state'] in ('failed', 'completed_with_unresolved') for r in rows):
        state = 'completed_with_unresolved' if not prepare_only else 'preparation_with_failures'
    summary = {'state': state, 'cases': rows, 'totals': {k: sum(r.get(k, 0) for r in rows) for k in
                ['candidates', 'terminal_objects', 'accepted_by_model', 'successful_parts', 'failed_parts', 'unresolved_objects']},
               'usage': used['totals'], 'updated_utc': now(), 'visual_accuracy_verified': False}
    save(root/'summary.json', summary)
    lines = ['# Shape 完整拟合 pipeline', '', f'状态：`{state}`。当前运行用量：输入 {used["totals"]["input_tokens"]:,}，输出 {used["totals"]["output_tokens"]:,}（包含 reasoning）。', '',
             '复用定位的来源与哈希在 manifest.json；历史定位 token 不计入本次新调用。每次观察处理一个对象的全部 parts；修改和重裁剪分别留存。', '',
             'geometry_final 为最终导出，包含各状态对象已有的可用几何；仅模型通过的筛选视图单独保存。模型接受数不是视觉准确率，原状态、停止原因与预算耗尽记录均保留。', '',
             '| 图 | 状态 | 候选 | 已结束 | 模型通过 | 有效 parts | 未解决对象 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| [{r["id"]}](cases/{r["id"]}/fitting/REPORT.zh-CN.md) | {r["state"]} | '+
                     ' | '.join(str(r.get(k, '—')) for k in ['candidates', 'terminal_objects', 'accepted_by_model', 'successful_parts', 'unresolved_objects'])+' |')
    for r in rows:
        case=root/'cases'/r['id']
        final=next((case/p for p in ('texts/combined.pptx','image_composition/combined.pptx','arrows/combined.pptx','styling/styled.pptx') if (case/p).is_file()),None)
        if final is not None:
            lines+=['',f'[{r["id"]} 最终组合 PPT]({final.relative_to(root)})']
        p = f'cases/{r["id"]}/fitting/fit_overlay.png'
        if (root/p).exists():
            lines += ['', f'## {r["id"]}', '', f'![{r["id"]} 拟合叠图]({p})']
    (root/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return summary


def audit(root, config):
    checked = []
    for p in (root/'cases').glob('*/fitting/objects/*/revision_*/job.json'):
        job = read(p); frame = job['frame']; prompt = read(p.parent/'prompt.json')['prompt']; size = frame['actual_crop_input_size_px']
        for name, expected in job['input_sha256'].items():
            if sha(p.parent/name) != expected:
                raise ValueError('Prepared image/prompt/frame changed')
        if not config['fitting']['crop_min_long_edge'] <= max(size) <= config['fitting']['crop_max_long_edge']:
            raise ValueError('Fitting crop size outside configured range')
        for name in ['clean.png', 'overlay.png']:
            with Image.open(p.parent/name) as image:
                assert list(image.size) == size
        assert coordinate_mode(job) == coordinate_mode(config['fitting'])
        if coordinate_mode(job) == 'norm1000':
            assert prompt['local_coordinates'] == {'convention': NORM1000_CONVENTION}
        else:
            assert prompt['local_coordinates']['width_px'] == size[0] and prompt['local_coordinates']['height_px'] == size[1]
        assert not {'frame', 'candidate', 'bbox_2d', 'global_size', 'identity_bbox_local'} & set(prompt)
        checked.append({'job': str(p.relative_to(root)), 'size': size, 'policy': frame['sizing']['policy']})
    return {'passed': True, 'fitting_inputs': checked, 'observation_requests': len(list((root/'cases').glob('*/fitting/objects/*/revision_*/calls/observe_*/request.json'))),
            'no_image_editing': not list((root/'cases').glob('*/images/calls/editor/*/request.json'))}


def run(workspace, root, config, items, runtime, *, localization_source=None, prepare_only=False):
    if prepare_only and localization_source is None:
        raise ValueError('Prepare-only requires existing localization; it never calls models')
    provenance = source_manifest(localization_source, items)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.freeze(workspace, root, config, items, provenance)
        if runtime is not None:
            image_bridge.configure(runtime)
        summaries = {}; report(root, items, summaries, prepare_only)
        style_render_slots=threading.BoundedSemaphore(config.get('styling',{}).get('render_workers',2))
        style_pixel_slots=threading.BoundedSemaphore(config.get('styling',{}).get('pixel_workers',4))
        with ThreadPoolExecutor(max_workers=config['fitting']['workers']) as fitting_pool, ThreadPoolExecutor(max_workers=config['case_workers']) as cases:
            futures = {cases.submit(run_case, workspace, root, item, runtime, config, fitting_pool,
                                    localization_source, provenance, prepare_only,style_render_slots,style_pixel_slots): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    summaries[item['id']] = future.result()
                except Exception as exc:
                    summaries[item['id']] = {'id': item['id'], 'state': 'failed', 'error': str(exc)}
                    save(root/'cases'/item['id']/'pipeline_error.json', summaries[item['id']])
                report(root, items, summaries, prepare_only)
        save(root/'audit.json', audit(root, config))
        summary = report(root, items, summaries, prepare_only)
        return summary
