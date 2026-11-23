#!/usr/bin/env python3
"""Additive tiled-SAM discovery AFTER a completed bbox frontend case.

Historical identities are immutable. Every SAM mask receives a recorded
disposition; touching ANY tile boundary excludes that observation. Overlapping
tiles can still supply another complete observation of the same physical ink.
"""
import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from core import area, intersection, inside, gap, union, crop_box
from evidence import frame, save, get_sam, mask, mask_footprint, project, local_box, overlay_masks
from pipeline_frames import Crop
from protocol import KINDS, box as validate_box
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
from visual_labels import render

SYSTEM = ('Inspect the actual scientific-figure pixels. Decide whether each new SAM proposal '
          'is a genuinely missing diagram arrow or line. Existing identities are read-only. '
          'Use absolute TARGET pixels and compact JSON. Backend performs coordinate transforms.')


def now():
    return datetime.now(timezone.utc).isoformat()


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-8, area(a) + area(b) - inter)


def edge_sides(bounds, size, margin=2):
    """Inclusive mask centers; even an original-image outer edge is excluded."""
    return [name for name, touched in zip(
        ('left', 'top', 'right', 'bottom'),
        (bounds[0] <= margin, bounds[1] <= margin,
         bounds[2] >= size[0]-1-margin, bounds[3] >= size[1]-1-margin)) if touched]


def ink_coverage(mrow, owners, padding=1):
    """BBox overlap is a nomination filter, not a semantic accuracy claim."""
    view = frame(mrow['frame'])
    raw = mask(mrow['mask_path'])
    covered = np.zeros(raw.shape, dtype=bool)
    hits = []
    for owner in owners:
        b = owner['bbox_native']
        padded = [b[0]-padding, b[1]-padding, b[2]+padding, b[3]+padding]
        if intersection(padded, mask_footprint(mrow)) <= 0:
            continue
        x0, y0, x1, y1 = local_box(padded, view)
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(raw.shape[1], math.ceil(x1)+1), min(raw.shape[0], math.ceil(y1)+1)
        if x1 > x0 and y1 > y0:
            count = int(raw[y0:y1, x0:x1].sum())
            if count:
                hits.append({'id': owner['id'], 'display_id': owner.get('display_id'),
                             'fraction': count/max(1, int(raw.sum()))})
                covered[y0:y1, x0:x1] = True
    return float((raw & covered).sum()/max(1, raw.sum())), hits


def same_ink(a, b, threshold=.85):
    """Require pixel agreement; nearby parallel strokes are separate proposals."""
    if intersection(a['bbox_native'], b['bbox_native']) <= 0:
        return False
    ia, ib = a['_ink'], b['_ink']
    if not len(ia) or not len(ib):
        return False
    if min(len(ia), len(ib))/max(len(ia), len(ib)) < .55:
        return False
    return len(np.intersect1d(ia, ib, assume_unique=True))/min(len(ia), len(ib)) >= threshold


def serializable(row):
    return {k: v for k, v in row.items() if not k.startswith('_')}


def local_view(size, bounds, cfg):
    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    edge = max(cfg['local_min_edge'], min(cfg['local_max_edge'], round(max(w, h)*cfg['local_scale'])))
    dims = tuple(max(2, round(v*edge/max(w, h))) for v in (w, h))
    return Crop(tuple(size), tuple(bounds), dims, cfg['local_scale'], cfg['local_max_edge'])


def prepare(root, source, case_ids, report_dir):
    if root.exists():
        raise RuntimeError('Fresh experiment directory required')
    root.mkdir(parents=True)
    cfg = read(source/'config.json')
    cfg.update(source_run=str(source), review_output_dir=str(report_dir),
               protocol_version='post_tile_sam_v16', api_concurrency=2,
               case_workers=1, sam_workers=1, review_workers=2,
               boundary_margin_target_px=2, covered_fraction=.90,
               existing_bbox_padding_native_px=1, candidate_dedup_ink_fraction=.85,
               readonly_neighbor_limit=3, readonly_neighbor_gap_native_px=32,
               review_crop_padding_native_px=48, max_review_proposals_per_case=None,
               stage='append_only_after_finished_case')
    assert cfg['max_tokens'] == 113000 and 'reasoning_effort' not in cfg
    write(root/'config.json', cfg)
    write(root/'cases.json', case_ids)
    shutil.copytree(Path(__file__).parent, root/'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'constraints_snapshot', root/'constraints_snapshot')
    (root/'constraints_snapshot/user_post_sam_20261025.txt').write_text(
        '用户要求：现有管线结束后再补充；增加分块SAM；图像边缘可能截断的候选直接去掉；'
        '根据没有被发现过的bbox判断是否添加进箭头清单。已有身份不修改。\n')
    sources = []
    for cid in case_ids:
        original = source/'cases'/cid/'result.json'
        if not (source/'cases'/cid/'finished.json').exists():
            raise RuntimeError('Source case not finished: '+cid)
        shutil.copytree(source/'inputs'/cid, root/'inputs'/cid)
        dst = root/'cases'/cid
        dst.mkdir(parents=True)
        shutil.copy2(original, dst/'baseline_result.json')
        sources.append({'case_id': cid, 'source': str(original), 'sha256': sha256(original),
                        'input_source_sha256': sha256(source/'inputs'/cid/'source.png')})
    write(root/'baseline_sources.json', sources)
    (root/'PROTOCOL.md').write_text(
        '# 后置分块 SAM 补漏实验\n\n'
        '固定 v15 已完成结果，只追加新身份。复用其重叠分块和1080–2048输入，SAM分类line/block_arrow。'
        '所有触及实际分块边缘2像素的mask先排除，包括原图外边界。全图SAM不参与新增。\n\n'
        '候选mask有90%以上像素落入已有身份框（原图1像素余量）则记录已覆盖；其余候选经跨块像素去重，'
        '每次一个新候选及至多3个只读邻居交给GLM确认。bbox覆盖只是筛选启发式，不能证明语义覆盖。'
        'SAM类别只是证据，不作为最终形态分类。\n\n'
        '模型可判add/covered/reject/uncertain；add必须以SAM为像素锚点，不得搬到附近另一箭头。'
        '补充结果串行检查与既有/已新增身份的重复，歧义保留待审。无候选截断上限。'
        '源对象逐字段保持，另存新增和合并结果。默认thinking，113000上限；SAM1、GLM2、图1。\n')
    files = [p for directory in ('code', 'inputs', 'constraints_snapshot') for p in (root/directory).rglob('*') if p.is_file()]
    files += list((root/'cases').glob('*/baseline_result.json'))
    files += [root/'config.json', root/'cases.json', root/'baseline_sources.json', root/'PROTOCOL.md']
    write(root/'manifest.json', {'created_at': now(), 'files': {str(p.relative_to(root)): sha256(p) for p in files}})
    write(root/'progress.json', {'state': 'prepared', 'finished': 0, 'total': len(case_ids), 'updated_at': now()})
    publish(root)


def scan_case(root, cid, cfg):
    c = root/'cases'/cid
    if (c/'scan.json').exists():
        return read(c/'scan.json')
    started = time.time()
    baseline = read(c/'baseline_result.json')['objects']
    packet = read(root/'inputs'/cid/'views.json')
    native_size = tuple(packet['native_size'])
    native_frame = Crop(native_size, (0, 0, *native_size), native_size)
    candidates, audits, errors = [], [], []
    for v in packet['views']:
        if v['view_id'] == 'G':
            continue
        vid = v['view_id']
        view = frame(v['transform'])
        image = root/v['path']
        if sha256(image) != v['sha256']:
            raise ValueError('Frozen tile hash differs')
        try:
            cache=None
            if cfg.get('post_sam_cache_run'):
                prior=Path(cfg['post_sam_cache_run'])/'cases'/cid/'tile_sam'/vid/'evidence.json'
                if prior.exists():cache=Path(read(prior)['prediction_path']).parent
            masks = get_sam(image, c/'tile_sam'/vid, cfg, view, 'T'+vid,cache)
        except Exception as exc:
            errors.append({'view_id': vid, 'error_type': type(exc).__name__, 'error': str(exc)})
            write(c/'tile_sam'/vid/'failed.json', errors[-1])
            continue
        for m in masks:
            audit = {'mask_id': m['id'], 'view_id': vid, 'category': m['category'],
                     'score': m['score'], 'bbox_native': mask_footprint(m)}
            sides = edge_sides(m['bbox_mask'], view.input_size, cfg['boundary_margin_target_px'])
            if sides:
                audits.append({**audit, 'status': 'excluded_tile_boundary', 'sides': sides})
                continue
            fraction, owners = ink_coverage(m, baseline, cfg['existing_bbox_padding_native_px'])
            audit.update(existing_ink_coverage=fraction, overlapping_owners=owners)
            if fraction >= cfg['covered_fraction']:
                audits.append({**audit, 'status': 'covered_by_existing_bbox'})
                continue
            ink = np.flatnonzero(project(m, native_frame))
            r = {'id': m['id'], 'kind': 'sam_proposal', 'bbox_native': mask_footprint(m),
                 'sam_category': m['category'], 'score': m['score'], 'sam': m,
                 'source_mask_ids': [m['id']], 'existing_ink_coverage': fraction,
                 'overlapping_owners': owners, '_ink': ink, 'uncertain': False, 'extent': 'complete'}
            prior = next((p for p in candidates if same_ink(r, p, cfg['candidate_dedup_ink_fraction'])), None)
            if prior is not None:
                prior['source_mask_ids'].append(m['id'])
                audits.append({**audit, 'status': 'duplicate_tile_observation', 'representative': prior['id']})
            else:
                candidates.append(r)
                audits.append({**audit, 'status': 'nominated'})
        write(c/'scan_progress.json', {'last_view': vid, 'masks_seen': len(audits),
                                      'candidate_count': len(candidates), 'updated_at': now()})
    candidates.sort(key=lambda r: (r['bbox_native'][1], r['bbox_native'][0], r['id']))
    for i, r in enumerate(candidates, 1):
        r['representative_mask_id'] = r['id']
        r['id'] = f'N{i:04d}'
    result = {'candidates': [serializable(r) for r in candidates], 'mask_audit': audits,
              'errors': errors, 'seconds': time.time()-started,
              'tile_count': len(packet['views'])-1}
    write(c/'scan.json', result)
    global_v = frame(packet['views'][0]['transform'])
    render(root/'inputs'/cid/'G/target.png', result['candidates'], c/'candidates.png', global_v)
    print(json.dumps({'event': 'scan_complete', 'case_id': cid, 'candidates': len(candidates),
                      'mask_count': len(audits), 'errors': len(errors)}, ensure_ascii=False), flush=True)
    publish(root)
    return result


def validate_review(ans, size, neighbor_ids):
    if not isinstance(ans, dict) or ans.get('decision') not in {'add', 'covered', 'reject', 'uncertain'}:
        raise ValueError('decision must be add/covered/reject/uncertain')
    if not isinstance(ans.get('reason'), str) or not ans['reason'].strip():
        raise ValueError('Visible reason required')
    if ans['decision'] == 'covered' and ans.get('existing_id') not in neighbor_ids:
        raise ValueError('covered requires one supplied read-only existing_id')
    if ans['decision'] == 'add':
        if ans.get('kind') not in KINDS or ans.get('extent') != 'complete':
            raise ValueError('Add requires a valid kind and complete visual extent')
        if not isinstance(ans.get('locator'), str) or not ans['locator'].strip():
            raise ValueError('Object locator required')
        validate_box(ans.get('bbox'), size)
    return ans


def review_one(root, cid, candidate, cfg, model):
    out = root/'cases'/cid/'review'/candidate['id']
    if (out/'decision.json').exists():
        return read(out/'decision.json')
    baseline = read(root/'cases'/cid/'baseline_result.json')['objects']
    native = Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    cb = crop_box([candidate['bbox_native']], native.size, cfg['review_crop_padding_native_px'])
    view = local_view(native.size, cb, cfg)
    nearby = [o for o in baseline if gap(o['bbox_native'], candidate['bbox_native']) <= cfg['readonly_neighbor_gap_native_px']]
    nearby.sort(key=lambda o: (-inside(candidate['bbox_native'], o['bbox_native']), gap(o['bbox_native'], candidate['bbox_native']), o['id']))
    neighbors = nearby[:cfg['readonly_neighbor_limit']]
    save(view.image(native), out/'target.png')
    render(out/'target.png', [candidate], out/'boxes.png', view, context=neighbors, mode='request')
    selected = {**candidate['sam'], 'alias': 'M1', 'projected': project(candidate['sam'], view),
                'bbox_target': local_box(candidate['bbox_native'], view)}
    overlay_masks(out/'target.png', [selected], out/'sam.png')
    refs = [{'id': f'r{i+1}', 'kind': o['kind'], 'bbox': local_box(o['bbox_native'], view),
             'locator': o.get('locator', '')} for i, o in enumerate(neighbors)]
    alias_map = {f'r{i+1}': o['id'] for i, o in enumerate(neighbors)}
    q = {'task': '现有箭头发现已结束。仅判断新增SAM候选1是否是真实且尚未发现的箭头/独立线条。',
         'images': 'GLOBAL全图语义参考；TARGET干净局部；BOXES中1为新增候选，r开头为已有身份；SAM只高亮候选1像素。',
         'target_size_px': list(view.input_size),
         'rules': [
             '这是追加补漏，不修改、移动、删除或重新发现已有身份。只判断候选1，不扫描别处。',
             '以SAM高亮的实际笔画为身份锚点。不能把它移动到上下左右另一支相似箭头。bbox含杆与头，允许稍松。',
             '已有bbox只是范围参考；对照实际笔画判断same/网络成员。真实独立箭头即使与已有框相交也可新增。',
             '普通箭头与独立无头线属于范围；连续多头链可serial_chain；真实共享主干可shared_network；密集网络或线束按整体dense_network。',
             '宽块箭头属于范围；非箭头闭合形状边框、格边、文字笔画、立体形状构造边、分子内部键、照片纹理排除。',
             '网络组内部已有成员属于covered，不另添一条；网络外部独立输入输出不能吞入网络。',
             '如果SAM仅覆盖残段，必须能在TARGET看清其完整对象才能add；不能确定完整性或新旧归属用uncertain。',
             'decision=add时只输出候选1对应的一个完整身份；covered引用r编号；reject说明非范围依据。理由一短句。'],
         'candidate': {'id': '1', 'bbox': local_box(candidate['bbox_native'], view), 'sam_category_hint': candidate['sam_category']},
         'existing_readonly': refs,
         'output_schema': {'decision': 'add|covered|reject|uncertain', 'reason': '可见依据',
                           'existing_id': 'r1 or null', 'kind': 'single|serial_chain|shared_network|dense_network|dense_region or null',
                           'extent': 'complete or null', 'bbox': [0, 0, 10, 10], 'locator': '短对象描述 or null'}}
    write(out/'packet.json', {'view': view.metadata(native.size), 'candidate_id': candidate['id'],
                            'existing_aliases': alias_map, 'nearby_count': len(nearby)})
    started = time.time()
    ans = model.ask(out/'request', q, [('GLOBAL', root/'inputs'/cid/'G/target.png'),
                    ('TARGET', out/'target.png'), ('BOXES', out/'boxes.png'), ('SAM', out/'sam.png')],
                    lambda x: validate_review(x, view.input_size, set(alias_map)))
    result = {'candidate_id': candidate['id'], 'model_decision': ans, 'seconds': time.time()-started}
    if ans['decision'] == 'add':
        b = view.to_global(ans['bbox'][:2])+view.to_global(ans['bbox'][2:])
        m = candidate['bbox_native']
        anchor_fraction = inside(m, b)
        result['anchor_fraction'] = anchor_fraction
        if anchor_fraction < .80:
            result.update(disposition='pending_anchor_mismatch', reason='Accepted bbox does not retain the SAM candidate ink extent')
        elif ((ans['bbox'][0] <= 1 and cb[0] > 0) or (ans['bbox'][1] <= 1 and cb[1] > 0)
              or (ans['bbox'][2] >= view.input_size[0]-2 and cb[2] < native.width)
              or (ans['bbox'][3] >= view.input_size[1]-2 and cb[3] < native.height)):
            result.update(disposition='pending_review_crop_boundary', reason='Accepted extent touches an internal review crop boundary')
        else:
            result.update(disposition='proposed_add', object={
                'id': 'post_sam/'+candidate['id'], 'kind': ans['kind'], 'bbox_native': b,
                'locator': ans['locator'], 'extent': 'complete', 'uncertain': ans['kind']=='dense_region',
                'reason': ans['reason'], 'source_ids': candidate['source_mask_ids'],
                'bbox_source': 'model_with_sam_anchor', 'sam_category_hint': candidate['sam_category'],
                'geometry_ready': False, 'handoff_stage': 'discovery_only_not_geometry_validated'})
    else:
        result['disposition'] = ans['decision']
        if ans['decision'] == 'covered':
            result['existing_original_id'] = alias_map[ans['existing_id']]
    write(out/'decision.json', result)
    print(json.dumps({'event': 'review', 'case_id': cid, 'candidate': candidate['id'],
                      'disposition': result['disposition'], 'seconds': result['seconds']}, ensure_ascii=False), flush=True)
    return result


def append_objects(baseline, proposals):
    """Immutable prefix; additions cannot silently alter historical identities."""
    objects = copy.deepcopy(baseline)
    existing_numbers = [int(o['display_id'][1:]) for o in baseline if str(o.get('display_id', '')).startswith('I')]
    next_number = max(existing_numbers, default=0)+1
    for p in proposals:
        o = copy.deepcopy(p)
        o['display_id'] = f'I{next_number:04d}'
        objects.append(o)
        next_number += 1
    assert objects[:len(baseline)] == baseline
    return objects


def execute_case(root, cid, cfg, model, stage):
    c = root/'cases'/cid
    if (c/'finished.json').exists():
        return read(c/'result.json')
    write(c/'status.json', {'phase': 'tile_sam', 'updated_at': now()})
    scan = scan_case(root, cid, cfg)
    if stage == 'scan':
        return None
    write(c/'status.json', {'phase': 'semantic_review', 'candidates': len(scan['candidates']), 'updated_at': now()})
    decisions, failures = [], []
    with ThreadPoolExecutor(max_workers=cfg['review_workers']) as pool:
        jobs = {pool.submit(review_one, root, cid, p, cfg, model): p for p in scan['candidates']}
        for job in as_completed(jobs):
            p = jobs[job]
            try:
                decisions.append(job.result())
            except Exception as exc:
                item = {'candidate_id': p['id'], 'disposition': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
                failures.append(item)
                write(c/'review'/p['id']/'failed.json', item)
    baseline = read(c/'baseline_result.json')['objects']
    additions, final_audit = [], []
    candidate_index = {p['id']: p for p in scan['candidates']}
    for d in sorted(decisions, key=lambda x: x['candidate_id']):
        if d['disposition'] != 'proposed_add':
            final_audit.append(d)
            continue
        o = d['object']
        # Numerical checks only defer conflicting additions; they never edit old rows.
        candidate = candidate_index[d['candidate_id']]
        frac, owners = ink_coverage(candidate['sam'], baseline+additions, cfg['existing_bbox_padding_native_px'])
        same_boxes = [e for e in baseline+additions if iou(o['bbox_native'], e['bbox_native']) >= .75]
        if frac >= cfg['covered_fraction'] or same_boxes:
            final_audit.append({**d, 'disposition': 'pending_existing_or_new_overlap',
                                'post_coverage': frac, 'overlapping_owners': owners,
                                'same_bbox_ids': [e['id'] for e in same_boxes]})
        else:
            additions.append(o)
            final_audit.append({**d, 'disposition': 'added'})
    combined = append_objects(baseline, additions)
    issues = failures + scan['errors']
    result = {'case_id': cid, 'state': 'completed_with_issues' if issues else 'completed',
              'baseline_count': len(baseline), 'candidate_count': len(scan['candidates']),
              'added_count': len(additions), 'output_identity_count': len(combined),
              'objects': combined, 'additions': combined[len(baseline):], 'decisions': final_audit,
              'pending_count': sum(d['disposition'].startswith('pending') or d['disposition']=='uncertain' for d in final_audit)+len(failures),
              'issues': issues, 'source_baseline_sha256': sha256(c/'baseline_result.json'),
              'source_objects_unchanged': combined[:len(baseline)] == baseline,
              'formal_accuracy_measured': False, 'completed_at': now()}
    write(c/'result.json', result)
    inp = root/'inputs'/cid
    view = frame(read(inp/'views.json')['views'][0]['transform'])
    render(inp/'G/target.png', combined, c/'overlay.png', view)
    render(inp/'G/target.png', result['additions'], c/'additions.png', view)
    write(c/'finished.json', {'completed_at': now(), 'added': len(additions)})
    write(c/'status.json', {'phase': 'finished', 'state': result['state'], 'updated_at': now()})
    publish(root)
    return result


def publish(root):
    if read(root/'config.json').get('integrated_mode'):return
    cfg = read(root/'config.json')
    dest = Path(cfg['review_output_dir'])
    (dest/'images').mkdir(parents=True, exist_ok=True)
    lines = ['# 后置分块 SAM 补漏对照', '', '固定v15最终清单，只追加经确认的新身份。短编号延续原清单。完成数量不代表召回率。', '',
             '[实验方案]('+str(root/'PROTOCOL.md')+') · [运行结果]('+str(root/'RESULTS.md')+')', '']
    summaries = []
    for cid in read(root/'cases.json'):
        c = root/'cases'/cid
        baseline = read(c/'baseline_result.json')
        inp = root/'inputs'/cid
        before = dest/'images'/f'{cid}_before.png'
        if not before.exists():
            shutil.copy2(Path(cfg['source_run'])/'cases'/cid/'overlay.png', before)
        lines += ['## '+cid, '']
        if (c/'result.json').exists():
            r = read(c/'result.json')
            summaries.append({k: r[k] for k in ('case_id', 'state', 'baseline_count', 'candidate_count', 'added_count', 'pending_count', 'source_objects_unchanged')})
            shutil.copy2(c/'overlay.png', dest/'images'/f'{cid}_after.png')
            shutil.copy2(c/'additions.png', dest/'images'/f'{cid}_additions.png')
            lines += [f'原身份 {r["baseline_count"]}；送审候选 {r["candidate_count"]}；新增 {r["added_count"]}；待确认 {r["pending_count"]}。', '',
                      '| v15 原结果 | 后置补漏结果 |', '|---|---|',
                      f'| ![](images/{cid}_before.png) | ![](images/{cid}_after.png) |', '',
                      f'[仅新增框](images/{cid}_additions.png) · [逐候选决策]({c}/result.json)', '']
        else:
            count = len(read(c/'scan.json')['candidates']) if (c/'scan.json').exists() else None
            lines += [f'尚未完成；SAM送审候选：{count if count is not None else "扫描中/待启动"}。', '', f'![](images/{cid}_before.png)', '']
    (dest/'后置分块SAM补漏对照_4例.md').write_text('\n'.join(lines))
    write(root/'summary.json', {'cases': summaries, 'finished': len(summaries), 'total': len(read(root/'cases.json')), 'updated_at': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--stage', choices=('scan', 'all'), default='all')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.prepare:
        prepare(root, args.source.resolve(), args.cases, args.report_dir.resolve())
        return
    if Path(__file__).resolve().parent != root/'code':
        raise RuntimeError('Execute the frozen snapshot')
    lock = (root/'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel, digest in read(root/'manifest.json')['files'].items():
        if sha256(root/rel) != digest:
            raise RuntimeError('Frozen artifact differs: '+rel)
    cfg = read(root/'config.json')
    model = Model(cfg, run_dir=root, system=SYSTEM)
    write(root/'process.json', {'pid': os.getpid(), 'started_at': now(), 'stage': args.stage})
    records = []
    for cid in args.cases or read(root/'cases.json'):
        write(root/'progress.json', {'state': 'running', 'case_id': cid, 'finished': len(records), 'updated_at': now()})
        try:
            result = execute_case(root, cid, cfg, model, args.stage)
            records.append({'case_id': cid, 'state': result['state'] if result else 'scanned'})
        except Exception as exc:
            import traceback
            item = {'case_id': cid, 'state': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
            write(root/'cases'/cid/'failed.json', item)
            records.append(item)
    write(root/'progress.json', {'state': 'scanned' if args.stage=='scan' else 'finished', 'records': records, 'updated_at': now()})
    publish(root)


if __name__ == '__main__':
    main()
