"""Add-only SAM-assisted localization: immutable baseline, no correction or reconcile calls."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import threading
import time
import traceback
from supplement_protocol import MissingProtocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import contracts as C
import draft_prompts as D
from frames import global_frame
from handoff import export
from pipeline_frames import Crop, resized_size
from transport.model import Model
from transport.common import sha256
from sam_client import SamClient
from containment_filter import filter_candidates, retained_mask_union

REPORT_LOCK = threading.RLock()
SYSTEM = '检查科学图中箭头和线条的定位遗漏。原图是依据，SAM和已有框都是线索。只输出请求指定的JSON，坐标为当前CLEAN局部图绝对像素，不自行换算。'
SCOPE = ('范围包括普通有头/无头线、曲线、虚线、分支、框内真实网络及块状箭头。普通箭头与块状箭头分开；'
         '排除非箭头闭合形状/容器的边框（包括虚线框残边）、格网形状边、字形和照片纹理。'
         '只沿实际可见笔画，不推导原图遮挡下的连接。SAM可能只分到头、残段、多个对象或误检。')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def save(image, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    image.save(tmp, format='PNG'); tmp.replace(path)


def overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def area(b):
    return max(0., b[2]-b[0])*max(0., b[3]-b[1])


def iou(a, b):
    x = overlap(a, b)
    return x / max(1e-12, area(a)+area(b)-x)


def union(boxes):
    return [min(b[k] for b in boxes) if k < 2 else max(b[k] for b in boxes) for k in range(4)]


def compress(rows, threshold):
    """Complete-link high-IoU observation clusters; never drop a member/mask."""
    groups = []
    for row in sorted(rows, key=lambda r: (-area(r['bbox']), r['id'])):
        target = next((g for g in groups if all(iou(row['bbox'], x['bbox']) >= threshold for x in g)), None)
        if target is None: groups.append([row])
        else: target.append(row)
    return [dict(id=f'S{i:04d}', bbox=union([r['bbox'] for r in g]),
                 member_ids=[r['id'] for r in g],
                 categories=sorted({r['category'] for r in g})) for i, g in enumerate(groups, 1)]


def compression_audit(raw, groups):
    before = sorted(r['id'] for r in raw)
    after = sorted(m for g in groups for m in g['member_ids'])
    if before != after or len(set(before)) != len(before): raise ValueError('Lost/duplicated observation membership')
    lookup = {r['id']: r for r in raw}
    evidence = lambda ids: sorted((x, lookup[x]['mask_sha256']) for x in ids)
    if evidence(before) != evidence(after): raise ValueError('Mask evidence changed')
    return dict(raw_count=len(raw), compressed_count=len(groups), preserved_observations=len(after),
                all_mask_hashes_preserved=True, deleted_mask_pixels=0,
                note='Evidence preservation only; not a semantic correctness or recall score.')


def local_box(b, view):
    return view.to_local(b[:2]) + view.to_local(b[2:])


def native_box(b, view):
    return view.to_global(b[:2]) + view.to_global(b[2:])


def visible_box(b, size):
    return [max(0., min(size[k % 2]-1., x)) for k, x in enumerate(b)]


def view_for(size, bounds, edge=1080):
    b = tuple(int(v) for v in bounds)
    return Crop(tuple(size), b, resized_size((b[2]-b[0], b[3]-b[1]), edge))


def tile_views(size, cfg):
    cols, rows = cfg['tile_columns'], cfg['tile_rows']
    for y in range(rows):
        for x in range(cols):
            w, h = size[0]/cols, size[1]/rows
            pad = cfg['tile_padding']
            b = (max(0, math.floor(x*w-pad*w)), max(0, math.floor(y*h-pad*h)),
                 min(size[0], math.ceil((x+1)*w+pad*w)), min(size[1], math.ceil((y+1)*h+pad*h)))
            yield f'T{y*cols+x+1:02d}', view_for(size, b, cfg['local_long_edge'])


def expand(view, cfg):
    center = [(view.box[k]+view.box[k+2])/2 for k in (0, 1)]
    extent = [v*cfg['context_expansion'] for v in view.size]
    b = [max(0, math.floor(center[k]-extent[k]/2)) for k in (0, 1)]
    b += [min(view.global_size[k], math.ceil(center[k]+extent[k]/2)) for k in (0, 1)]
    return view_for(view.global_size, b, cfg['expanded_long_edge'])


def nearby(rows, view):
    return [r for r in rows if overlap(r['bbox'], view.box) > 0]


def context_rows(rows, view):
    result = []
    for r in nearby(rows, view):
        exact = local_box(r['bbox'], view); shown = visible_box(exact, view.input_size)
        result.append(dict(id=r['id'], bbox=shown, clipped=shown != exact,
                           label=r['label'], grouping=r['grouping'], render_type=r['render_type']))
    return result


def corners(draw, bbox, color, label=None):
    x0, y0, x1, y1 = bbox
    e = min(7, max(1, (x1-x0)/3), max(1, (y1-y0)/3))
    for x, dx in ((x0, e), (x1, -e)):
        for y, dy in ((y0, e), (y1, -e)):
            draw.line([(x, y+dy), (x, y), (x+dx, y)], fill=color, width=1)
    if label:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
        draw.text((x0, max(0, y0-12)), label, fill=color, font=font, stroke_width=1, stroke_fill='white')


def annotated(source, view, baseline, candidates, mask_layer=None):
    im = view.image(source).convert('RGBA')
    if mask_layer is not None:
        mask = view.image(mask_layer).convert('L').point(lambda v: round(v*.16))
        tint = Image.new('RGBA', im.size, (30, 175, 210, 0)); tint.putalpha(mask); im.alpha_composite(tint)
    im = im.convert('RGB'); draw = ImageDraw.Draw(im)
    for r in nearby(candidates, view): corners(draw, visible_box(local_box(r['bbox'], view), im.size), '#208fad')
    for r in nearby(baseline, view): corners(draw, visible_box(local_box(r['bbox'], view), im.size), '#d37811', r['id'])
    return im


def plot(source, rows, path, color='#2877cf'):
    im = source.copy().convert('RGB'); draw = ImageDraw.Draw(im)
    for r in rows: corners(draw, visible_box(r['bbox'], im.size), color, r.get('id'))
    save(im, path)


def scan(root, cid, cfg, sem):
    out = root/'cases'/cid; image = Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    if cfg.get('reuse_frozen_sam'):
        return image,read(out/'retained_candidates.json'),read(out/'compression_audit.json')['scan_errors']
    views = [('G', Crop(image.size, (0, 0, *image.size), image.size))] + list(tile_views(image.size, cfg))
    raw = []; errors = []; mask_union = np.zeros((image.height, image.width), dtype=np.uint8)
    for vid, frame in views:
        folder = out/'sam'/vid; folder.mkdir(parents=True, exist_ok=True)
        save(frame.image(image), folder/'input.png'); write(folder/'frame.json', frame.metadata(image.size))
        try:
            start = time.monotonic()
            with sem:
                reply = SamClient(cfg['sam_url'], cfg['sam_timeout_seconds']).segment(
                    folder/'input.png', folder/'raw', cfg['sam_categories'], cfg['score_threshold'])
            write(folder/'response.json', reply)
            data = read(folder/'raw/predictions.json'); meta = data['model']
            if data['image_size'] != list(frame.input_size): raise ValueError('SAM output frame mismatch')
            if meta['checkpoint_sha256'] != cfg['sam_checkpoint_sha256']: raise ValueError('SAM checkpoint mismatch')
            if meta['checkpoint']['epoch'] != cfg['sam_epoch'] or meta['mask_pixel_threshold'] != .5:
                raise ValueError('SAM checkpoint or mask threshold mismatch')
            rows = []
            for i, p in enumerate(data['predictions'], 1):
                f = Path(p['mask_path']); f = f if f.is_absolute() else folder/'raw'/f
                with Image.open(f) as im: mask = np.asarray(im.convert('L')) > 127
                if mask.shape != (frame.input_size[1], frame.input_size[0]): raise ValueError('Mask dimensions mismatch')
                yy, xx = np.nonzero(mask)
                if not len(xx): continue
                footprint = [float(xx.min())-.5, float(yy.min())-.5, float(xx.max())+.5, float(yy.max())+.5]
                native = native_box(footprint, frame)
                row = dict(id=f'{vid}_M{i:04d}', bbox=native, input_bbox_footprint=footprint,
                           category=p['category'], score=p['score'], view=vid,
                           mask_path=str(f), mask_sha256=sha256(f), prediction_id=p['id'])
                rows.append(row)
                raster = np.asarray(Image.fromarray(mask.astype('uint8')*255).resize(frame.size, Image.Resampling.NEAREST))
                x0, y0, x1, y1 = frame.box
                np.maximum(mask_union[y0:y1, x0:x1], raster, out=mask_union[y0:y1, x0:x1])
            raw.extend(rows); write(folder/'evidence.json', dict(rows=rows, seconds=time.monotonic()-start))
        except Exception as exc:
            error = dict(stage='sam', view=vid, error=str(exc)); errors.append(error); write(folder/'failure.json', error)
        write(out/'scan_progress.json', dict(last_view=vid, observations=len(raw), issues=errors))
    save(Image.fromarray(mask_union), out/'sam_union.png')
    groups = compress(raw, cfg['dedup_iou']); audit = compression_audit(raw, groups)
    audit.update(iou=cfg['dedup_iou'], global_observations=sum(r['view']=='G' for r in raw),
                 tile_observations=sum(r['view']!='G' for r in raw),
                 groups_with_global=sum(any(x.startswith('G_') for x in g['member_ids']) for g in groups),
                 groups_without_global=sum(not any(x.startswith('G_') for x in g['member_ids']) for g in groups),
                 scan_errors=errors, threshold_counts={str(t):len(compress(raw,t)) for t in (.90,.95)})
    write(out/'raw_candidates.json', raw); write(out/'compressed_candidates.json', groups); write(out/'compression_audit.json', audit)
    plot(image, [{**r, 'id':None} for r in raw], out/'raw_candidates.png', '#208fad')
    plot(image, [{**r, 'id':None} for r in groups], out/'compressed_candidates.png', '#208fad')
    baseline = read(root/'inputs'/cid/'baseline_native.json')
    retained, excluded, filtering = filter_candidates(groups, baseline, cfg.get('sam_existing_coverage_threshold', 1.0))
    frames = {name: frame for name, frame in views}
    filtered = retained_mask_union(raw, retained, frames, image.size)
    write(out/'retained_candidates.json', retained)
    write(out/'excluded_candidates.json', excluded)
    write(out/'filter_audit.json', filtering)
    save(filtered, out/'retained_sam_union.png')
    plot(image, retained, out/'retained_candidates.png', '#208fad')
    return image, retained, errors


def proposal_prompt(size, existing, candidates, expanded=False):
    rules = D.DISCOVERY_RULES.replace(
        '看不清目标范围时标 extent=uncertain，不靠推断补齐。',
        '只定位当前图中实际可见的笔画；目标跨出视野时先提出可见范围，后续新增框整理再补上下文。')
    return dict(task='发现遗漏的箭头/线条候选，框住当前图中实际可见范围。原始框只读；完整范围由后续新增框整理确认。',
        input_size=list(size), coordinate_frame='当前CLEAN局部图绝对像素中心：x为0..W-1，y为0..H-1。后台换算，不输出frame字段。',
        rules=[SCOPE, rules,
            'CLEAN为原图；EVIDENCE为同坐标辅助图。橙色是原始定位框，青色是SAM候选与弱高亮；标注不是原图笔画。',
            '完整位于一个原始框内的目标交给该框恢复；不要逐个重新评价原框，不修改、删除或替换原框。',
            '原框只覆盖一部分时，可以另提同一目标在当前图中的可见范围，允许与旧框重叠。可见完整目标尽量框全，不只框出旧框外的一小截。',
            '目标被当前图边界裁断时，框住图内可见部分即可。不要猜测窗口外连接，不要求本阶段确认完整性。',
            'SAM可能只是残段、多条线或误检；根据CLEAN判断，不逐mask分类。没有SAM候选的明显遗漏也可以提出。',
            '只返回regions；无新增时返回regions=[]。每个候选只包含bbox、render_type、grouping、label；ID、坐标系与待审状态由后台记录。',
            '不同窗口重复由后续整理判断。single是一个可见目标，group是多条相关连线联合定位，不代表同一支箭头。'],
        existing=existing, sam_candidates=candidates,
        output_schema={'regions':[{'bbox':[0,0,10,10], 'render_type':'linework|block_arrow',
                                   'grouping':'single|group','label':'简短视觉描述'}]})


def validate_proposals(answer, size):
    return MissingProtocol(size)(answer)


def propose(root, cid, tid, view, image, baseline, groups, model, cfg):
    folder = root/'cases'/cid/'proposal_views'/tid
    mask = Image.open(root/'cases'/cid/'retained_sam_union.png').convert('L')
    for k in range(cfg['max_context_expansions']+1):
        d = folder/f'view_{k:02d}'
        old = context_rows(baseline, view)
        hints = [dict(id=r['id'], bbox=visible_box(local_box(r['bbox'], view), view.input_size))
                 for r in nearby(groups, view)]
        save(view.image(image), d/'clean.png')
        save(annotated(image, view, baseline, groups, mask), d/'evidence.png')
        write(d/'frame.json', view.metadata(image.size))
        a = model.ask(d/'request', proposal_prompt(view.input_size, old, hints, bool(k)),
            [('CLEAN: sole LOCAL coordinates', d/'clean.png'), ('EVIDENCE: annotations only', d/'evidence.png')],
            MissingProtocol(view.input_size), system=SYSTEM)
        if a['decision'] == 'expand':
            larger = expand(view, cfg)
            if larger.box == view.box or k == cfg['max_context_expansions']:
                return [], [dict(stage='missing', view=tid, reason=a['reason'], pending=True)]
            view = larger
            continue
        rows = []
        for i, r in enumerate(a['regions'], 1):
            exact = native_box(r['bbox'], view)
            rows.append({**copy.deepcopy(r), 'id': f'ADD_{tid}_{i:03d}',
                'bbox': visible_box(exact, image.size),
                'raw_local_bbox': list(r.get('raw_model_bbox', r['bbox'])),
                'effective_local_bbox': list(r['bbox']),
                'mapped_exact_bbox': exact, 'trace': str(d/'request'), 'view_id': tid,
                'operation': 'add_only'})
        C.regions_valid(rows, image.size)
        write(folder/'additions_native.json', rows)
        plot(image, rows, folder/'additions.png', '#d44b28')
        rejected = a.get('_protocol', {}).get('rejected', [])
        write(folder/'protocol_pending.json', rejected)
        return rows, ([dict(stage='missing_uncertain', view=tid, detail=u) for u in a['uncertain']] +
                      ([dict(stage='missing_protocol', view=tid, pending=True, count=len(rejected),
                             trace=str(d/'request/protocol_audit.json'))] if rejected else []))
    raise RuntimeError('No missing-stage outcome')


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def append_only(baseline, additions, size, allow_unreviewed=False):
    """No semantic filtering, union, deduplication, old-ID writeback or mutation."""
    C.regions_valid(baseline, size)
    C.regions_valid(additions, size)
    old_ids = {r['id'] for r in baseline}
    if old_ids.intersection(r['id'] for r in additions):
        raise ValueError('New IDs cannot overwrite existing regions')
    if any(r.get('operation') != 'add_only' or
           (r['extent'] != 'complete' and not (allow_unreviewed and r.get('candidate_status') == 'unreviewed'))
           for r in additions):
        raise ValueError('Only complete add_only candidates may be appended')
    forbidden = {'existing_ids', 'source_ids', 'replace', 'replacements', 'discard', 'delete', 'updates'}
    if any(forbidden.intersection(r) for r in additions):
        raise ValueError('Edit commands are not valid additions')
    final = copy.deepcopy(baseline) + copy.deepcopy(additions)
    before = rows_digest(baseline)
    after = rows_digest(final[:len(baseline)])
    if before != after:
        raise ValueError('Original rows changed')
    return final, dict(passed=True, baseline_count=len(baseline), additions_count=len(additions),
        before_sha256=before, after_sha256=after, original_order_and_fields_identical=True,
        old_rows_removed=0, old_rows_modified=0, output_dedup_executed=False)


def persist_snapshot(root, cid, image, baseline, additions):
    d = root/'cases'/cid
    if rows_digest(read(root/'inputs'/cid/'baseline_native.json')) != rows_digest(baseline):
        raise ValueError('Frozen baseline changed during execution')
    current, audit = append_only(baseline, additions, image.size, allow_unreviewed=True)
    write(d/'additions.json', additions)
    write(d/'current_native.json', current)
    write(d/'baseline_preservation.json', audit)
    plot(image, current, d/'current.png')
    return current, audit


def export_additive(cid, final, source_path):
    with Image.open(source_path) as im:
        view = global_frame(im.size)
    global_rows = [{**copy.deepcopy(r), 'bbox': local_box(r['bbox'], view)} for r in final]
    handoff, audits = export(cid, global_rows, view, source_path)
    for r, task, audit in zip(final, handoff['tasks'], audits):
        # GLOBAL is an interchange view. Native boxes remain authoritative exactly,
        # including original float values, instead of accumulating roundtrip noise.
        delta = max(abs(a-b) for a, b in zip(r['bbox'], task['bbox_native']))
        if delta > 1e-8:
            raise ValueError('Unexpected handoff coordinate drift')
        audit['roundtrip_max_abs_error_px'] = delta
        audit['canonical_native_bbox'] = list(r['bbox'])
        task['bbox_native'] = list(r['bbox'])
        if r.get('candidate_status') == 'unreviewed':
            task.update(task_role='candidate_only', render_owner=False, requires_review=True)
    handoff['missing_stage_policy'] = 'add_only; original rows retained; output dedup not executed'
    return global_rows, handoff, audits


def publish(root):
    with REPORT_LOCK:
        records = []; sections = []
        for cid in read(root/'cases.json'):
            d = root/'cases'/cid
            s = read(d/'status.json', {'state':'queued', 'stage':'queued'})
            a = read(d/'baseline_preservation.json', {})
            f = read(d/'filter_audit.json', {})
            records.append(dict(case=cid, **s))
            links = [f'[原图](../inputs/{cid}/source.png)', f'[原始定位](../inputs/{cid}/baseline_native.json)']
            for name, label in [('additions.json', '新增框'), ('baseline_preservation.json', '旧框一致性'),
                                ('reconstruction_handoff.json', '重建交接')]:
                if (d/name).exists(): links.append(f'[{label}](../cases/{cid}/{name})')
            sections += [f'## {cid}', '',
                f"{s['state']} / {s['stage']}；已返回窗口 {s.get('views_finished',0)}/6；新增框 {s.get('additions',0)}；原框逐项一致 {a.get('passed','待检查')}。", '',
                ' · '.join(links), '', '| 原定位 | 本轮原定位＋新增框 |', '|---|---|',
                f'| [![原定位](../cases/{cid}/baseline.png)](../cases/{cid}/baseline.png) | '+
                (f'[![当前](../cases/{cid}/current.png)](../cases/{cid}/current.png)' if (d/'current.png').exists() else '等待补漏')+' |', '']
        finished = sum(r['state'] in ('completed','completed_with_issues','failed') for r in records)
        write(root/'progress.json', dict(updated_at=now(), finished=finished, total=len(records), cases=records))
        dest = root/'reports'; dest.mkdir(exist_ok=True)
        text = ['# 只追加补漏：24图对照', '', f'更新 {now()}；结束 {finished}/{len(records)}。', '',
            '既有定位和24图SAM证据全部复用。本阶段只发现未审核候选，允许当前视野内的可见片段，旧框保持；候选随后进入独立新增框整理。',
            '24图全部重新请求补漏；16图、每图6视野，与整理共享API128。默认思考；结果返回即更新叠图。未执行几何/样式，候选数量不是新增箭头数量或召回率。', '',
            '[实验说明](../README.md) · [配置](../config.json) · [离线检查](../preflight/checks.json)', ''] + sections
        tmp = dest/'24图_实时对照.md.tmp'; tmp.write_text('\n'.join(text)+'\n'); tmp.replace(dest/'24图_实时对照.md')
        (root/'RESULTS.md').write_text(f'# 只追加SAM补漏实验\n\n更新 {now()}；结束 {finished}/{len(records)}。\n\n'
            '[24图对照](reports/24图_实时对照.md) · [方案与变更](README.md) · [离线检查](preflight/checks.json)\n\n'
            'completed只代表执行结束。没有修旧框或结果去重，没有几何恢复；质量和耗时须待实际模型实验。\n')


def run_case(root, cid, cfg, model, sam_sem):
    d = root/'cases'/cid
    baseline = read(root/'inputs'/cid/'baseline_native.json')
    additions = []; issues = []; answers = {}; returned = 0
    def state(stage, **kw):
        write(d/'status.json', dict(state='running', stage=stage, updated_at=now(),
            additions=len(additions), views_finished=returned, **kw))
        publish(root)
    try:
        state('reuse_filtered_sam' if cfg.get('reuse_frozen_sam') else 'sam_scan')
        image, groups, errors = scan(root, cid, cfg, sam_sem); issues.extend(errors)
        persist_snapshot(root, cid, image, baseline, [])
        state('add_only_missing')
        with ThreadPoolExecutor(max_workers=cfg['view_workers']) as pool:
            jobs = {pool.submit(propose, root, cid, tid, v, image, copy.deepcopy(baseline), groups, model, cfg): tid
                    for tid, v in tile_views(image.size, cfg)}
            for future in as_completed(jobs):
                tid = jobs[future]; returned += 1
                try: answers[tid] = future.result()
                except Exception as exc:
                    issue = dict(stage='missing', view=tid, error=str(exc))
                    issues.append(issue); write(d/'proposal_views'/tid/'failure.json', issue)
                additions = [r for key in sorted(answers) for r in answers[key][0]]
                persist_snapshot(root, cid, image, baseline, additions)
                state('add_only_missing')
        issues.extend(u for key in sorted(answers) for u in answers[key][1])
        final, audit = persist_snapshot(root, cid, image, baseline, additions)
        global_rows, handoff, coords = export_additive(cid, final, root/'inputs'/cid/'source.png')
        write(d/'reconstruction_handoff.json', handoff); write(d/'coordinate_audit.json', coords)
        st = 'completed_with_issues' if issues else 'completed'
        write(d/'result.json', dict(case_id=cid, state=st, regions=global_rows, native_regions=final,
            region_count=len(final), baseline_count=len(baseline), additions_count=len(additions),
            baseline_preserved=audit['passed'], issues=issues, geometry_executed=False,
            correction_executed=False, reconcile_executed=False, output_dedup_executed=False,
            formal_accuracy_measured=False))
        write(d/'status.json', dict(state=st, stage='finished', updated_at=now(),
            additions=len(additions), views_finished=returned))
    except Exception as exc:
        write(d/'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        write(d/'status.json', dict(state='failed', stage='failed', updated_at=now(), error=str(exc)))
    publish(root)


def run(root):
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in read(root/'manifest.json')['files'].items():
        if sha256(root/name)!=h:raise ValueError('Frozen file changed: '+name)
    cfg=read(root/'config.json')
    for item in read(root/'sam_evidence_manifest.json')['external_masks']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Reused mask changed: '+item['path'])
    model=Model(cfg,run_dir=root);sem=threading.Semaphore(cfg['sam_workers'])
    write(root/'process.json',dict(pid=os.getpid(),started_at=now()))
    write(root/'execution.json',dict(state='running',started_at=now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
            jobs=[pool.submit(run_case,root,cid,cfg,model,sem) for cid in cfg.get('execution_case_ids', read(root/'cases.json'))]
            for f in as_completed(jobs):f.result()
        # The already-running five cases are linked for reporting, never resubmitted.
        while any(read(root/'cases'/cid/'status.json',{}).get('state') not in ('completed','completed_with_issues','failed')
                  for cid in cfg.get('linked_case_ids',[])):
            write(root/'execution.json',dict(state='waiting_linked_cases',updated_at=now()))
            time.sleep(15)
        write(root/'execution.json',dict(state='finished',finished_at=now()))
    finally:stop.set();watcher.join();publish(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','report']);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();(run if args.action=='run' else publish)(args.run_dir.resolve())
