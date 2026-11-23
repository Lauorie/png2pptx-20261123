"""Matched CLEAN+box vs CLEAN+box+mask experiments, selected by a data manifest."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import numpy as np
from PIL import Image, ImageDraw
import evidence as ev
import region_geometry
import visual
from lib import fit, native, viz
from transport.common import read_json as read, write_json as write, sha256 as sha
from transport.model import Model

VARIANTS = ('A_bbox', 'B_bbox_mask')


def relative(path, root):
    return os.path.relpath(path, root)


def prepare_sample(spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    inputs = folder / 'input'
    inputs.mkdir(exist_ok=True)
    case = Path(spec['source_case'])
    source = case / 'clean.png'
    mask_sources = []
    if spec.get('source_group'):
        group = case / 'groups' / spec['source_group']
        view = read(group / 'input/transform.json')
        shutil.copy2(group / 'input/clean.png', inputs / 'clean.png')
        meta = read(group / 'prepared_observation.json')
        candidates = {m['id']: m for m in meta['candidates']}
        for mid in meta['selected_ids']:
            c = candidates[mid]
            mask_sources.append(dict(id=mid, path=c['path'], view=view))
    else:
        image = Image.open(source).convert('RGB')
        image.save(inputs / 'clean.png')
        view = ev.identity_view(image.size)
        for hint in read(case / 'global_hints.json'):
            mask_sources.append(dict(id=hint['id'], path=hint['path'], view=hint['view']))
    if spec.get('legacy_frame'):
        # Reuse an earlier experiment's exact pixels and documented transport.
        frame = read(spec['legacy_frame'])
        scale = float(frame['scale'])
        x0, y0 = frame['box'][:2]
        ox, oy = frame['offset']
        matrix = np.array([[scale, 0, (-x0 + .5) * scale - .5 + ox],
                           [0, scale, (-y0 + .5) * scale - .5 + oy],
                           [0, 0, 1.]])
        view = dict(bbox_global=frame['box'], size=frame['size'], global_size=frame['source_size'],
                    scale=[scale, scale], global_to_view=matrix.tolist(),
                    view_to_global=np.linalg.inv(matrix).tolist())
        shutil.copy2(spec['clean_override'], inputs / 'clean.png')
    clean = Image.open(inputs / 'clean.png').convert('RGB')
    if list(clean.size) != view['size']:
        raise ValueError('CLEAN dimensions differ from recorded frame')
    write(inputs / 'transform.json', view)
    write(inputs / 'source.json', dict(path=str(source), sha256=sha(source)))
    box = spec.get('focus_global', [0, 0, *view['global_size']])
    corners = np.array([[box[0], box[1], 1], [box[2], box[3], 1]], float)
    local = (corners @ np.asarray(view['global_to_view']).T)[:, :2]
    local = np.clip(local, [0, 0], np.asarray(clean.size) - 1).reshape(-1).tolist()
    if local[2] <= local[0] or local[3] <= local[1]:
        raise ValueError('Target box does not intersect CLEAN')
    target = clean.copy()
    ImageDraw.Draw(target).rectangle(local, outline='#c00080', width=max(2, round(max(clean.size) / 700)))
    target.save(inputs / 'target.png')
    layers = []
    mask_manifest = []
    for row in mask_sources:
        mask = ev.project(row, view)
        if not mask.any():
            continue
        # Unlabelled evidence: no per-M matching, D axes or mask census task.
        layers.append(('', mask))
        mask_manifest.append(dict(id=row['id'], path=row['path'], sha256=sha(row['path']),
                                  source_view=row['view'], projected_pixels=int(mask.sum())))
    visual.masks(clean, layers, inputs / 'masks.png')
    write(inputs / 'mask_sources.json', mask_manifest)
    prompt = region_geometry.prompt(clean.size, local, spec.get('description', ''))
    write(folder / 'prompt.json', prompt)
    write(folder / 'input_manifest.json', dict(
        selection=spec, view=view, target_box=local, mask_count=len(layers), new_sam_calls=0,
        files={name: sha(inputs / name) for name in ('clean.png', 'target.png', 'masks.png')},
        same_prompt_for_both_variants=True, only_input_difference='additional MASK image in B',
    ))
    return view


def usage(folder):
    calls = []
    for p in sorted(folder.rglob('call_*/status.json')):
        s = read(p)
        calls.append(dict(path=str(p.parent), state=s['state'], seconds=s.get('seconds'),
                          usage=s.get('usage'), failure_kind=s.get('failure_kind')))
    completed = [c for c in calls if c['state'] == 'transport_complete']
    return dict(attempts=len(calls), failed_attempts=sum(c['state'] == 'failed' for c in calls),
                completed_call_seconds=sum(c.get('seconds') or 0 for c in completed),
                reported_tokens=sum((c.get('usage') or {}).get('total_tokens', 0) for c in completed),
                reasoning_tokens=sum((c.get('usage') or {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                                     for c in completed), calls=calls)


def work(spec, variant, run, config, model, cpu, render_slot, report):
    sample = run / 'cases' / spec['name']
    folder = sample / variant
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'finished.json').exists():
        return read(folder / 'finished.json')
    started = time.time()
    state = dict(state='requesting', started=started)
    report(spec['name'], variant, state)
    try:
        view = read(sample / 'input/transform.json')
        prompt = read(sample / 'prompt.json')
        images = [('CLEAN source pixels', sample / 'input/clean.png'),
                  ('TARGET box: observation locator, not source ink', sample / 'input/target.png')]
        if variant == 'B_bbox_mask':
            images.append(('MASK: optional imperfect pixel evidence; colors carry no identity', sample / 'input/masks.png'))
        answer = model.ask(folder / 'geometry', prompt, images,
                           lambda a: region_geometry.validate(a, view['size']))
        write(folder / 'geometry_local.json', answer)
        scale = float(np.sqrt(np.prod(view['scale'])))
        rows = region_geometry.compile_rows(answer, spec['name'], config.get('preview_width_global_px', 1.5) * scale)
        for row in rows:
            row.update(local_input=str(sample / 'input/clean.png'), geometry_frame=view,
                       experiment_variant=variant)
        write(folder / 'rows_local.json', rows)
        global_rows = [fit.transform(row, view) for row in rows]
        write(folder / 'rows_global.json', global_rows)
        state.update(state='geometry_ready', model_status=answer['status'], paths=len(rows),
                     uncertain=answer.get('uncertain', []), api=usage(folder))
        report(spec['name'], variant, state)
        with cpu:
            viz.render(sample / 'input/clean.png', rows, folder / 'preview')
            native.write(view['size'], rows, [r['style'] for r in rows], folder / 'geometry.pptx')
        with render_slot:
            native.render(folder / 'geometry.pptx', folder / 'native', view['size'])
        state.update(state='completed' if answer['status'] == 'ready' else answer['status'],
                     native_rendered=True, geometry_critic_run=False, style_fitted=False)
    except Exception as exc:
        state.update(state='failed', error=str(exc), traceback=traceback.format_exc(), api=usage(folder))
    state.update(finished=time.time(), seconds=time.time() - started)
    write(folder / 'finished.json', state)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    lock = (run / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(run / 'code_manifest.json')
    if any(sha(run / 'code' / name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('Frozen experiment code changed')
    if sha(run / 'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen experiment config changed')
    selections = read(run / 'selection.json')
    if len({s['name'] for s in selections}) != len(selections):
        raise ValueError('Sample names must be unique')
    config = read(run / 'config.json')
    for spec in selections:
        folder = run / 'cases' / spec['name']
        previous = read(folder / 'input_manifest.json') if (folder / 'input_manifest.json').exists() else None
        if previous:
            if previous['selection'] != spec:
                raise ValueError('Prepared selection changed')
            if any(sha(folder / 'input' / n) != h for n, h in previous['files'].items()):
                raise ValueError('Prepared input changed')
        else:
            prepare_sample(spec, folder)
    states = read(run / 'progress.json').get('cases', {}) if (run / 'progress.json').exists() else {}
    mutex = threading.RLock()
    started = time.time()

    def report(name=None, variant=None, value=None):
        with mutex:
            if name:
                states.setdefault(name, {})[variant] = copy.deepcopy(value)
            ended = sum((run / 'cases' / s['name'] / v / 'finished.json').exists()
                        for s in selections for v in VARIANTS)
            write(run / 'progress.json', dict(updated=time.time(), seconds=time.time() - started,
                                             total=len(selections) * len(VARIANTS), completed=ended, cases=states))
            lines = ['# bbox 区域几何：有 / 无 mask 消融', '',
                     '局部几何验证；显示统一黑色实线，箭头仅用三角形表示头部存在。没有拟合颜色、线宽、虚线或头型。'
                     '无新 SAM、补漏、critic 或坐标吸附。两组的原图、TARGET 框、文字任务、输出协议和模型参数一致，B 仅增加一张无编号 mask 叠图。',
                     '', f'已结束 {ended}/{len(selections) * len(VARIANTS)} 个任务。', '',
                     '| 样例 | 同视野原图 | A：原图＋整体框 | B：再加 mask |', '|---|---|---|---|']
            for spec in selections:
                key = spec['name']
                cells = [key, f'![](cases/{key}/input/clean.png)']
                for v in VARIANTS:
                    f = run / 'cases' / key / v
                    s = states.get(key, {}).get(v, {})
                    image = f'![](cases/{key}/{v}/native/styled.png)' if (f / 'native/styled.png').exists() else '等待'
                    api = s.get('api', {})
                    calls = api.get('completed_call_seconds')
                    summary = f'{s.get("state","prepared")}；{s.get("paths","—")} 条；API {calls:.1f}s' if calls is not None else s.get('state', 'prepared')
                    cells.append(summary + '<br>' + image)
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines += ['', '| 样例 | A：原图叠图 | B：原图叠图 | B：额外输入 |', '|---|---|---|---|']
            for spec in selections:
                key = spec['name']
                cells = [key]
                for v in VARIANTS:
                    path = run / 'cases' / key / v / 'preview/overlap.png'
                    cells.append(f'![](cases/{key}/{v}/preview/overlap.png)' if path.exists() else '等待')
                cells.append(f'![](cases/{key}/input/masks.png)')
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines += ['', '[实验设计](实验设计.md) · [进度](progress.json) · [样例来源](selection.json) · [输入核验](input_audit.json)', '']
            temporary = run / 'RESULTS.md.tmp'
            temporary.write_text('\n'.join(lines))
            temporary.replace(run / 'RESULTS.md')
    report()
    if args.prepare_only:
        write(run / 'prepared.json', dict(samples=len(selections), tasks=len(selections) * len(VARIANTS), model_calls=0))
        print('Prepared matched inputs; no model calls.', flush=True)
        return
    write(run / 'process.json', dict(pid=os.getpid(), started=time.time(), kind='region_geometry_ablation'))
    model = Model(config, run_dir=run, system=region_geometry.SYSTEM)
    cpu = threading.Semaphore(config['cpu_workers'])
    render_slot = threading.Semaphore(1)
    with ThreadPoolExecutor(max_workers=config['task_workers']) as pool:
        jobs = {pool.submit(work, s, v, run, config, model, cpu, render_slot, report): (s['name'], v)
                for s in selections for v in VARIANTS}
        for future in as_completed(jobs):
            name, variant = jobs[future]
            result = future.result()
            report(name, variant, result)
            print(name, variant, result['state'], result.get('paths'), flush=True)
    write(run / 'finished.json', dict(cases=states, seconds=time.time() - started))


if __name__ == '__main__':
    main()
