"""Replay original groups and post-coverage scope review from a data manifest."""
import argparse
import copy
import fcntl
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from PIL import Image
import group_recovery
import protocol
import scope_review
from lib import native, viz
from transport.common import read_json as read, write_json as write, sha256 as sha
from transport.model import Model
from run_region_ablation import usage


def prepare(spec, folder):
    folder.mkdir(parents=True, exist_ok=True)
    case = Path(spec['source_case'])
    if spec['mode'] == 'group':
        source = case / 'groups' / spec['group']
        shutil.copytree(source / 'input', folder / 'input', dirs_exist_ok=True)
        meta = read(source / 'prepared_observation.json')
        write(folder / 'prepared_observation.json', meta)
        hints = read(case / 'global_hints.json')
        plan = read(case / 'observation_plan.json')['groups']
        labels = next(p['sources'] for p in plan if p['id'] == spec['group'])
        group = [r for r in hints if r['id'] in labels]
        write(folder / 'group.json', group)
        if (source / 'rows_local.json').exists():
            rows = read(source / 'rows_local.json')
            viz.render(folder / 'input/clean.png', rows, folder / 'before')
        sources = [source / 'input/clean.png', source / 'input/context.png',
                   source / 'input/transform.json', source / 'prepared_observation.json',
                   case / 'global_hints.json', case / 'observation_plan.json']
        sources += [Path(c['path']) for c in meta['candidates'] if c['id'] in meta['selected_ids']]
        write(folder / 'input_audit.json', dict(
            selection=spec, original_group_ids=labels, original_crop=meta['view'],
            new_sam_calls=0, source_hashes={str(p.resolve()): sha(p) for p in sources},
            identity_inputs='CLEAN + original group boxes + CONTEXT; no local masks or D estimates',
            prepared_from=str(source.resolve())))
    elif spec['mode'] == 'scope_review':
        shutil.copy2(case / 'clean.png', folder / 'clean.png')
        rows = read(case / 'scene.json')['arrows']
        write(folder / 'source_results.json', [dict(group='saved_postcoverage', state='completed', rows=rows)])
        viz.render(folder / 'clean.png', rows, folder / 'before')
        write(folder / 'input_audit.json', dict(
            selection=spec, source_hashes={str((case / n).resolve()): sha(case / n) for n in ('scene.json', 'clean.png')},
            new_sam_calls=0, input_paths=len(rows), immutable_geometry_style=True, dedup=False))
    else:
        raise ValueError('Unknown trial mode')


def work(spec, run, config, model, locks, render_slot, report):
    folder = run / 'cases' / spec['name']
    started = time.monotonic()
    state = dict(state='running', mode=spec['mode'])
    report(spec['name'], state)
    try:
        if spec['mode'] == 'group':
            case = Path(spec['source_case'])
            clean = Image.open(case / 'clean.png').convert('RGB')
            hints = read(case / 'global_hints.json')
            plan = read(case / 'observation_plan.json')['groups']
            group_of = {s: p['id'] for p in plan for s in p['sources']}
            result = group_recovery.group_task(
                clean, read(folder / 'group.json'), hints, group_of, spec['group'], folder,
                config, model, locks, prepared=read(folder / 'prepared_observation.json'))
            rows = read(folder / 'rows_local.json')
            local = Image.open(folder / 'input/clean.png')
            size = local.size
            local.close()
            state.update(state=result['state'], paths=len(rows), identities=len(result['identities']),
                         excluded=result['excluded'], issues=result['issues'],
                         followups=result['followups'], geometry_only=True, style_fitted=False,
                         coverage_run=False, sam_calls=0)
        else:
            clean = Image.open(folder / 'clean.png').convert('RGB')
            results, issues = scope_review.run(clean, read(folder / 'source_results.json'),
                                              folder / 'review', model, locks)
            rows = [r for g in results for r in g['rows']]
            size = clean.size
            with locks['cpu']:
                viz.render(folder / 'clean.png', rows, folder / 'result')
            state.update(state='completed_with_issues' if issues else 'completed',
                         paths=len(rows), issues=issues, audit=read(folder / 'review/audit.json'),
                         existing_style_preserved=True, coverage_run=False, sam_calls=0)
        report(spec['name'], {**state, 'state': 'rendering'})
        with locks['cpu']:
            native.write(size, rows, [r['style'] for r in rows], folder / 'arrows.pptx')
        with render_slot:
            native.render(folder / 'arrows.pptx', folder / 'native', size)
        state['native_rendered'] = True
    except Exception as exc:
        state.update(state='failed', error=str(exc), traceback=traceback.format_exc())
    state.update(seconds=time.monotonic() - started, api=usage(folder), accuracy_measured=False)
    write(folder / 'finished.json', state)
    report(spec['name'], state)
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
    if any(sha(run / 'code' / n) != h for n, h in manifest['files'].items()):
        raise ValueError('Frozen code changed')
    if sha(run / 'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen config changed')
    config = read(run / 'config.json')
    specs = read(run / 'selection.json')
    if len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Duplicate experiment selection names')
    for spec in specs:
        folder = run / 'cases' / spec['name']
        audit = folder / 'input_audit.json'
        if audit.exists():
            saved = read(audit)
            if saved['selection'] != spec or any(sha(n) != h for n, h in saved['source_hashes'].items()):
                raise ValueError('Prepared evidence changed')
        else:
            prepare(spec, folder)
    mutex = threading.RLock()
    started = time.monotonic()
    states = {}

    def report(name=None, value=None):
        with mutex:
            if name:
                states[name] = copy.deepcopy(value)
            ended = sum((run / 'cases' / s['name'] / 'finished.json').exists() for s in specs)
            write(run / 'progress.json', dict(total=len(specs), completed=ended, cases=states,
                                             seconds=time.monotonic() - started, updated=time.time()))
            lines = ['# 原分组 bbox 身份与补漏后范围复核', '',
                     '分组样例重放身份 → 目标 mask / GLM 几何，使用原分组、原 crop、缓存 SAM。'
                     '这部分未跑新补漏或样式，灰色线是待恢复样式的几何。0409 全图 review 使用旧补漏后的实际场景，保留剩余路径的几何与样式。',
                     '', f'已结束 {ended}/{len(specs)}。', '',
                     '| 样例 | 原图 | 原结果 | 本轮结果 |', '|---|---|---|---|']
            for spec in specs:
                name = spec['name']
                f = run / 'cases' / name
                rel = 'cases/' + name
                source = 'input/clean.png' if spec['mode'] == 'group' else 'clean.png'
                old = f'![]({rel}/before/instances.png)' if (f / 'before/instances.png').exists() else '原组没有成功输出'
                image = f'![]({rel}/result/instances.png)' if (f / 'result/instances.png').exists() else '等待'
                state = states.get(name, {})
                lines.append('| ' + ' | '.join([name, f'![]({rel}/{source})', old,
                             state.get('state', 'prepared') + f'；{state.get("paths", "—")} 条<br>' + image]) + ' |')
            lines += ['', '| 样例 | 身份定位／review 输入 | 最终叠图 |', '|---|---|---|']
            for spec in specs:
                name = spec['name']; f = run / 'cases' / name; rel = 'cases/' + name
                input_name = 'input/target_boxes.png' if spec['mode'] == 'group' else 'review/recovered.png'
                a = f'![]({rel}/{input_name})' if (f / input_name).exists() else '等待'
                b = f'![]({rel}/result/overlap.png)' if (f / 'result/overlap.png').exists() else '等待'
                lines.append(f'| {name} | {a} | {b} |')
            lines += ['', '[运行状态](progress.json) · [样例来源](selection.json) · [实验说明](实验说明.md)',
                      '', '实例数与执行成功不代表视觉准确；重复暂不处理。']
            tmp = run / 'RESULTS.md.tmp'
            tmp.write_text('\n'.join(lines) + '\n')
            tmp.replace(run / 'RESULTS.md')
    report()
    if args.prepare_only:
        print('Prepared original group inputs and saved post-coverage scene; no model calls.', flush=True)
        return
    write(run / 'process.json', dict(pid=os.getpid(), started=time.time(), kind='group_bbox_scope_review'))
    model = Model(config, run_dir=run, system=protocol.SYSTEM)
    locks = {k: threading.Semaphore(config[k + '_workers']) for k in ('cpu', 'sam')}
    render_slot = threading.Semaphore(1)
    done = {}
    with ThreadPoolExecutor(max_workers=min(8, config['task_workers'])) as pool:
        jobs = {pool.submit(work, s, run, config, model, locks, render_slot, report): s for s in specs}
        for f in as_completed(jobs):
            name = jobs[f]['name']
            done[name] = f.result()
            print(name, done[name]['state'], flush=True)
    write(run / 'finished.json', dict(total=len(specs), completed=len(done), cases=done,
                                      seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
