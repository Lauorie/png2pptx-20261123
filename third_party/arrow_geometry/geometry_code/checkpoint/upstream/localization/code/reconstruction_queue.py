"""Consume completed fresh localization cases with frozen geometry components."""
import argparse
import copy
import fcntl
import hashlib
import html
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(source, cid):
    folder = Path(source) / 'cases' / cid
    return all((folder / name).exists() for name in ('finished.json', 'result.json', 'reconstruction_handoff.json'))


def budget_available(config):
    previous = Path(config['wait_for_geometry_run'])
    return (previous / 'geometry_finished.json').exists() or (previous / 'finished.json').exists()


def freeze_case(source, root, cid):
    if not ready(source, cid):
        raise ValueError('Localization has not finished: ' + cid)
    folder = root / 'cases' / cid
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        'clean.png': source / 'inputs' / cid / 'source.png',
        'reconstruction_handoff.json': source / 'cases' / cid / 'reconstruction_handoff.json',
        'localization_result.json': source / 'cases' / cid / 'result.json',
        'localization_finished.json': source / 'cases' / cid / 'finished.json',
        'localization.png': source / 'cases' / cid / 'previews' / '05_final.png',
    }
    evidence = {}
    for name, src in files.items():
        if name == 'localization.png' and not src.exists():
            continue
        before = digest(src)
        dst = folder / name
        if dst.exists() and digest(dst) != before:
            raise ValueError('Previously frozen localization changed: ' + str(src))
        if not dst.exists():
            shutil.copy2(src, dst)
        if digest(src) != before or digest(dst) != before:
            raise ValueError('Localization changed while being copied: ' + str(src))
        evidence[name] = {'source': str(src), 'sha256': before}
    atomic(folder / 'input_provenance.json', evidence)
    return folder


def report(root, source, ids, config):
    assets = root / 'reports' / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    records, sections = [], []
    stamp = datetime.now(timezone.utc).isoformat()
    md = [f"# {config.get('run_label', source.name)}：定位到重新重建", '', '更新时间：' + stamp,
          '', '所有模型几何与审核重新请求。运行中图片只含已返回目标；完成不代表视觉正确。',
          '', '[自动刷新网页](live.html) · [配置](../config.json)', '']
    for cid in ids:
        out = root / 'cases' / cid
        status = read(out / 'progress.json') if (out / 'progress.json').exists() else {
            'state': 'queued', 'stage': 'waiting_budget' if ready(source, cid) else 'waiting_localization'}
        records.append({'case': cid, **status})
        pictures = []
        for name, src in [('source', source / 'inputs' / cid / 'source.png'),
                          ('boxes', source / 'cases' / cid / 'previews' / '05_final.png'),
                          ('geometry', out / 'result' / 'styled.png'),
                          ('overlay', out / 'result' / 'overlap.png')]:
            if not src.exists():
                pictures.append(None)
                continue
            dst = assets / f'{cid}_{name}_{src.stat().st_mtime_ns}.png'
            if not dst.exists():
                shutil.copy2(src, dst)
            pictures.append('assets/' + dst.name)
        label = status['stage'] + f" · {status.get('ended', 0)}/{status.get('total', '?')} 目标"
        md += ['## ' + cid, '', label,
               '', f'[Trace](../cases/{cid}/) · [本图状态](../cases/{cid}/progress.json)',
               '', '| 原图 | 新定位 | 几何 | 叠图 |', '|---|---|---|---|',
               '| ' + ' | '.join(f'[![]({x})]({x})' if x else '等待' for x in pictures) + ' |', '']
        cells = ''.join(f'<div><a href="{x}"><img src="{x}"></a></div>' if x else '<div>等待</div>' for x in pictures)
        sections.append(f'<section id="{cid}"><h2>{cid} · {html.escape(label)}</h2><div class="grid">{cells}</div></section>')
    done = sum(r['state'] in ('completed', 'completed_with_issues', 'failed', 'upstream_failed') for r in records)
    atomic(root / 'progress.json', {'updated_at': stamp, 'ended': done, 'total': len(ids),
        'api_concurrency': config['api_concurrency'], 'budget_available': budget_available(config), 'cases': records})
    atomic(root / 'reports' / '24图_重建实时对照.md', '\n'.join(md) + '\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>定位与重新重建</title><style>body{font:16px system-ui;background:#f6f7fa;margin:24px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;display:block}section{background:white;padding:16px;margin:24px 0}</style>'
    atomic(root / 'reports' / 'live.html', page + f'<h1>新定位到重建 {done}/{len(ids)}</h1><p>{stamp} · 每15秒刷新。原图、新定位、几何、叠图按顺序排列。</p>' + ''.join(sections))


def run(root):
    lock = (root / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in read(root / 'manifest.json')['files'].items():
        if digest(root / name) != expected:
            raise ValueError('Frozen file changed: ' + name)
    cfg = read(root / 'config.json')
    source = Path(cfg['source_localization'])
    ids = read(root / 'cases.json')
    sys.path[:0] = [str(root / 'geometry_code'), str(root / 'geometry_code' / 'lib')]
    from PIL import Image
    import bbox_geometry, bbox_inputs, pipeline, protocol, viz
    from scene_snapshot import canonical_rows
    from transport.model import Model
    from transport.common import write_json as write
    from native_runtime import validated_renderer
    renderer, version = validated_renderer()
    write(root / 'renderer_preflight.json', {'executable': str(renderer), 'version': version,
        'isolated_profile': True, 'cached_per_build': True, 'validated_at': datetime.now(timezone.utc).isoformat()})
    protocol.OCCLUSION = ('Reconstruct visible stroke portions only. Do not infer or draw hidden connectors behind text, nodes or opaque objects. '
        'Separate visible portions may be separate paths. If only the local crop truncates a stroke, use the whole-image context to recover its actually visible continuation. '
        'Crossings alone are not junctions; a group region may contain multiple independent paths.')
    model = Model(cfg, run_dir=root, system=protocol.SYSTEM)
    locks = {'cpu': threading.Semaphore(cfg['cpu_workers']), 'sam': threading.Semaphore(cfg['sam_workers'])}
    write(root / 'process.json', {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()})

    def case(cid):
        out = freeze_case(source, root, cid)
        with Image.open(out / 'clean.png') as im:
            clean = im.convert('RGB')
        plan = bbox_inputs.normalize(read(out / 'reconstruction_handoff.json'), clean.size, cid)
        write(out / 'geometry_plan.json', plan)
        upstream = read(out / 'localization_result.json')
        write(out / 'discovery_provenance.json', {'provider': cfg.get('source_provider', 'model_led_localization'),
            'source': str(source), 'upstream_state': upstream['state'], 'upstream_issues': upstream.get('issues', []),
            'handoff_sha256': digest(out / 'reconstruction_handoff.json'), 'new_geometry_requests': True})
        results = {}
        start, last_preview = time.monotonic(), 0

        def progress(stage, state='running'):
            write(out / 'progress.json', {'stage': stage, 'state': state, 'ended': len(results),
                'total': len(plan['tasks']), 'failed': sum(x['state'] in ('failed', 'unresolved') for x in results.values()),
                'seconds': time.monotonic() - start})

        def recover(task):
            folder = out / 'geometry_tasks' / task['id']
            # Only this run's own exact completed task may resume after interruption.
            saved = folder / 'replay_result.json'
            if saved.exists():
                return read(saved)
            try:
                value = bbox_geometry.recover_target(clean, task, folder, model, cfg, locks)
            except Exception as exc:
                value = {'group': task['id'], 'state': 'failed', 'rows': [],
                    'issues': [{'stage': 'geometry', 'error': str(exc), 'traceback': traceback.format_exc()}]}
            write(saved, value)
            return value

        def snapshot(final=False):
            ordered = [results[t['id']] for t in plan['tasks'] if t['id'] in results]
            write(out / 'geometry_results.json', ordered)
            with locks['cpu']:
                if final:
                    pipeline.publish(clean, ordered, out)
                    if not (out / 'native_geometry' / 'styled.png').exists():
                        from lib import native
                        from native_foreground import render_layer
                        native.render(out / 'arrows.pptx', out / 'native_geometry', clean.size)
                        shutil.copy2(out / 'native_geometry/styled.png', out / 'result/styled.png')
                        foreground = render_layer(out / 'native_geometry/styled.svg', clean.size, out / 'native_geometry')
                        overlay = clean.convert('RGBA'); overlay.alpha_composite(foreground)
                        overlay.convert('RGB').save(out / 'result/overlap.png')
                else:
                    rows = canonical_rows(copy.deepcopy(ordered))
                    for row in rows:
                        row['label'] = row.get('parent_id', row['id'])
                    viz.render(out / 'clean.png', rows, out / 'result')

        progress('geometry_and_critic')
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
            jobs = {pool.submit(recover, task): task['id'] for task in plan['tasks']}
            for future in as_completed(jobs):
                results[jobs[future]] = future.result()
                progress('geometry_and_critic')
                if time.monotonic() - last_preview > 30:
                    try:
                        snapshot()
                    except Exception as exc:
                        write(out / 'preview_error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
                    last_preview = time.monotonic()
        progress('native_render')
        snapshot(True)
        issues = plan['issues'] + [e for x in results.values() for e in x.get('issues', [])]
        unresolved = [k for k, v in results.items() if v['state'] in ('failed', 'unresolved')]
        state = 'completed_with_issues' if issues or unresolved else 'completed'
        write(out / 'completed.json', {'case': cid, 'state': state, 'issues': issues, 'unresolved': unresolved,
            'targets': len(plan['tasks']), 'paths': len(canonical_rows(list(results.values()))),
            'upstream_state': upstream['state'], 'visual_accuracy_measured': False})
        progress('finished', state)
        print(cid, state, len(plan['tasks']), flush=True)

    pending, active, ended = set(ids), {}, set()
    for cid in ids:
        if (root / 'cases' / cid / 'completed.json').exists():
            pending.remove(cid); ended.add(cid)
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        while pending or active:
            for future, cid in list(active.items()):
                if not future.done():
                    continue
                try:
                    future.result()
                except Exception as exc:
                    write(root / 'cases' / cid / 'fatal.json', {'error': str(exc), 'traceback': traceback.format_exc()})
                    write(root / 'cases' / cid / 'progress.json', {'state': 'failed', 'stage': 'failed', 'error': str(exc)})
                ended.add(cid); del active[future]
            for cid in ids:
                if cid not in pending:
                    continue
                if ready(source, cid) and budget_available(cfg) and len(active) < cfg['case_workers']:
                    pending.remove(cid); active[pool.submit(case, cid)] = cid
                elif (source / 'execution.json').exists() and not ready(source, cid):
                    write(root / 'cases' / cid / 'progress.json', {'state': 'upstream_failed', 'stage': 'no_completed_localization'})
                    pending.remove(cid); ended.add(cid)
            report(root, source, ids, cfg)
            if pending or active:
                time.sleep(10)
    write(root / 'finished.json', {'finished_at': datetime.now(timezone.utc).isoformat(), 'cases_ended': len(ended)})
    report(root, source, ids, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.report_only:
        cfg = read(root / 'config.json')
        report(root, Path(cfg['source_localization']), read(root / 'cases.json'), cfg)
    else:
        run(root)
