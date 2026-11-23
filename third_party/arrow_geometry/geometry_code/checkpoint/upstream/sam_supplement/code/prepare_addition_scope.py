"""Freeze a scope-only replay from completed SAM addition organization results."""
import argparse
import json
import shutil
from pathlib import Path
import experiment as E


def prepare(source, root, cases=None, selection=None, api_concurrency=128):
    source = Path(source).resolve(); root = Path(root).resolve()
    if root.exists(): raise ValueError('Use a new run directory; historical artifacts are immutable')
    work = Path(__file__).resolve().parent
    cases = list(cases or E.read(source / 'cases.json'))
    if not cases or len(cases) != len(set(cases)): raise ValueError('Need distinct case IDs')
    # Check readiness before creating the new run.
    for cid in cases:
        status = E.read(source / 'cases' / cid / 'status.json', {})
        if status.get('state') not in ('completed', 'completed_with_issues'):
            raise ValueError('Source SAM organization is not complete: ' + cid)
        for path in [source / 'inputs' / cid / 'source.png', source / 'inputs' / cid / 'baseline_native.json',
                     source / 'cases' / cid / 'reviewed_additions.json', source / 'cases' / cid / 'result.json']:
            if not path.is_file(): raise ValueError('Missing source artifact: ' + str(path))
        if selection and cid in selection:
            available = {r['id'] for r in E.read(source / 'cases' / cid / 'reviewed_additions.json')}
            if not set(selection[cid]).issubset(available): raise ValueError('Unknown selected additions: ' + cid)
    cfg = E.read(source / 'config.json')
    cfg.update(source_run=str(source), sam_addition_scope_review_enabled=True, sam_scope_workers=4,
               sam_scope_global_long_edge=2048, sam_scope_local_long_edge=1536,
               sam_scope_context_scale=2.0, sam_scope_min_context_px=96,
               review_case_workers=4, api_concurrency=api_concurrency,
               thinking={'type': 'enabled'}, missing_cache_only=False)
    cfg.pop('reasoning_effort', None)
    cfg.pop('review_selection', None)
    if selection is not None: cfg['review_selection'] = selection
    root.mkdir(parents=True)
    shutil.copytree(work, root / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sources = {}
    for cid in cases:
        out = root / 'inputs' / cid; out.mkdir(parents=True)
        mapping = {'source.png': source / 'inputs' / cid / 'source.png',
                   'baseline_native.json': source / 'inputs' / cid / 'baseline_native.json',
                   'reviewed_additions.json': source / 'cases' / cid / 'reviewed_additions.json',
                   'source_result.json': source / 'cases' / cid / 'result.json'}
        for name, path in mapping.items():
            expected = E.sha256(path); shutil.copy2(path, out / name)
            if E.sha256(path) != expected or E.sha256(out / name) != expected:
                raise ValueError('Input changed during freeze: ' + str(path))
            sources[str((out / name).relative_to(root))] = dict(source=str(path), sha256=expected)
        E.write(out / 'pending_additions.json', E.read(source / 'cases' / cid / 'pending_additions.json', []))
    E.write(root / 'config.json', cfg); E.write(root / 'cases.json', cases)
    E.write(root / 'provenance.json', dict(created_at=E.now(), source_review=str(source),
        working_source=str(work), inputs=sources, fresh_scope_calls=True,
        new_sam_calls=False, new_geometry_calls=False))
    (root / 'README.md').write_text(
        '# SAM新增候选独立范围审核\n\n既有新增框整理 → 本轮范围审核 → 几何交接。'
        '完整原图＋局部干净图＋目标标记图，仅返回保留/排除/不确定。明确排除才删除；不确定/失败原样保留并报告。'
        '原框不变，默认思考，不重复调用发现、SAM或几何。\n\n'
        '运行：配置ARROW_API_KEY_FILE后，python -B code/replay_addition_scope.py --run-dir 本目录。'
        '结果见RESULTS.md；几何运行的source_localization/bbox_discovery_run指向本目录。'
        '本轮交接必须结束后才能启动该图几何；旧缓存交接不会自动改变。\n')
    files = [p for name in ('code', 'inputs') for p in (root / name).rglob('*') if p.is_file()]
    files += [root / name for name in ('config.json', 'cases.json', 'provenance.json')]
    E.write(root / 'manifest.json', dict(files={str(p.relative_to(root)): E.sha256(p) for p in files}))
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--selection-json', type=Path)
    parser.add_argument('--api-concurrency', type=int, default=128)
    args = parser.parse_args()
    print(prepare(args.source_run, args.run_dir, args.cases,
                  E.read(args.selection_json) if args.selection_json else None, args.api_concurrency))
