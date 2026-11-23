"""Apply the new scope stage to frozen, organized SAM additions without rerunning SAM."""
import argparse
import copy
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import experiment as E
import addition_scope_review as S
from transport.model import Model


def run_case(root, cid, cfg, model):
    inputs = root / 'inputs' / cid; out = root / 'cases' / cid
    image = Image.open(inputs / 'source.png').convert('RGB')
    baseline = E.read(inputs / 'baseline_native.json')
    all_rows = E.read(inputs / 'reviewed_additions.json')
    requested = cfg.get('review_selection', {}).get(cid)
    ids = set(requested if requested is not None else [r['id'] for r in all_rows])
    if not ids.issubset({r['id'] for r in all_rows}): raise ValueError('Unknown selected candidate')
    selected = [r for r in all_rows if r['id'] in ids]
    E.write(out / 'status.json', dict(state='running', stage='sam_addition_scope_review', updated_at=E.now()))
    before, _ = E.append_only(baseline, all_rows, image.size)
    E.plot(image, before, out / 'before.png')
    kept, audit = S.review(root, cid, selected, baseline, image, model, cfg)
    rejected = ids - {r['id'] for r in kept}
    retained = [copy.deepcopy(r) for r in all_rows if r['id'] not in rejected]
    final, preservation = E.append_only(baseline, retained, image.size)
    global_rows, handoff, coordinate_audit = E.export_additive(cid, final, inputs / 'source.png')
    pending = E.read(inputs / 'pending_additions.json', [])
    upstream = E.read(inputs / 'source_result.json', {})
    issues = list(audit['issues'])
    if upstream.get('state') != 'completed':
        issues.append(dict(stage='upstream', state=upstream.get('state'),
                           note='Historical upstream issues retained; not a new scope request failure'))
    untested = [r['id'] for r in all_rows if r['id'] not in ids]
    result = dict(case_id=cid, state='completed_with_issues' if issues else 'completed',
        native_regions=final, regions=global_rows, issues=issues,
        baseline_count=len(baseline), baseline_preserved=True, reviewed_additions_count=len(retained),
        scope_reviewed_count=len(selected), scope_untested_ids=untested,
        scope_all_additions_reviewed=not untested, addition_scope_review=audit,
        pending_count=len(pending), geometry_executed=False, formal_accuracy_measured=False)
    for name, value in [('current_native.json', final), ('reviewed_additions.json', retained),
                        ('pending_additions.json', pending), ('baseline_preservation.json', preservation),
                        ('coordinate_audit.json', coordinate_audit), ('reconstruction_handoff.json', handoff),
                        ('result.json', result)]:
        E.write(out / name, value)
    E.plot(image, final, out / 'current.png')
    E.write(out / 'status.json', dict(state=result['state'], stage='finished', updated_at=E.now()))


def report(root):
    cfg = E.read(root / 'config.json'); cases = E.read(root / 'cases.json')
    rows = []; sections = []; finished = 0
    for cid in cases:
        out = root / 'cases' / cid
        result = E.read(out / 'result.json', {})
        s = E.read(out / 'status.json', {'state': 'waiting'})
        audit = E.read(out / 'addition_scope_review' / 'audit.json', {})
        finished += int(s['state'] in ('completed', 'completed_with_issues', 'failed'))
        rows.append(dict(case=cid, state=s['state'], selected=audit.get('before_count'),
                         rejected=audit.get('rejected_count'), uncertain=audit.get('uncertain_count')))
        sections += [f'## {cid}', '', f"状态：{s['state']}；本轮审核 {audit.get('before_count', '—')} 个新增候选；排除 {audit.get('rejected_count', '—')}；不确定 {audit.get('uncertain_count', '—')}。", '',
                     '| 原始定位＋SAM整理后新增 | 仅删除明确误检后 |', '|---|---|',
                     f'| [![之前](cases/{cid}/before.png)](cases/{cid}/before.png) | [![之后](cases/{cid}/current.png)](cases/{cid}/current.png) |', '']
        if result.get('scope_untested_ids'):
            sections += [f"本轮未审核的其余新增框原样保留（{len(result['scope_untested_ids'])} 个）；不是整图范围验收。", '']
        for a in audit.get('decisions', []):
            trace = Path(a['trace']).relative_to(root)
            sections += [f"### {a['id']}：{a['decision']}（{a['elapsed_seconds']:.1f}秒）", '',
                         a['reason'], '', f'[请求与推理]({trace}/request/) · [坐标映射]({trace}/frames.json)', '',
                         '| 模型完整原图 | 模型局部目标图（另给同尺寸干净图） |', '|---|---|',
                         f'| [![GLOBAL](<{trace}/global_clean.png>)](<{trace}/global_clean.png>) | [![LOCAL](<{trace}/local_target.png>)](<{trace}/local_target.png>) |', '']
    E.write(root / 'progress.json', dict(updated_at=E.now(), finished=finished, total=len(cases), cases=rows))
    text = ['# SAM新增候选范围审核：小实验', '',
            f'更新：{E.now()}；结束 {finished}/{len(cases)} 张；只审核清单内的新增候选，不重新运行定位、SAM或几何。', '',
            '流程：既有SAM新增框整理 → 独立范围审核 → 几何重建交接。仅明确reject删除；uncertain/失败保留原候选并记录问题。原定位框只读。', '',
            '每个候选一次模型请求：完整原图＋局部干净图＋局部目标角标图。只返回保留/排除/不确定及一句依据。旧语义描述不送入模型。', '',
            '默认思考，未设置reasoning_effort。样本是否通过需看视觉证据，完成数不是准确率；历史上游issues独立保留。', '',
            '[实验说明](README.md) · [配置](config.json) · [来源](provenance.json) · [输入与代码哈希](manifest.json)', ''] + sections
    dest = root / 'RESULTS.md'; temp = root / '.RESULTS.md.tmp'; temp.write_text('\n'.join(text) + '\n'); temp.replace(dest)


def run(root):
    lock = (root / 'run.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in E.read(root / 'manifest.json')['files'].items():
        if E.sha256(root / name) != digest: raise ValueError('Frozen input changed: ' + name)
    cfg = E.read(root / 'config.json'); model = Model(cfg, run_dir=root)
    E.write(root / 'execution.json', dict(state='running', pid=os.getpid(), started_at=E.now()))
    with ThreadPoolExecutor(max_workers=cfg.get('review_case_workers', 4)) as pool:
        jobs = {pool.submit(run_case, root, cid, cfg, model): cid for cid in E.read(root / 'cases.json')}
        for future in as_completed(jobs):
            cid = jobs[future]
            try: future.result()
            except Exception as exc:
                E.write(root / 'cases' / cid / 'status.json', dict(state='failed', error=str(exc)))
            report(root)
    E.write(root / 'execution.json', dict(state='finished', finished_at=E.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(); run(args.run_dir.resolve())
