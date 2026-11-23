"""Read-only review of evidence flow and committed recovery stages.

Counts describe execution, never validated recall. This module does not choose
targets, edit geometry, or decide whether a recovered line is correct.
"""
import json
import shutil
from pathlib import Path


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def snapshot(out, phase):
    out = Path(out); target = out / 'stages' / phase
    target.mkdir(parents=True, exist_ok=True)
    for name in ['scene.json', 'result/instances.png', 'result/styled.png', 'result/overlap.png']:
        dest = target / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / name, dest)


def write_case(out):
    out = Path(out)
    groups = []
    for folder in sorted((out / 'groups').glob('G*')):
        evidence = read(folder / 'shared_evidence.json', {})
        decision = read(folder / 'decision.json', {})
        groups.append(dict(group=folder.name,
            shared_observations=evidence.get('observations', 0),
            additional_mask_pixels=evidence.get('additional_pixels', 0),
            followups=decision.get('followups', []),
            decision_available=(folder / 'decision.json').exists()))
    phases = {}
    repairs = []
    for phase in ['continuation', 'missing']:
        summary = read(out / phase / 'parallel_summary.json', {})
        phases[phase] = dict(
            queued=len(read(out / (phase + '_queue.json'), [])),
            bundled_tasks=summary.get('tasks', 0),
            committed=summary.get('committed', 0),
            issues=summary.get('issues', []))
        for folder in sorted((out / phase).glob(phase + '*')):
            if (folder / 'rebase' / 'completed.json').exists():
                folder = folder / 'rebase'
            done = read(folder / 'completed.json')
            if done is None:
                if (folder / 'error.json').exists():
                    repairs.append(dict(phase=phase, folder=str(folder.relative_to(out)), status='failed',
                                        replace=0, new_paths=0, shared_observations=0, existing_paths=0))
                continue
            attempts = read(folder / 'attempts.json', [])
            last = folder / attempts[-1]['folder'] if attempts else folder / 'local'
            q = read(last / 'prompt.json', {})
            repairs.append(dict(phase=phase, folder=str(folder.relative_to(out)), status=done.get('status'),
                                replace=len(done.get('replace', [])), new_paths=len(done.get('rows', [])),
                                shared_observations=q.get('shared_evidence', {}).get('native_observations', 0),
                                existing_paths=len(q.get('existing', []))))
    scenes = [('初次恢复', 'stages/initial/'), ('相关连接补全后', 'stages/continuation/'), ('全局补漏后', '')]
    stages = []
    for label, prefix in scenes:
        rows = read(out / prefix / 'scene.json', {}).get('arrows', [])
        stages.append(dict(stage=label, paths=len(rows), identities=len({r['parent_id'] for r in rows})))
    coverage=read(out/'coverage_status.json',{'state':'not_recorded'})
    audit = dict(case=out.name, visual_recall_verified=False, groups=groups, phases=phases, repairs=repairs,coverage=coverage,
                 stages=stages, explanation='Execution counts only; added or replaced paths can still be wrong or duplicated.')
    write(out / 'recall_audit.json', audit)
    lines = [f'# {out.name}：漏线恢复核对', '',
        '重点核对原图中先前缺失的线是否完整恢复。以下数量是执行记录，不是补漏准确率；重复和错误新增都需目视检查。', '',
        '[整图结果](RESULTS.md) · [阶段与请求数据](recall_audit.json)', '',
        '补漏发现状态：'+coverage['state']+'。失败或未执行不能解释为没有遗漏。', '',
        '**原图**', '', '![](clean.png)', '',
        '| 阶段 | 实际路径数 | 实例数 | 样式 | 原图叠加 |', '|---|---:|---:|---|---|']
    for (label, prefix), stage in zip(scenes, stages):
        lines.append(f'| {label} | {stage["paths"]} | {stage["identities"]} | '
                     f'![]({prefix}result/styled.png) | ![]({prefix}result/overlap.png) |')
    lines += ['', '**信息是否传入初始身份请求**', '',
              '共享像素表示分割证据，不能当作已经重建。点击证据图、模型决定查看是否提出相关缺段。', '',
              '| 组 | 共享证据条目 | 额外覆盖像素 | 补全建议 | 实际证据图 | 模型决定 |',
              '|---|---:|---:|---:|---|---|']
    for g in groups:
        gid = g['group']
        lines.append(f'| {gid} | {g["shared_observations"]} | {g["additional_mask_pixels"]} | '
                     f'{len(g["followups"])} | [图](groups/{gid}/input/proposals.png) | '
                     f'[输出](groups/{gid}/decision.json) |')
    lines += ['', '**相关连接补全与全局补漏**', '',
              '“complete”可表示模型认为无需修改。“repair”的新增路径也可能是替换旧路径，不能直接计作补回一条箭头。', '',
              '| 阶段 | 建议数 | 合并后的请求任务 | 提交结果 | 失败数 |', '|---|---:|---:|---:|---:|']
    for phase, summary in phases.items():
        lines.append(f'| {phase} | {summary["queued"]} | {summary["bundled_tasks"]} | '
                     f'{summary["committed"]} | {len(summary["issues"])} |')
    lines += ['', '| 请求 | 状态 | 输入已有路径 | 共享证据条目 | 替换路径 | 输出路径 | 修复前 / 后 |',
              '|---|---|---:|---:|---:|---:|---|']
    for r in repairs:
        folder = r['folder']
        lines.append(f'| [{folder}]({folder}/README.md) | {r["status"]} | {r["existing_paths"]} | '
                     f'{r["shared_observations"]} | {r["replace"]} | {r["new_paths"]} | '
                     f'[前]({folder}/before.png) / [后]({folder}/after.png) |')
    lines += ['', '**人工结论：待查看原图与最终叠图。**', '']
    (out / 'RECALL.md').write_text('\n'.join(lines))


def write_run(run, ids):
    run = Path(run)
    lines = ['# 本轮重点：之前漏掉的箭头是否补出', '',
        '先看 001、ds004、ds012、gpt_ppt_page_01；同时检查 073、ds022 等其他漏检。所有 40 张使用同一算法。', '',
        '每张完成后可查看初次恢复、相关连接补全、全局补漏三个阶段，以及共享 SAM 证据和实际请求。'
        '新增实例数不代表补漏成功；该表不自动宣称正确率。', '',
        '[所有整图](RESULTS.md) · [旧版 / 本轮对照](COMPARISON.md) · [实验说明](PLAN.md)', '',
        '| case | 状态 | 传入额外证据的组 | 相关补全提交 / 任务 | 全局补漏提交 / 任务 | 阶段与请求 |',
        '|---|---|---:|---|---|---|']
    for cid in ids:
        folder = run / 'cases' / cid; audit = read(folder / 'recall_audit.json')
        if audit is None:
            lines.append(f'| {cid} | 等待完整结果 | | | | |'); continue
        state = read(folder / 'completed.json', {}).get('state', '待确认')
        phases = audit['phases']
        cells = [cid, state, str(sum(g['additional_mask_pixels'] > 0 for g in audit['groups']))]
        cells += [f'{phases[p]["committed"]} / {phases[p]["bundled_tasks"]}' for p in ['continuation', 'missing']]
        cells += [f'[核对漏线](cases/{cid}/RECALL.md)']
        lines.append('| ' + ' | '.join(cells) + ' |')
    tmp = run / 'RECALL.md.tmp'; tmp.write_text('\n'.join(lines) + '\n'); tmp.replace(run / 'RECALL.md')
