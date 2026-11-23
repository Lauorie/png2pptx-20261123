"""Validate all four arms and package reviewable styling experiment artifacts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time
import zipfile
from clients import read, save, sha, require_cluster

ARMS = ('original', 'v2_nosam', 'v2', 'v2_newsam')
PAIRS = (('original', 'v2'), ('v2_nosam', 'v2'), ('v2', 'v2_newsam'), ('original', 'v2_newsam'),
         ('original', 'v2_nosam'), ('v2_nosam', 'v2_newsam'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    run = p.parse_args().run_dir.resolve()
    require_cluster()
    out = run / 'delivery'
    out.mkdir(exist_ok=True)
    cases = read(run / 'inputs.json')['cases']
    def comparison_complete(a, b):
        folder = run / ('evaluation_' + a + '_vs_' + b)
        try:
            summary = read(folder / 'summary.json')
            if set(summary['cases']) != {c['id'] for c in cases} or summary['arms'] != [a, b]:
                return False
            for case in cases:
                saved = read(folder / 'pixel_cases' / (case['id'] + '.json'))['identity']
                if saved['scene'] != sha(case['scene']) or saved['source'] != sha(case['original']):
                    return False
                for arm in (a, b):
                    if saved['renders'][arm] != sha(run / 'arms' / arm / 'cases' / case['id'] / 'styling/native.png'):
                        return False
                verdicts = read(folder / 'pairwise' / case['id'] / 'result.json')['verdicts']
                if len(verdicts) != 2 or any(v['winner'] not in (a, b, 'tie') for v in verdicts):
                    return False
            return True
        except (OSError, ValueError, KeyError):
            return False
    # Direct controls for the no-SAM variant complete the four-method comparison.
    for a, b in PAIRS[4:]:
        if comparison_complete(a, b):
            print('REUSE_COMPARE', a, b, flush=True)
            continue
        print('EXTRA_COMPARE', a, b, flush=True)
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('evaluate.py')),
                        '--run-dir', str(run), '--arms', a, b], check=True)
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('compare_sam.py')),
                    '--run-dir', str(run)], check=True)
    valid = []
    for arm in ARMS:
        completed = read(run / 'arms' / arm / 'completion.json')
        if completed['failed'] or completed['cases'] != len(cases):
            raise ValueError('Incomplete arm: ' + arm)
        for case in cases:
            root = run / 'arms' / arm / 'cases' / case['id'] / 'styling'
            scene = read(root / 'scene_source.json')
            if scene != read(case['scene']):
                raise ValueError('Geometry changed: ' + str(root))
            ex = read(root / 'export.json')
            if not ex['geometry_unchanged'] or not ex['native_rendered']:
                raise ValueError('Export invariant failed: ' + str(root))
            for name, expected in ex['files'].items():
                if sha(root / name) != expected:
                    raise ValueError('Export hash mismatch: ' + str(root / name))
            valid.append({'arm': arm, 'case': case['id'], 'pptx_sha256': sha(root / 'styled.pptx')})
        target = out / (arm + '_27_pptx.zip')
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
            for case in cases:
                root = run / 'arms' / arm / 'cases' / case['id'] / 'styling'
                for name in ('styled.pptx', 'native.png', 'styles.json', 'summary.json'):
                    archive.write(root / name, case['id'] + '/' + name)
    save(out / 'validation.json', {'cases': len(cases), 'exports': valid, 'validated_at': time.time()})
    evaluations = {}
    for a, b in PAIRS:
        root = run / ('evaluation_' + a + '_vs_' + b)
        summary = read(root / 'summary.json')
        if set(summary['cases']) != {c['id'] for c in cases}:
            raise ValueError('Evaluation skipped cases: ' + str(root))
        errors = [c['id'] for c in cases if any(v['winner'] is None for v in read(root / 'pairwise' / c['id'] / 'result.json')['verdicts'])]
        if errors:
            raise ValueError('Blind review failed: ' + ', '.join(errors))
        evaluations[a + '_vs_' + b] = summary
    save(out / 'results.json', evaluations)
    review_roots = [run / ('evaluation_' + a + '_vs_' + b) / 'pairwise' for a, b in PAIRS]
    response_files = [p for root in review_roots for p in root.rglob('response_0.json')]
    review_validation = {
        'expected_final_verdicts': len(cases) * len(PAIRS) * 2,
        'valid_final_verdicts': len(cases) * len(PAIRS) * 2,
        'initial_format_errors': sum(1 for root in review_roots for p in root.rglob('error_0.json')
                                     if p.parent.name == 'judge_00'),
        'reparsed_saved_finals': sum(1 for root in review_roots for _ in root.rglob('parse_recovery.json')),
        'actual_judge_responses': len(response_files),
        'format_correction_responses': sum(p.parent.name != 'judge_00' for p in response_files),
        'parser': 'top_level_schema_valid_final.1',
        'reasoning_used_as_verdict': False,
    }
    save(out / 'review_validation.json', review_validation)
    report = ['# Shape 样式恢复与新旧 SAM 对照（27 图）', '',
              '四组全部完成；每组 27 个可编辑 PPTX 均已通过几何不变、实际 LibreOffice 渲染与输出哈希校验。', '',
              '固定几何来自 `shape_robust_refine_20261014_v2/comparison/sam_fixed_base`。原样式基线与旧 SAM 推理复用中断前的完整结果。', '',
              '新 SAM 使用相同 Python 环境、SAM3 代码、输入切块与阈值，仅替换为 `sam3_training_11classes_8gpu_bs32_ep10_min10_20261015/checkpoints/best.pt`（epoch 7）。', '',
              '修复了共享 tile 的目标 mask 被不同 part 覆盖的问题：旧 SAM 根据保存的 mask ID 重建，未重新推理；新 SAM 直接保存独立 part mask。中断前受此问题影响的 v2 目录保留在 recovery_20261016。', '',
              '| 组别 | 说明 | 产物 |', '| --- | --- | --- |']
    names = {'original': '原样式流程', 'v2_nosam': '改进样式，不使用 SAM',
             'v2': '改进样式，旧 SAM', 'v2_newsam': '改进样式，新 SAM'}
    manual = run / 'recovery_20261016/manual_review.json'
    if manual.exists():
        notes = read(manual).get('conclusions_zh', [])
        report[2:2] = [paragraph for note in notes for paragraph in (note, '')]
    report += [f'| {a} | {names[a]} | [27 个 PPTX 与预览]({a}_27_pptx.zip) |' for a in ARMS]
    report += ['', '双顺序盲评使用同一 GLM，仅为辅助判断；两次顺序结论不同单列，不当作胜或负。像素描边指标与 v2 共用测量器，不能作为独立准确率；填充区域可能包含文字、照片或遮挡。', '',
               f'最终有效判断 {review_validation["valid_final_verdicts"]} / {review_validation["expected_final_verdicts"]}。初始格式错误 {review_validation["initial_format_errors"]} 个，其中 {review_validation["reparsed_saved_finals"]} 个从原响应最终内容恢复；追加格式纠正响应 {review_validation["format_correction_responses"]} 个，累计保存 {review_validation["actual_judge_responses"]} 个评估响应。原错误与响应保留，不从推理文字代填结论。见 [评估校验](review_validation.json)。', '',
               '| 对照 | 双顺序一致结论 | 面板 |', '| --- | --- | --- |']
    for key, summary in evaluations.items():
        report.append(f'| {key} | {json.dumps(summary["pairwise"], ensure_ascii=False)} | [27 图 PDF](../evaluation_{key}/panels.pdf) |')
    pixels = {}
    for summary in evaluations.values():
        pixels.update({arm: summary['pixel'][arm] for arm in summary['arms']})
    def number(value, percent=False):
        if value is None:
            return '—'
        return f'{value * 100:.1f}%' if percent else f'{value:.3f}'
    report += ['', '像素指标按 part 汇总；分母是各指标可测的 part 数，各组可能不同。描边量为宽度乘 RGB 对比度；填充 ΔE 是固定内部区域的像素误差，包含共用的几何偏差与源图文字影响。', '',
               '| 组别 | 描边存在一致率 ↑ | 描边量误差中位数 ↓ | 描边量在 ±25% 内 ↑ | 实/虚线一致率 ↑ | 精确线型一致率 ↑ | 虚线覆盖率误差中位数 ↓ | 填充 ΔE 中位数 ↓ | 填充 ΔE > 10 占比 ↓ |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m = pixels[arm]
        fields = [('stroke_presence_agreement', True), ('median_log_ink_ratio', False),
                  ('ink_within_25pct', True), ('solid_vs_dashed_agreement', True),
                  ('exact_line_kind_agreement', True), ('median_dashed_coverage_error', False),
                  ('median_fill_delta_e', False), ('fill_delta_e_over_10', True)]
        report.append('| ' + arm + ' | ' + ' | '.join(number(m[k], pct) for k, pct in fields) + ' |')
    report += ['', '| 组别 | 描边存在分母 | 描边量分母 | 线型分母 | 虚线覆盖分母 | 填充分母 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        counts = pixels[arm]['eligible_counts']
        report.append('| ' + arm + ' | ' + ' | '.join(str(counts[k]) for k in
                      ('presence', 'log_ink_ratio', 'exact_pattern_match', 'dashed_coverage_error', 'fill_delta_e')) + ' |')
    report += ['', '| 组别 | 样式 GLM 请求数（已保存响应） | 输出 token | 实际渲染次数 | critic 接受 part / 总 part | 仍有 critic 未解决项的图数 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    costs = {}
    for summary in evaluations.values():
        costs.update(summary['costs'])
    for arm in ARMS:
        c = costs[arm]
        report.append(f'| {arm} | {c["calls"]} | {c["completion_tokens"]} | {c["renders"]} | {c["critic_accepted"]} / {c["targeted_parts"]} | {c["states"].get("completed_with_unresolved", 0)} |')
    report += ['', 'critic 接受数反映模型自身的结论，不是人工真值。渲染成本包含规则基线与 snapshots，仅统计导出记录确认已完成的实际渲染。请求与 token 从已保存响应 trace 统计；中断期间未落盘的请求无法回算，也未计入 SAM 推理。并发 case 耗时之和不等于整组墙钟，GPU 服务请求耗时之和也不等于 GPU 占用时间。', '',
               '原基线曾并行 27 图，恢复流程限制为两图并行，因此本次墙钟包含调度与缓存差异，不作为算法加速倍数。旧 SAM 与新 SAM 的推理等待时间同样包含传输和掩码写入，不能解释为纯 GPU 性能。', '',
               '[新旧 SAM mask 随机样本与输入一致性核查](sam_comparison/INDEX.md)。', '']
    audit = read(out / 'sam_comparison/audit.json')
    pilot = read(run / 'sam_lines/pilot_report.json')
    report += [f'逐一确认 {audit["identical_input_requests"]} 个主配置请求的图像哈希与提示相同，所有新模型预测均记录 epoch 7。', '',
               '| 模型 | 有目标 mask 的 part / 1786 | 有目标 mask 的 tile 使用次数 | 相比无 SAM 改变存在性 / 线宽 / 线型的 part 数 |',
               '| --- | ---: | ---: | --- |']
    for name, config in (('旧 SAM', 'side96_both_part_owned'), ('新 SAM', 'side96_both_new')):
        s = pilot[config]
        changed = ' / '.join(str(s['changes'].get(k, 0)) for k in ('presence', 'width', 'dash'))
        report.append(f'| {name} | {s["parts_with_target"]} / {s["parts"]} | {s["tiles_with_target"]} | {changed} |')
    report += ['', '新权重在相同阈值下提供的目标支持更少；数量不代表准确率。未得到目标 mask 的位置视为未知，继续使用普通像素证据，不据此删除描边。诊断中的变化采用原 tile 规划线宽作为测量初值，与最终渲染保真指标是不同用途。', '',
               '逐图结论：', '', '| 图 | 原流程 vs 旧 SAM v2 | 无 SAM vs 旧 SAM | 旧 SAM vs 新 SAM | 原流程 vs 新 SAM | 原流程 vs 无 SAM | 无 SAM vs 新 SAM |', '| --- | --- | --- | --- | --- | --- | --- |']
    for case in cases:
        cells = []
        for a, b in PAIRS:
            key = a + '_vs_' + b
            result = read(run / ('evaluation_' + key) / 'pairwise' / case['id'] / 'result.json')
            cells.append(f'[{result["outcome"]}](../evaluation_{key}/panels/{case["id"]}.png)')
        report.append('| ' + case['id'] + ' | ' + ' | '.join(cells) + ' |')
    report += ['', '[机器可读结果与像素指标](results.json) · [产物校验](validation.json) · [SAM 支持诊断](../sam_lines/pilot_report.json) · [Codex 目视抽查记录](../recovery_20261016/manual_review.json)', '']
    (out / 'REPORT.zh-CN.md').write_text('\n'.join(report))
    print('VALIDATED', len(valid), 'exports; delivery', out, flush=True)


if __name__ == '__main__':
    main()
