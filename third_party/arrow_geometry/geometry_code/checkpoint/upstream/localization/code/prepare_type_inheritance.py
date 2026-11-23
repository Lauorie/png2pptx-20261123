"""Freeze A-only localization replay. Preparation never calls a model."""
import argparse
import difflib
import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import contracts as C
from run import prepare


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--partition', choices=['development', 'validation'], required=True)
    args = parser.parse_args()
    project = next(p for p in Path(__file__).resolve().parents if (p / 'PROJECT.json').is_file())
    out = args.run_dir.resolve();out.relative_to(project)
    if out.exists():
        raise ValueError('Use a new immutable run directory')
    plan = read(project / 'analysis/pptbench_failure_insights_20261104/candidate_review_scope.json')
    assert set(plan['variants']) == {'A'}
    baseline = project / plan['baseline_run'] / 'localization'
    code = Path(__file__).resolve().parent
    version = read(project / 'CURRENT_DEBUG_VERSION.json')['version']
    if sha(code / 'review_type_policy.py') != sha(project / 'versions' / version / 'code/checkpoint/upstream/localization/code/review_type_policy.py'):
        raise ValueError('Snapshot the current code before preparing the run')
    ids = [row['case_id'] for row in plan['cases'] if row['partition'] == args.partition]
    cfg = read(baseline / 'config.json')
    cfg.update(review_inherit_existing_type=True, replay_observations_from=str(baseline),
               new_discovery_calls=False, new_global_discovery_calls=False, new_local_discovery_calls=False,
               new_local_review_calls=True, comparison_run=str(out / 'baseline_view'),
               comparison_title='v049 候选整理结果', request_slot_dir=str(out / 'api_slots'),
               experiment_variant='A_type_inheritance_only', code_version=version,
               reconstruction_enabled=False, localization_only=True)
    assert cfg['thinking'] == {'type': 'enabled'} and 'reasoning_effort' not in cfg
    preparation = out.parent.parent / 'preparation'
    preparation.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + '_', dir=preparation))
    constraints = staging / 'constraints';constraints.mkdir()
    for name in ['AGENTS.md', 'README.md', 'PROJECT.json', 'docs/TASK.md', 'docs/PIPELINE.md',
                 'analysis/pptbench_failure_insights_20261104/候选整理实验范围.md',
                 'analysis/pptbench_failure_insights_20261104/candidate_review_scope.json']:
        source = project / name
        target = constraints / name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (constraints / 'ADOPTED_SCOPE.md').write_text(
        '# 本次实际采用的范围\n\n仅执行 A：已有候选继承重建类型，框整理任务优化留到下阶段。'
        '小箭头通常更适合普通线条，大型装饰箭头通常更适合块状轮廓；这是表达偏好，'
        '没有新增尺寸阈值或重分类器。输入仍为局部 CLEAN 和 BOXES。'
        '保留原分组、修框、局部新增、扩图及最多三轮配置。'
        '模型 glm-5.3-flash，thinking enabled，不发送 reasoning_effort，max_tokens=113000。'
        'API 并发上限 96，每图局部并发 32；止于整理和 handoff。\n')
    write(staging / 'config.json', cfg)
    prepare(out, baseline, code, staging / 'config.json', constraints, ids)
    provenance = []
    for cid in ids:
        source = baseline / 'cases' / cid
        observed = read(out / 'replay_observations' / (cid + '.json'))['regions']
        compressed = C.merge_near_identical(observed, cfg['observation_merge_iou'])
        assert compressed == read(source / 'compressed_observations.json')['regions'], cid
        original = read(source / 'latest.json')
        preview = baseline / original['overlay']
        target = out / 'baseline_view/cases' / cid / 'representation.png'
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(preview, target)
        provenance.append(dict(case_id=cid, original_observations=str(source / 'observations.json'),
            observation_sha256=sha(out / 'replay_observations' / (cid + '.json')),
            expected_compressed=str(source / 'compressed_observations.json'),
            expected_compressed_sha256=sha(source / 'compressed_observations.json'),
            pre_review_candidates_identical=True, source_png_identical=sha(out / 'inputs' / cid / 'source.png') == sha(baseline / 'inputs' / cid / 'source.png')))
    write(out / 'A_provenance.json', dict(baseline=str(baseline), code_version=version,
          variant='A', partition=args.partition, cases=provenance, frozen_input_verified=True,
          B_enabled=False, global_image_added=False, type_separated_batching=False,
          geometry_executed=False, formal_accuracy_measured=False))
    old = read(baseline / 'code/draft_prompt_blocks.json')
    new = read(out / 'code/draft_prompt_blocks.json')
    text = []
    for key in old:
        if old[key] != new[key]:
            assert key in ['review_system', 'review_user', 'review_output']
            text.extend(difflib.unified_diff(old[key].splitlines(True), new[key].splitlines(True),
                                            fromfile='v049/' + key, tofile='A/' + key))
    (out / 'prompt_changes.diff').write_text(''.join(text))
    write(out / 'PROMPTS.json', {k: new[k] for k in ['review_system', 'review_user', 'review_output', 'boundary_task']})
    manifest = read(out / 'manifest.json')
    for folder in ['baseline_view']:
        for p in (out / folder).rglob('*'):
            if p.is_file():manifest['files'][str(p.relative_to(out))] = sha(p)
    for name in ['A_provenance.json', 'PROMPTS.json', 'prompt_changes.diff']:
        manifest['files'][name] = sha(out / name)
    write(out / 'manifest.json', manifest)
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    (out / 'RESULTS.md').write_text(
        '# A：候选整理继承已有类型\n\n'
        f'准备时间：{created}。状态：prepared；尚未发起模型请求。\n\n'
        f'代码归档：{version}。分组：{args.partition}，case：{", ".join(ids)}。\n\n'
        '仅改变已有候选的类型继承协议；原始回复、继承审计和映射结果分别保留。'
        '分组、框整理、局部新增与裁图规则沿用 v049，不执行 B。'
        '输入发现结果复用 v049，重压缩后逐字段核对一致。\n\n'
        'glm-5.3-flash，thinking enabled，省略 reasoning_effort，max_tokens 113000；'
        '每请求总预算最多 2 次实际调用，总/空闲超时 2400 秒。'
        'API 并发上限 96，每图局部并发 32；case_workers 沿用 24，实际由样本数限制。\n\n'
        '止于完整 local_review、候选框报告和 handoff；不执行几何、SAM、后补漏、样式或 PPT 评分。'
        '类型继承是表达选择保护，不把所有普通/块状表示互换都当成语义错误。'
        '后续需检查目标保留、范围、分组和真实块状箭头有无退化。\n\n'
        '[实时报告](reports/live.html) · [输入与来源](A_provenance.json) · '
        '[Prompt 差异](prompt_changes.diff) · [实际配置](config.json)\n')
    print(json.dumps(dict(run_dir=str(out), cases=ids, code_version=version,
                         verified_inputs=len(provenance), model_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
