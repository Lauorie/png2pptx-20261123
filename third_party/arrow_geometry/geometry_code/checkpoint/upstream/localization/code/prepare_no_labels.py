"""Freeze the historical 24-case review replay with candidate descriptions omitted."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import contracts as C
from run import prepare


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--case-list', required=True, type=Path)
    args = parser.parse_args()
    project = next(p for p in Path(__file__).resolve().parents if (p / 'PROJECT.json').is_file())
    out = args.run_dir.resolve()
    out.relative_to(project)
    if out.exists():
        raise ValueError('Use a new immutable run directory')
    case_source = args.case_list.resolve()
    ids = read(case_source)
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError('Expected the requested 24 distinct cases')
    baseline = project / read(project / 'PROJECT.json')['baseline_run'] / 'localization'
    parent = project / 'experiments/candidate_review_type_inheritance/runs/A_dev8_20261104_02'
    code = Path(__file__).resolve().parent
    version = read(project / 'CURRENT_DEBUG_VERSION.json')['version']
    archived = project / 'versions' / version / 'code/checkpoint/upstream/localization/code'
    for p in code.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            if sha(p) != sha(archived / p.relative_to(code)):
                raise ValueError('Snapshot the working code before preparation: ' + str(p))
    cfg = read(parent / 'config.json')
    cfg.update(review_omit_candidate_labels=True, review_inherit_existing_type=True,
               replay_observations_from=str(baseline), comparison_run=str(out / 'baseline_view'),
               comparison_title='v049 候选整理结果', request_slot_dir=str(out / 'api_slots'),
               experiment_variant='A_plus_omit_candidate_labels', code_version=version,
               reconstruction_enabled=False, localization_only=True)
    assert cfg['thinking'] == {'type': 'enabled'} and 'reasoning_effort' not in cfg
    assert cfg['max_tokens'] == 113000 and cfg['api_concurrency'] == 96
    assert cfg['case_workers'] == 24 and cfg['local_workers'] == 32
    assert not cfg.get('adaptive_review', False)
    prep = out.parent.parent / 'preparation'
    prep.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + '_', dir=prep))
    constraints = staging / 'constraints'
    for name in ['AGENTS.md', 'README.md', 'PROJECT.json', 'docs/TASK.md', 'docs/PIPELINE.md',
                 'CURRENT_DEBUG_VERSION.json']:
        dest = constraints / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, dest)
    (constraints / 'ADOPTED_SCOPE.md').write_text(
        '用户于 2026-11-04 授权：使用删除描述，跑实验的 24 图。\n'
        '沿用历史固定 24 图名单，不重新抽样。复用 v049 发现输入，保留 A 类型继承；'
        '所有整理请求（后续轮次、扩图及拆批重试均包含）不向模型提供候选 label 字段。'
        '其他候选字段、图片和整理规则不变，模型仍可输出新 label。'
        '完整运行最多三轮 local_review 和 handoff，暂不运行几何、SAM、样式或 PPT。\n'
        '当前实验不修改职责/分组规则或跨批次调度；这些问题单独观察。\n')
    write(staging / 'config.json', cfg)
    prepare(out, baseline, code, staging / 'config.json', constraints, ids)
    provenance = []
    for cid in ids:
        observed = read(out / 'replay_observations' / (cid + '.json'))['regions']
        old_compressed = baseline / 'cases' / cid / 'compressed_observations.json'
        assert C.merge_near_identical(observed, cfg['observation_merge_iou']) == read(old_compressed)['regions'], cid
        latest = read(baseline / 'cases' / cid / 'latest.json')
        dest = out / 'baseline_view/cases' / cid / 'representation.png'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline / latest['overlay'], dest)
        assert sha(out / 'inputs' / cid / 'source.png') == sha(baseline / 'inputs' / cid / 'source.png'), cid
        provenance.append({'case_id': cid, 'source_png_identical': True,
                           'pre_review_candidates_identical': True,
                           'observations_sha256': sha(out / 'replay_observations' / (cid + '.json')),
                           'baseline_compressed_sha256': sha(old_compressed)})
    write(out / 'experiment_provenance.json', {
        'case_list_source': str(case_source), 'case_list_sha256': sha(case_source),
        'baseline': str(baseline), 'configuration_parent': str(parent), 'code_version': version,
        'only_algorithm_change_from_A': 'omit candidate label fields from all local-review model inputs',
        'cases': provenance, 'geometry_executed': False, 'accuracy_measured': False,
        'historical_suite_is_not_independent_holdout': True})
    write(out / 'PROMPTS.json', read(code / 'draft_prompt_blocks.json'))
    blocks = read(code / 'draft_prompt_blocks.json')
    (out / 'PROMPTS.md').write_text('# 实际整理协议\n\n'
        'CANDIDATES_JSON 使用 id、bbox、render_type、grouping、extent，删除 label。'
        '实际每个请求的完整文本和图片哈希见 cases/*/local_review/*/*/*/request/。\n\n' +
        '\n\n'.join('## ' + key + '\n\n' + blocks[key]
                      for key in ['review_system', 'review_user', 'review_output', 'boundary_task']))
    (out / 'EXPERIMENT_SETUP.md').write_text(
        '# 整理时删除描述：历史固定 24 图\n\n'
        f'代码归档：{version}。原始名单：{case_source.relative_to(project)}。\n\n'
        '只删除整理模型输入中的候选 label，保留 A 的类型继承和其余协议。'
        '所有轮次、扩图和拆批重试使用同一开关；不删除后台候选描述，keep 仍保留原描述。\n\n'
        '输入发现结果全部复用 v049，重压缩后逐字段核验一致。比较图为 v049；'
        '由于 v049 尚未继承类型，该对照包含 A 类型继承与删除描述两项差异，不能把全部变化都归因于删除描述。\n\n'
        '停止于候选整理与 handoff，不运行几何、SAM、样式或 PPT。'
        '重点核查分组、遗漏、框外延伸、容器误识别与跨批次重复责任；执行完成不代表质量通过。\n\n'
        '模型 glm-5.3-flash，thinking enabled，不传 reasoning_effort，max_tokens=113000；'
        'API 并发 96、图并发 24、每图局部并发 32、最多三轮、每请求最多两次实际尝试、超时 2400 秒。\n\n'
        '这份历史 24 图不是独立验证集，也不包含 0961；0961 的固定输入对照保存在 A 运行的分析目录。\n\n'
        '样本：' + '、'.join(ids) + '\n\n'
        '[实时两栏图](reports/24图_实时对照.md) · [网页](reports/live.html) · '
        '[实际配置](config.json) · [输入来源与哈希](experiment_provenance.json)\n')
    manifest = read(out / 'manifest.json')
    for p in (out / 'baseline_view').rglob('*'):
        if p.is_file():
            manifest['files'][str(p.relative_to(out))] = sha(p)
    for name in ['experiment_provenance.json', 'PROMPTS.json', 'PROMPTS.md', 'EXPERIMENT_SETUP.md']:
        manifest['files'][name] = sha(out / name)
    write(out / 'manifest.json', manifest)
    from presentation import publish
    publish(out)
    print(json.dumps({'prepared': str(out), 'cases': len(ids), 'code_version': version,
                      'pre_review_inputs_verified': len(provenance), 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
