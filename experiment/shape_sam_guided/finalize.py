"""Package one complete raw method; no per-case selection or geometry edits."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from core import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, target = args.run_dir, args.output
    aggregate = read(root / 'aggregate.json')
    native = read(root / 'native_review/completion.json')
    items = read(root / 'manifest.json')['images']
    if aggregate['cases'] != len(items) or native['cases'] != len(items):
        raise ValueError('Evaluation/rendering incomplete')
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for result in aggregate['results']:
        cid = result['id']
        metrics = result['metrics'][args.method]
        source = root / args.method / 'cases' / cid / 'fitting'
        if sha(source / 'scene.json') != metrics['scene_sha256']:
            raise ValueError('Evaluated scene changed: ' + cid)
        if sha(source / 'geometry_final.pptx') != metrics['ppt_sha256']:
            raise ValueError('Evaluated PPT changed: ' + cid)
        dest = target / 'pptx' / (cid + '.pptx')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != metrics['ppt_sha256']:
            raise ValueError('Existing delivery differs: ' + cid)
        if not dest.exists():
            shutil.copy2(source / 'geometry_final.pptx', dest)
        records.append({'id': cid, 'method': args.method, 'pptx': str(dest.relative_to(target)),
                        'ppt_sha256': metrics['ppt_sha256'], 'scene_sha256': metrics['scene_sha256'],
                        'source': str(source.resolve())})
    for name in ['metrics.csv', 'aggregate.json']:
        shutil.copy2(root / name, target / name)
    shutil.copy2(root / 'native_review/overview.pdf', target / 'overview.pdf')
    save(target / 'manifest.json', {'method': args.method, 'cases': records,
         'per_case_selection': False, 'postfit_geometry_edits': False,
         'note': 'Raw full-method output; model votes are auxiliary, not human ground truth.',
         'preferences': aggregate['preferences']})
    lines = ['# Shape 几何交付', '',
             f'全部 {len(records)} 张统一使用 `{args.method}`。没有逐图择优或导出后几何修改。',
             '文件只包含可编辑 shape 几何；文字、外部连接箭头和图片内容由其他阶段处理。', '',
             '[原生 PPT 对比总览](overview.pdf) · [指标](metrics.csv) · [完整评价记录](aggregate.json)', '',
             '模型双顺序评价是辅助证据，不是人工真值。生成波动、未解决形状和评价不一致应结合原图复核。', '',
             '| 样本 | 可编辑 PPT |', '|---|---|']
    lines += [f'| {row["id"]} | [PPT]({row["pptx"]}) |' for row in records]
    costs = aggregate['usage'][args.method]
    if costs.get('replayed_calls'):
        lines += ['', '本组为受控增量消融：完全相同的有效请求复用冻结对照响应，'
                  '变化的请求重新推理。实际调用不包含复用；logical calls/tokens 包含复用，'
                  '用于区别增量成本与新图完整运行成本。']
    lines += ['', '原始实验目录：`' + str(root.resolve()) + '`', '']
    (target / 'REPORT.zh-CN.md').write_text('\n'.join(lines))
    archive = target / (args.method + f'_{len(records)}pptx.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
        for path in sorted((target / 'pptx').glob('*.pptx')):
            out.write(path, 'pptx/' + path.name)
        for name in ['manifest.json', 'REPORT.zh-CN.md', 'metrics.csv', 'aggregate.json', 'overview.pdf']:
            out.write(target / name, name)
    print(json.dumps({'cases': len(records), 'archive': str(archive), 'sha256': sha(archive)}))


if __name__ == '__main__':
    main()
