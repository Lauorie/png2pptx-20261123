"""Read-only study report: old/new localization and native reconstruction."""
import argparse
import html
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def build(root, baseline):
    ids = load(root / 'cases.json')
    config = load(root / 'config.json')
    report = root / 'reports'; assets = report / 'study_assets'
    assets.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    md = ['# 局部bbox修正：定位与重建对照', '', '更新时间：' + stamp,
          '', ('固定历史发现观察，仅重跑局部修正与几何。' if config.get('replay_observations_from') else '全新全局与局部发现，再接局部修正、几何恢复和审核。')+
          '没有全局修正请求。默认头尺寸/线宽；1轮几何审核；尚未做几何后路径去重。',
          '', '[自动刷新网页](本轮对照_live.html) · [实验范围](../README.md)', '']
    sections = []

    def picture(src, label):
        if not src or not src.exists():
            return '等待', '<div>等待</div>'
        target = assets / (label + '_' + str(src.stat().st_mtime_ns) + '.png')
        try:
            if not target.exists():
                shutil.copy2(src, target)
            with Image.open(target) as image:
                image.verify()
        except (OSError, ValueError):
            return '图片正在更新', '<div>图片正在更新</div>'
        rel = 'study_assets/' + target.name
        return f'[![]({rel})]({rel})', f'<a href="{rel}"><img src="{rel}"></a>'

    for cid in ids:
        old_case = baseline / 'cases' / cid
        current = root / 'cases' / cid
        latest = load(current / 'latest.json')
        result = load(current / 'result.json')
        audit = load(current / 'local_review/audit.json')
        state = load(current / 'status.json')
        geom = load(root / 'reconstruction/cases' / cid / 'progress.json')
        label = f"定位 {state.get('state', 'queued')} / {state.get('stage', 'queued')}；区域 {result.get('region_count', '未定')}；几何 {geom.get('state', 'queued')} / {geom.get('stage', '等待定位')} {geom.get('ended', 0)}/{geom.get('total', '?')}"
        if audit.get('unchecked_overlaps'):
            label += f"；仍有 {len(audit['unchecked_overlaps'])} 对相交框未共同复查（不等于重复）"
        md += ['## ' + cid, '', label, '', '| v005 全局修正后 | 本轮小批局部修正 |', '|---|---|']
        images = [old_case / 'previews/05_final.png', root / latest['overlay'] if latest.get('overlay') else None,
                  baseline / 'reconstruction/cases' / cid / 'result/overlap.png', root / 'reconstruction/cases' / cid / 'result/overlap.png']
        pairs = [picture(src, cid + '_' + str(i)) for i, src in enumerate(images)]
        md += ['| ' + pairs[0][0] + ' | ' + pairs[1][0] + ' |', '', '| 旧重建叠图 | 新重建叠图 |', '|---|---|',
               '| ' + pairs[2][0] + ' | ' + pairs[3][0] + ' |', '',
               f'[新定位trace](../cases/{cid}/) · [新几何trace](../reconstruction/cases/{cid}/)', '']
        body = '<div class="pair">' + ''.join(x[1] for x in pairs) + '</div>'
        wide = root / 'ablations/context16/cases' / cid
        if wide.exists():
            late = load(wide / 'latest.json')
            wp = root / 'ablations/context16' / late['overlay'] if late.get('overlay') else None
            wide_picture = picture(wp, cid + '_wide')
            md += ['局部上下文比较：', '', '| 1.25×（主组） | 1.6×（对照） |', '|---|---|',
                   '| ' + pairs[1][0] + ' | ' + wide_picture[0] + ' |', '']
            body += '<h3>上下文1.25× / 1.6×</h3><div class="pair">' + pairs[1][1] + wide_picture[1] + '</div>'
        sections.append(f'<section id="{cid}"><h2>{cid}</h2><p>{html.escape(label)}</p>{body}</section>')
    (report / '本轮定位与重建对照.md').write_text('\n'.join(md) + '\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>局部bbox修正对照</title><style>body{font:16px system-ui;margin:24px;background:#f3f6fa}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;display:block}section{background:white;padding:16px;margin-bottom:30px}</style>'
    (report / '本轮对照_live.html').write_text(page + '<h1>局部bbox修正与重新重建</h1><p>' + stamp + ' · 每20秒刷新</p>' + ''.join(sections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            build(args.run_dir.resolve(), args.baseline.resolve())
        except (OSError, ValueError) as exc:
            print('Report update will retry:', exc, flush=True)
        runs = [args.run_dir]
        if (args.run_dir / 'ablations/context16/config.json').exists():runs.append(args.run_dir / 'ablations/context16')
        if args.once or all((r / 'reconstruction/finished.json').exists() for r in runs):
            break
        time.sleep(20)
