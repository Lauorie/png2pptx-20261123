"""Two-column per-view discovery comparison, refreshed with the main report."""
import html
from pathlib import Path
from transport.common import read_json as read, write_json as write


def publish_local(root, overlay, updated):
    cfg = read(root / 'config.json')
    if not cfg.get('frozen_global_from'):
        return
    base = Path(cfg['frozen_global_from'])
    ids = read(root / 'cases.json')
    dest = root / 'reports'; assets = dest / 'local_assets'
    assets.mkdir(parents=True, exist_ok=True)
    md = ['# 24图局部发现：规则对齐前后', '',
          '固定 v015 全局结果，仅重跑 LOCAL；这里只看局部发现原始输出，尚未经过后续调整。',
          '全局候选没有输入局部模型；共同规则、图像与坐标保持既定契约。框数不是准确率。', '',
          '[后续最终定位对照](24图_实时对照.md) · [自动刷新](local_live.html)', '', '更新：' + updated, '']
    sections = []; completed = failed = 0
    for cid in ids:
        md += ['## ' + cid, '']
        cells = []
        for view in read(root / 'inputs' / cid / 'frames.json')['local_views']:
            vid = view['id']; row = []; hrow = []
            for label, source in [('v015', base), ('对齐后', root)]:
                folder = source / 'cases' / cid / 'local_discovery' / vid
                result = folder / 'result.json'; failure = folder / 'unresolved.json'
                if result.exists():
                    value = read(result)
                    name = f'{cid}_{vid}_{label}_{result.stat().st_mtime_ns}.png'
                    target = assets / name
                    if not target.exists():
                        overlay(source / 'inputs' / cid / 'global.png', value['regions'], target)
                    desc = str(len(value['regions'])) + ' 个候选'
                    row.append(f'{desc}<br>[![](local_assets/{name})](local_assets/{name})')
                    hrow.append(f'{html.escape(desc)}<a href="local_assets/{name}"><img src="local_assets/{name}"></a>')
                    if source == root: completed += 1
                elif failure.exists():
                    desc = '请求失败：' + str(read(failure).get('error', 'unknown'))
                    row.append(desc.replace('|', '/')); hrow.append(html.escape(desc))
                    if source == root: failed += 1
                else:
                    row.append('等待返回'); hrow.append('等待返回')
            md += ['### ' + vid, '', f'[本次请求](../cases/{cid}/local_discovery/{vid}/)', '',
                   '| v015 局部发现 | 对齐后局部发现 |', '| --- | --- |', '| ' + ' | '.join(row) + ' |', '']
            cells.append(f'<h3>{vid}</h3><div class="pair"><div>v015{hrow[0]}</div><div>对齐后{hrow[1]}</div></div>')
        sections.append(f'<section id="{cid}"><h2>{cid}</h2>' + ''.join(cells) + '</section>')
    total = sum(len(read(root / 'inputs' / cid / 'frames.json')['local_views']) for cid in ids)
    md[2:2] = [f'本次 LOCAL 已返回 {completed}/{total}，失败 {failed}。', '']
    (dest / '24图_局部发现对照.md').write_text('\n'.join(md) + '\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>局部发现规则对齐</title><style>body{font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%}section{margin:40px 0}a{color:#1767bb}</style>'
    page += f'<h1>LOCAL 返回 {completed}/{total}，失败 {failed}</h1><p>每20秒刷新；固定GLOBAL，当前展示审核之前的局部发现。{updated}</p>'
    page += '<p><a href="live.html">后续最终定位对照</a></p>'
    page += ' · '.join(f'<a href="#{cid}">{cid}</a>' for cid in ids)
    (dest / 'local_live.html').write_text(page + ''.join(sections))
    write(root / 'local_progress.json', {'updated_at': updated, 'returned': completed, 'failed': failed, 'total': total})
