"""Expose global discovery separately so later local edits cannot obscure the ablation."""
import html
import json
from pathlib import Path
from datetime import datetime, timezone


def publish_global(root, overlay):
    cfg=json.loads((root/'config.json').read_text())
    baseline=Path(cfg['comparison_run'])
    baseline_title=cfg.get('comparison_title', baseline.name)
    ids=json.loads((root/'cases.json').read_text())
    dest=root/'reports';assets=dest/'assets';assets.mkdir(parents=True,exist_ok=True)
    md=['# 24图：全局发现阶段对照','',f'左：{baseline_title} 已保存全局发现；右：本轮新全局发现。输入图和推理参数保持一致。',
        '这是首轮候选，不是局部修正后的最终结果；框数不代表质量。',
        '','[自动刷新](global_live.html) · [完整定位进度](24图_实时对照.md) · [本轮批准草案](../PROMPTS.md)','']
    sections=[];done=0
    for cid in ids:
        cells=[];imgs=[]
        for label,run in [('old',baseline),('new',root)]:
            p=run/'cases'/cid/'global_discovery/result.json'
            if p.exists():
                data=json.loads(p.read_text());target=assets/(cid+'_global_'+label+'_'+str(p.stat().st_mtime_ns)+'.png')
                if not target.exists():overlay(run/'inputs'/cid/'global.png',data['regions'],target)
                cells.append(f'[![](assets/{target.name})](assets/{target.name})')
                imgs.append(f'<div><a href="assets/{target.name}"><img src="assets/{target.name}"></a></div>')
                if label=='new':done+=1
            else:
                failed=(run/'cases'/cid/'global_discovery/unresolved.json').exists()
                text='请求失败，查看trace' if failed else '等待'
                cells.append(text);imgs.append('<div>'+text+'</div>')
        md += ['## '+cid,'',f'[新请求trace](../cases/{cid}/global_discovery/)','',
               f'| {baseline_title} 全局发现 | 本轮全局发现 |','|---|---|','| '+' | '.join(cells)+' |','']
        sections.append('<section id="'+cid+'"><h2>'+cid+'</h2><div class="pair">'+''.join(imgs)+'</div></section>')
    (dest/'24图_全局发现对照.md').write_text('\n'.join(md)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>全局发现对照</title><style>body{font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}section{margin:30px 0}</style>'
    page+=f'<h1>全局发现已返回 {done}/{len(ids)}</h1><p>左为{html.escape(baseline_title)}，右为新提示词；每20秒刷新。'+html.escape(datetime.now(timezone.utc).isoformat())+'</p>'
    page+='<a href="live.html">完整定位进度</a> · <a href="../PROMPTS.md">新提示词</a>'
    (dest/'global_live.html').write_text(page+''.join(sections))
