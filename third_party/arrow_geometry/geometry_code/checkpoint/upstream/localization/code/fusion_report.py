"""Atomic, relative-link live reports for the paired fusion experiment."""
import html
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from transport.common import read_json as read, write_json as write


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(temp, path)


def report(root):
    root = Path(root); dest = root/'reports'; assets = dest/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    ids = read(root/'cases.json'); stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    report_name=f'{len(ids)}图_实时对照.md'
    md = [f'# 区域融合与几何：{len(ids)} 图实时对照', '', '更新：'+stamp,
          '', '[自动刷新网页](live.html) · [实验说明](../README.md)', '',
          '左：v002 定位；右：模型重新安排恢复区域。两边使用同一套几何/critic，未变区域共享一次结果。',
          '几何图只有已返回任务；未结束时不当作最终结果。头尺寸 med/med、默认线宽；没有补漏、去重、精细样式。', '']
    blocks=[]; progress=[]
    def asset(path, label):
        if not path.exists(): return '等待', '等待'
        name = str(path.relative_to(root)).replace('/', '__')+'_'+str(path.stat().st_mtime_ns)+'.png'
        dst = assets/name
        if not dst.exists(): shutil.copy2(path, dst)
        link='assets/'+name
        return f'[![{label}]({link})]({link})', f'<a href="{link}"><img loading="lazy" src="{link}"></a>'
    for cid in ids:
        c=root/'cases'/cid
        status=read(c/'status.json') if (c/'status.json').exists() else {'stage':'queued'}
        geometry=read(c/'geometry_progress.json') if (c/'geometry_progress.json').exists() else {}
        result=read(c/'result.json') if (c/'result.json').exists() else {}
        label=status.get('stage','queued')
        if geometry: label+=f"；几何 {geometry.get('ended',0)}/{geometry.get('total',0)}；失败 {geometry.get('failed',0)}"
        progress.append({'case':cid, 'localization':status, 'geometry':geometry,
                         'before_regions':result.get('before_count'), 'after_regions':result.get('region_count')})
        md+=['## '+cid,'',label,'',f'[原图](../inputs/{cid}/source.png) · [融合 Trace](../cases/{cid}/fusion/) · [几何 Trace](../cases/{cid}/geometry_tasks/)', '']
        block=f'<section id="{cid}"><h2>{cid}</h2><p>{html.escape(label)}</p><a href="../inputs/{cid}/source.png">原图</a>'
        for name, paths in [('定位', [c/'before_boxes.png',c/'after_boxes.png']),
                            ('几何叠图', [c/'before/result/overlap.png',c/'after/result/overlap.png']),
                            ('几何白底', [c/'before/result/styled.png',c/'after/result/styled.png'])]:
            pair=[asset(p,name) for p in paths]
            md += [f'| 融合前 · {name} | 融合后 · {name} |','|---|---|',f'| {pair[0][0]} | {pair[1][0]} |','']
            block+=f'<h3>{name}</h3><div class="pair"><div>融合前{pair[0][1]}</div><div>融合后{pair[1][1]}</div></div>'
        blocks.append(block+'</section>')
    atomic(dest/report_name,'\n'.join(md)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>区域融合与几何对照</title><style>body{font:16px system-ui;margin:24px;background:#f5f7fb}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{display:block;width:100%;background:white}section{margin:30px 0;background:white;padding:16px}a{color:#1767bb}</style>'
    atomic(dest/'live.html',page+'<h1>区域融合与几何对照</h1><p>15 秒自动刷新；'+stamp+'</p>'+' · '.join(f'<a href="#{c}">{c}</a>' for c in ids)+''.join(blocks))
    write(root/'progress.json',{'updated_at':stamp,'cases':progress})
    atomic(root/'RESULTS.md',f'# {root.name} 区域融合与几何对照\n\n[两栏实时结果](reports/{report_name}) · [自动刷新](reports/live.html) · [配置与实验变量](README.md) · [进度](progress.json)\n\n执行完成不代表视觉正确；结论待逐例核查。\n')
