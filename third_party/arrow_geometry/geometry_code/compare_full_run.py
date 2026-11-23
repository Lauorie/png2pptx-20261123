"""Read-only comparison of complete cases; never reconstructs or combines rows."""
import argparse
import os
import time
from pathlib import Path
from transport.common import read_json as read, write_json as write, sha256 as sha


def link(path, root):
    return os.path.relpath(path, root)


def update(run, baseline):
    scope=read(run/'scope.json')['cases']
    progress=read(run/'progress.json') if (run/'progress.json').exists() else {'cases':{},'completed':0}
    lines=['# 完整 case 对照','',
        '每张图重新运行身份、几何、样式和补漏。只有完整 case 结束后才加入新结果；运行中图不作为质量结论。',
        '各 case 链接内按原图、旧结果、新结果、实例、叠图纵向排列，可查看原尺寸。','',
        '| case | 状态 | 完整对照 |','|---|---|---|']
    evidence=[]
    for cid in scope:
        old=baseline/'cases'/cid;new=run/'cases'/cid
        state=progress.get('cases',{}).get(cid,{}).get('stage','等待')
        if not (new/'completed.json').exists():
            lines.append(f'| {cid} | {state} | 等待完整结果 |');continue
        identical=sha(old/'clean.png')==sha(new/'clean.png')
        if not identical:
            lines.append(f'| {cid} | 原图来源不一致，未生成对照 | |');continue
        panels=[('原图',new/'clean.png'),('旧版：实际重建',old/'result/styled.png'),
                ('本轮：实际重建',new/'result/styled.png'),('本轮：彩色实例',new/'result/instances.png'),
                ('本轮：原图叠加',new/'result/overlap.png')]
        content=[f'# {cid}：完整结果对照','',f'状态：{state}。尚需视觉审核；程序完成不代表图形正确。','',
                 '[逐组结果与请求](RESULTS.md) · [最终几何与来源](scene.json)','']
        for label,path in panels:content += [f'**{label}**','',f'![{label}]({link(path,new)})','']
        (new/'COMPARE.md').write_text('\n'.join(content))
        lines.append(f'| {cid} | {state} | [原尺寸对照](cases/{cid}/COMPARE.md) |')
        evidence.append(dict(case=cid,source_identical=identical,images=[dict(label=l,path=str(p),sha256=sha(p)) for l,p in panels]))
    tmp=run/'COMPARISON.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(run/'COMPARISON.md')
    write(run/'comparison_sources.json',evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args();run=a.run.resolve();baseline=a.baseline.resolve()
    while True:
        update(run,baseline)
        if not a.watch or (run/'finished.json').exists():break
        if (run/'process.json').exists() and not Path('/proc',str(read(run/'process.json')['pid'])).exists():break
        time.sleep(20)
