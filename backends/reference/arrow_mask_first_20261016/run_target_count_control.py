"""Replay one saved geometry call; vary ONLY the requested target list."""
import argparse
import copy
import fcntl
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from PIL import Image
import protocol
import segmented_geometry as sg
import style_contract
from lib import viz
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model
from run_region_ablation import usage


def refresh_journal(run):
    spec=read(run/'selection.json');journal=Path(spec['journal']);parent=run.parent
    start='<!-- AUTO_RESULTS_BEGIN -->';end='<!-- AUTO_RESULTS_END -->'
    lines=[start,'','## 自动更新的执行记录','',
           '以下内容从保存的请求结果生成；仅报告执行、用量与点坐标，视觉结论仍需人工检查。',
           '', '| 实验 | 执行状态 | 墙钟秒 | 已报告 token |', '|---|---|---:|---:|']
    def row(name,folder):
        path=folder/'finished.json'
        if path.exists():
            d=read(path)
            lines.append(f"| {name} | {d['state']} | {d['seconds']:.1f} | {d.get('api',{}).get('reported_tokens','—')} |")
        else:lines.append(f'| {name} | 运行中／待返回 | — | — |')
    for s in read(parent/'selection.json'):row(s['name'],parent/'cases'/s['name'])
    for name in ('A_three_targets','B_single_target'):row(name,run/name)
    lines+=['','### 严格目标数对照：原始输出摘要','']
    for name in ('A_three_targets','B_single_target'):
        folder=run/name;answer=folder/'answer.json'
        lines.append(f"- **{name}**：[实际输入](target_count_control/{name}/request/prompt.json) · [原始回复](target_count_control/{name}/request/call_000/content.txt) · [叠图](target_count_control/{name}/result/overlap.png)。")
        if answer.exists():
            data=read(answer);nodes={n[0]:n[1:] for n in data['nodes']}
            for r in data['routes']:
                if r['id']!=spec['target_id']:continue
                for p in r['paths']:
                    points=[[s['kind'],[nodes[n] for n in s['nodes']]] for s in p['segments']]
                    lines.append('  - 黄色目标 '+r['id']+' / '+p['id']+'：`'+str(points)+'`。尚不自动判定视觉准确。')
    lines+=['',end]
    text=journal.read_text();prefix,tail=text.split(start,1);_,suffix=tail.split(end,1)
    tmp=journal.with_suffix('.md.tmp');tmp.write_text(prefix+'\n'.join(lines)+suffix);tmp.replace(journal)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(run/'code_manifest.json')
    if any(sha(run/'code'/n)!=h for n,h in manifest['files'].items()):raise ValueError('Frozen code changed')
    for n in ('config.json','selection.json'):
        if sha(run/n)!=manifest[n+'_sha256']:raise ValueError('Changed config/selection')
    spec=read(run/'selection.json');source=Path(spec['source_geometry'])
    original=read(source/'prompt.json');images=[(r['label'],Path(r['path'])) for r in read(source/'input/images.json')]
    metadata=read(source/'baseline_metadata.json');config=read(run/'config.json')
    model=Model(config,run_dir=run,system=protocol.SYSTEM);cpu=threading.Lock();started=time.monotonic()
    inputs={str(p.resolve()):sha(p) for _,p in images}
    inputs.update({str((source/n).resolve()):sha(source/n) for n in ('prompt.json','baseline_metadata.json')})
    write(run/'input_audit.json',dict(source_hashes=inputs,sole_prompt_variable='targets',
        same_images_including_all_three_target_boxes=True,same_target_id=spec['target_id'],
        fresh_calls_for_both_arms=True,previous_control_request=str(source/'request')))
    write(run/'process.json',dict(pid=os.getpid(),started=time.time()))
    def work(name,single):
        folder=run/name;folder.mkdir(exist_ok=True);begin=time.monotonic()
        if (folder/'finished.json').exists():return read(folder/'finished.json')
        q=copy.deepcopy(original)
        if single:q['targets']=[t for t in q['targets'] if t['id']==spec['target_id']]
        if not q['targets']:raise ValueError('Unknown target')
        assert {k:v for k,v in q.items() if k!='targets'}=={k:v for k,v in original.items() if k!='targets'}
        baseline={t['id']:copy.deepcopy(metadata[t['id']]) for t in q['targets']}
        for row in baseline.values():row['geometry_request']=str(folder/'request')
        write(folder/'prompt.json',q);write(folder/'input/images.json',[
            dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
        state={}
        try:
            fitted=model.ask(folder/'request',q,images,lambda a:sg.validate(a,baseline,q['size']))
            write(folder/'answer.json',fitted)
            with cpu:
                rows=sg.geometry(fitted,baseline);clean=Image.open(images[0][1]).convert('RGB')
                for row in rows:row['style']=style_contract.provisional(clean,None,row)
                write(folder/'rows_local.json',rows);viz.render(images[0][1],rows,folder/'result')
                sg.draw_points(fitted,images[0][1],folder/'points.png')
            state=dict(state='completed',paths=len(rows),requested_targets=len(q['targets']))
        except Exception as exc:state=dict(state='failed',error=str(exc),traceback=traceback.format_exc())
        state.update(seconds=time.monotonic()-begin,api=usage(folder),accuracy_measured=False)
        write(folder/'finished.json',state);print(name,state['state'],flush=True);return state
    refresh_journal(run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(work,'A_three_targets',False),pool.submit(work,'B_single_target',True)]
        while not all(f.done() for f in jobs):
            time.sleep(10);refresh_journal(run)
        results=[f.result() for f in jobs]
    write(run/'finished.json',dict(cases=results,seconds=time.monotonic()-started))
    # The earlier dense-network regression may still be completing; keep its
    # execution record current without changing any manual diagnosis text.
    while not (run.parent/'finished.json').exists():
        refresh_journal(run);time.sleep(10)
    refresh_journal(run)


if __name__=='__main__':main()
