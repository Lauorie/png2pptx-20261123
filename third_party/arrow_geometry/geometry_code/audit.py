"""Read-only multi-case operational audit. No proxy metric is a quality score."""
import argparse,json,statistics,time,collections
from pathlib import Path
import numpy as np
from transport.common import read_json as read,write_json as write


def audit(run):
    run=Path(run).resolve();out=run/'analysis';out.mkdir(exist_ok=True)
    manifest=read(run/'manifest.json');records=[];requests=[]
    for cid in manifest['ids']:
        folder=run/'cases'/cid;end=read(folder/'completed.json') if (folder/'completed.json').exists() else {}
        ss=[read(p) for p in folder.glob('groups/*/sam_summary.json')]
        groups=[read(p) for p in folder.glob('groups/*/completed.json')]
        repairs=[read(p) for p in folder.glob('groups/*/repair_*/answer.json')]
        calls=[]
        for p in folder.glob('**/call_*/status.json'):
            s=read(p);stage='coverage' if 'coverage' in p.parts else 'repair' if any(v.startswith('repair_') for v in p.parts) else 'identity'
            item=dict(case=cid,stage=stage,path=str(p),state=s['state'],seconds=s.get('seconds'),usage=s.get('usage') or {},reasoning_chars=s.get('reasoning_chars',0))
            calls.append(item);requests.append(item)
        accepted=[s for s in calls if s['state']=='transport_complete']
        token=sum(s['usage'].get('total_tokens',0) for s in calls)
        proposal=[read(p) for p in folder.glob('groups/*/proposal/draft.json')]
        scene=read(folder/'scene.json') if (folder/'scene.json').exists() else {'arrows':[]}
        counts=collections.Counter(r['coordinate_source'] for r in scene['arrows'])
        records.append(dict(case=cid,state=end.get('state','running'),groups_prepared=len(ss),groups_ended=len(groups),
            selected_masks=sum(s['selected_candidates'] for s in ss),raw_local_masks=sum(s['raw_candidates'] for s in ss),
            empty_local_groups=sum(not s['selected_candidates'] for s in ss),
            measured_paths=sum(len(s['paths']) for s in proposal),mask_mode_repairs=sum(s['mode']=='mask' for s in repairs),
            coordinate_mode_repairs=sum(s['mode']=='points' for s in repairs),
            targets=end.get('targets'),final_paths=len(scene['arrows']),coordinate_sources=dict(counts),
            fresh_calls=len(calls),accepted_calls=len(accepted),recorded_tokens=token,
            median_call_seconds=round(statistics.median(s['seconds'] for s in accepted),1) if accepted else None,
            reused_requests=len(list(folder.glob('**/reuse.json'))),
            failed_groups=len(list(folder.glob('groups/*/error.json'))),format_retries=len(list(folder.glob('**/validation_error_*.json')))))
    write(out/'metrics.json',dict(cases=records,requests=requests,time=time.time(),accuracy_measured=False))
    ended=[r for r in records if r['state']!='running'];accepted=[r for r in requests if r['state']=='transport_complete']
    lines=['# 本轮运行诊断','',f'已结束 {len(ended)}/{len(records)} 张。以下为运行及路径来源统计，不是人工准确率。',
        '新请求 token 包括成功和失败请求的已返回 usage；无 usage 的中断不计入，缓存复用不算新调用。','',
        '| case | 状态 | 已结束组 / 已准备组 | 空局部 mask 组 | 路径数 | GLM 点位修复次数 | 新调用 | 已记录 token |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| [{r["case"]}](../cases/{r["case"]}/RESULTS.md) | {r["state"]} | {r["groups_ended"]}/{r["groups_prepared"]} | {r["empty_local_groups"]} | {r["final_paths"]} | {r["coordinate_mode_repairs"]} | {r["fresh_calls"]} | {r["recorded_tokens"]:,} |')
    lines+=['','观察限制：同一个目标可能由多个绘图路径组成，路径数不等于箭头数。模型接受 mask 草稿不证明它正确；完整图上仍需检查漏检、重复、共享关系和几何偏差。','',
        '[全图原图/实例/样式/叠图](../RESULTS.md) · [逐请求数据](metrics.json)','']
    (out/'README.md').write_text('\n'.join(lines))
    print(dict(ended=len(ended),total=len(records),groups_ended=sum(r['groups_ended'] for r in records),requests=len(requests),
        accepted=len(accepted),recorded_tokens=sum(r['recorded_tokens'] for r in records),
        median_call_seconds=round(statistics.median(r['seconds'] for r in accepted),1) if accepted else None))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');audit(p.parse_args().run)
