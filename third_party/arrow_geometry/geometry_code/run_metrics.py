"""Read-only request accounting and same-case performance comparisons."""
import collections
import json
import time
from pathlib import Path
from transport.common import write_json as write


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def collect(root,cid):
    case=root/'cases'/cid;now=time.time();start=read(case/'case_started.json').get('started')
    exact=start is not None
    if start is None:start=read(root/'process.json').get('started',now)
    done=case/'completed.json';initial=case/'initial_results.json';counts=collections.Counter();tokens=0;active=[]
    stages=collections.Counter();logical=set();valid=0;validation=0
    for p in case.glob('**/active_request.json'):
        path=read(p).get('path')
        if path:logical.add(path)
    for path in logical:
        d=Path(path);valid+=int((d/'result.json').exists());validation+=len(list(d.glob('validation_error_*.json')))
        parts=d.parts
        stage='geometry_critic' if 'geometry_critic' in parts else 'mask_selection' if 'target_mask_geometry' in parts else 'style' if 'styles' in parts else 'identity' if 'identity' in parts else 'geometry_or_review'
        stages[stage]+=1
    for p in case.glob('**/call_*/status.json'):
        s=read(p);state=s.get('state','unknown');counts[state]+=1
        if state=='transport_complete':tokens+=(s.get('usage') or {}).get('total_tokens',0)
        if state=='requesting':active.append(dict(path=str(p.relative_to(case)),seconds=s.get('seconds',0),last_update_age=round(now-p.stat().st_mtime,1)))
    return dict(case=cid,state=read(done).get('state','running'),
        elapsed_seconds=max(0,(done.stat().st_mtime if done.exists() else now)-start),
        initial_seconds=max(0,initial.stat().st_mtime-start) if initial.exists() else None,
        timing_source='case_start' if exact else 'run_start_approximation',logical_requests=len(logical),valid_requests=valid,
        request_stages=dict(stages),attempts=dict(counts),validation_rejections=validation,
        transport_complete_reported_tokens=tokens,active=active,
        initial_groups_complete=len(list(case.glob('groups/*/completed.json'))),
        initial_groups_total=len(read(case/'observation_plan.json').get('groups',[])))


def update(run,spec):
    rows=[];baseline=Path(spec['performance_baseline'])
    for cid in spec['cases']:rows.append(dict(case=cid,baseline=collect(baseline,cid),current=collect(run,cid)))
    write(run/'performance.json',dict(updated=time.time(),rows=rows,
        token_scope='Transport-complete reported usage, including validation retries; excludes unreported active/failed usage.',
        accuracy_measured=False))
    lines=['# 四图速度与请求量对照','',
        '运行中每30秒刷新文件。优先按逐图起点统计；缺少逐图起点的旧运行以整轮起点近似。'
        '请求数量含补漏与样式，尚未结束的一方不是完整总量。token仅计完整传输且已报告usage的尝试，含校验重试。','',
        '| 图 | 轮次 | 初始组 | 初始阶段分钟 | 全流程分钟 | 逻辑请求 | 几何review | mask选择 | 校验拒绝 | 已报告token | 状态 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for item in rows:
        for key,label in [('baseline',spec.get('performance_baseline_label','共享覆盖基线')),('current',spec.get('comparison_label','GLM点几何'))]:
            r=item[key];initial='—' if r['initial_seconds'] is None else f"{r['initial_seconds']/60:.1f}"
            lines.append(f"| {item['case']} | {label} | {r['initial_groups_complete']}/{r['initial_groups_total']} | {initial} | {r['elapsed_seconds']/60:.1f} | {r['logical_requests']} | {r['request_stages'].get('geometry_critic',0)} | {r['request_stages'].get('mask_selection',0)} | {r['validation_rejections']} | {r['transport_complete_reported_tokens']:,} | {r['state']} |")
    lines+=['','## 本轮正在请求','', '| 图 | 请求 | 持续秒 | 状态最近更新距今秒 |','|---|---|---:|---:|']
    for item in rows:
        for a in item['current']['active']:
            lines.append(f"| {item['case']} | {a['path']} | {a['seconds']:.1f} | {a['last_update_age']:.1f} |")
    lines+=['','[图像对照](COMPARISON.md) · [当前结果](RESULTS.md) · [详细统计](performance.json)','',
        '完成、耗时或请求减少都不能替代几何、漏线、重复和样式的视觉检查。']
    tmp=run/'PERFORMANCE.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(run/'PERFORMANCE.md')
