"""Atomic report-only publication, full resolution linked images."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import shutil
import statistics
import threading
from transport.common import read_json as read,write_json as write,sha256
from visual_labels import render as render_labels
from evidence import frame

LOCK=threading.Lock()


def text_file(path, value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(value);tmp.replace(path)


def copy_image(source,dest):
    if not dest.exists() or source.stat().st_mtime_ns>dest.stat().st_mtime_ns:
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name('.'+dest.name+'.tmp');shutil.copy2(source,tmp);tmp.replace(dest)


def speed(root):
    stages=defaultdict(list);active=[];states=Counter();retries=0
    for p in (root/'cases').glob('*/*/*/request/**/call_*/status.json'):
        s=read(p);states[s.get('state','unknown')]+=1;stage='format_only' if 'format_only' in p.relative_to(root).parts else p.relative_to(root).parts[2]
        if s.get('state')=='requesting':active.append({'path':str(p.relative_to(root)),**{k:s.get(k) for k in ['seconds','reasoning_chars','content_chars']}})
        if s.get('state')=='transport_complete' and s.get('sse_done') and s.get('finish_reason')=='stop':
            stages[stage].append(s)
    for _ in (root/'cases').glob('*/*/*/request/validation_error_*.json'):retries+=1
    stats={k:{'complete_transport_attempts':len(v),'median_seconds':statistics.median(s['seconds'] for s in v),
        'completion_tokens':sum(s.get('usage',{}).get('completion_tokens',0) for s in v),
        'reasoning_tokens':sum(s.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) for s in v)} for k,v in stages.items()}
    reuse=len(list((root/'cases').glob('*/discovery/*/request/reuse.json')))
    format_fixed=len(list((root/'cases').glob('*/*/*/request/format_repair_resolution.json')))
    return {'updated_at':datetime.now(timezone.utc).isoformat(),'new_call_attempt_states':dict(states),'new_calls_by_stage':stats,
        'active_requests':active,'reused_discovery_requests':reuse,'format_retry_records':retries,'format_only_successes':format_fixed,
        'timing_note':'Transport attempts are not unique validated requests. Reused discovery latency/tokens excluded.'}


def publish(root):
    if read(root/'config.json').get('integrated_mode'):return
    with LOCK:
        cfg=read(root/'config.json');pub=Path(cfg['review_output_dir']);pub.mkdir(parents=True,exist_ok=True)
        for d in ['images','data','batches']: (pub/d).mkdir(exist_ok=True)
        progress=read(root/'progress.json');sp=speed(root);write(root/'analysis/speed_latest.json',sp)
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        order=read(root/'execution_order.json')['case_ids'];summary=[];phases=Counter()
        lines=['# 轻量箭头发现：图集两栏对照','',f'更新：{stamp}（北京时间）。整例结束 **{progress["ended"]}/{progress["total"]}**；状态 {progress["state"]}。',
            '',f'同时最多8张，结束即补位；API并发上限128；默认thinking、113k上限。已复用 {sp["reused_discovery_requests"]} 个完整发现请求，活跃新请求 {len(sp["active_requests"])}。',
            '', '[方案与取舍](方案.md) · [实时进度](实时进度.md) · [旧版与轻量版](旧版与轻量版对照.md)',
            '', '原图与轻量版结果直接链接无损PNG。中间候选明确标注。完成/框数不是准确率；几何去重可能误合并，仍需看图。','']
        compare=['# 上轮与本轮对照','',f'左侧 {Path(cfg["source_run"]).name}，右侧 {root.name}。旧版未完成时仅展示候选，不能作为最终质量对比。复用发现的case不用于声称冷启动总耗时改善。','']
        live=['# 实时进度','',f'更新：{stamp}（北京时间）；整例结束 {progress["ended"]}/{progress["total"]}；当前在跑 {len(progress.get("current_cases",[]))}/8 张。','',
            '| case | 阶段 | 发现视图 | 压缩后候选 | 校准返回/计划 | 两框审核返回/计划 |','|---|---|---:|---:|---:|---:|']
        for cid in order:
            d=root/'cases'/cid;inp=root/'inputs'/cid;status=read(d/'status.json') if (d/'status.json').exists() else {'state':'queued'}
            phase=status.get('phase',status['state']);phases[phase]+=1
            source=pub/'images'/(cid+'_source.png');copy_image(inp/'G/target.png',source)
            result=read(d/'result.json') if (d/'result.json').exists() else None
            views=len(list((d/'discovery').glob('*/observations.json')));plan=read(d/'repair_plan.json') if (d/'repair_plan.json').exists() else []
            returned=sum((d/'light_refine'/p['batch_id']/'result.json').exists() for p in plan)
            compression=read(d/'compression.json') if (d/'compression.json').exists() else {}
            compressed=compression.get('post_network_unit_count',compression.get('compressed_count','—'))
            residual=read(d/'residual_review/plan.json') if (d/'residual_review/plan.json').exists() else {'pairs':[]}
            reviewed=len(list((d/'residual_review').glob('P*/result.json')))
            if phase!='queued':live.append(f'| {cid} | {phase} | {views} | {compressed} | {returned}/{len(plan)} | {reviewed}/{len(residual["pairs"])} |')
            lines += ['## '+cid,'',f'状态：{status["state"]}；阶段：{phase}。','']
            if result:
                summary.append({k:result[k] for k in ['case_id','state','input_observation_count','compressed_count','output_identity_count','pending_count','repair_batches','timings','local_sam_tightened']})
                lines += [f'GLM原始观察 {result["input_observation_count"]} → 预归并 {result["compressed_count"]} → 最终身份 {result["output_identity_count"]}；校准 {result["repair_batches"]} 批；待确认 {result["pending_count"]}。','']
                available=d/'overlay.png';label='最终发现框（未做几何恢复）';write(pub/'data'/(cid+'.json'),result)
            else:
                available=next((d/n for n in ['pre_residual.png','repair_progress.png','provisional.png','compressed.png','discovery/G/overlay.png'] if (d/n).exists()),None)
                label='中间结果，尚未完成整例'
            right='等待返回'
            if available:
                fingerprint=sha256(d/'result.json' if result else available)[:12]
                dest=pub/'images'/(cid+'_result_'+fingerprint+'.png')
                if result:
                    if not dest.exists() or (d/'result.json').stat().st_mtime_ns>dest.stat().st_mtime_ns:
                        view=frame(read(inp/'views.json')['views'][0]['transform'])
                        render_labels(inp/'G/target.png',result['objects'],dest,view)
                else:copy_image(available,dest)
                right=f'[![](images/{dest.name})](images/{dest.name})<br>{label}'
            lines += ['| 原图 | 轻量版 |','|---|---|',f'| [![](images/{cid}_source.png)](images/{cid}_source.png) | {right} |','']
            if result:lines += [f'[来源、合并、排除与计时](data/{cid}.json)','']
            if available:
                old=Path(cfg['source_run'])/'cases'/cid
                oldpic=next((old/n for n in ['overlay.png','refined.png','observations.png'] if (old/n).exists()),None)
                if oldpic:
                    copy_image(oldpic,pub/'images'/(cid+'_previous.png'));oldlabel='上轮最终' if (old/'finished.json').exists() else '上轮中间候选'
                    compare += ['## '+cid,'','| '+oldlabel+' | 本轮 '+label+' |','|---|---|',f'| [![](images/{cid}_previous.png)](images/{cid}_previous.png) | {right} |','']
        live+=['','| 新调用阶段 | 完整返回尝试 | 中位耗时（秒） |','|---|---:|---:|']
        for stage,s in sp['new_calls_by_stage'].items():live.append(f'| {stage} | {s["complete_transport_attempts"]} | {s["median_seconds"]:.1f} |')
        live+=['',f'复用发现请求 {sp["reused_discovery_requests"]}；首答/重答校验问题记录 {sp["format_retry_records"]}；纯文本格式修复成功 {sp["format_only_successes"]}；活跃新请求 {len(sp["active_requests"])}。',
            '',f'历史失败传输尝试 {sp["new_call_attempt_states"].get("failed",0)} 次；这与最终case失败分别记录。',
            '', '耗时表按新请求的完整传输尝试统计；重试不等于额外最终对象，复用发现的历史耗时没有计入。',
            '', '[125图结果](轻量箭头发现_125例.md)','']
        text_file(pub/'轻量箭头发现_125例.md','\n'.join(lines)+'\n');text_file(pub/'旧版与轻量版对照.md','\n'.join(compare)+'\n');text_file(pub/'实时进度.md','\n'.join(live))
        write(root/'summary.json',{'cases':summary,'total':len(order),'finished':len(summary),'updated_at':sp['updated_at']})
        write(root/'monitor.json',{'updated_at':sp['updated_at'],'progress':progress,'phases':dict(phases),'active_new_requests':len(sp['active_requests']),'reused_discovery_requests':sp['reused_discovery_requests']})
        text_file(root/'RESULTS.md','\n'.join(live)+'\n')
