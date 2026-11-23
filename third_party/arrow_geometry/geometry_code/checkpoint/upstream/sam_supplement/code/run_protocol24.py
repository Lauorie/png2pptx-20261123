"""One run, shared API semaphore, 24 cases through supplement then review."""
import argparse
import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import experiment as E
import review_additions as R
from transport.model import Model

LOCK=threading.RLock()
TERMINAL={"completed","completed_with_issues","failed"}

def publish(root):
    with LOCK:
        cfg=E.read(root/"config.json")
        cases=E.read(root/"cases.json")
        old=Path(cfg["comparison_run"])
        records=[];sections=[]
        for cid in cases:
            m=root/"missing/cases"/cid;r=root/"review/cases"/cid
            ms=E.read(m/"status.json",{"state":"queued","stage":"queued"})
            rs=E.read(r/"status.json",{"state":"waiting_source","stage":"waiting_source"})
            result=E.read(m/"result.json",{})
            protocol_pending=sum(x.get("count",0) for x in result.get("issues",[]) if x.get("stage")=="missing_protocol")
            records.append(dict(case=cid,missing=ms,review=rs,protocol_pending=protocol_pending))
            raw=m/"current.png"
            image=r/"current.png"
            right=(f"[![当前](../review/cases/{cid}/current.png)](../review/cases/{cid}/current.png)"
                   if image.exists() else
                   f"[![补漏候选，尚未审核](../missing/cases/{cid}/current.png)](../missing/cases/{cid}/current.png)")
            sections += [f"## {cid}","",
                f"补漏 {ms['state']}，窗口 {ms.get('views_finished',0)}/6，候选 {ms.get('additions',0)}；"
                f"整理 {rs['state']}，保留 {rs.get('reviewed_additions',0)}，待审 {rs.get('pending',0)}；"
                f"协议待处理条目 {protocol_pending}。","",
                f"[原图](../inputs/{cid}/source.png) · [本轮补漏候选](../missing/cases/{cid}/current.png) · "
                f"[补漏trace](../missing/cases/{cid}/proposal_views/) · [整理trace](../review/cases/{cid}/review/) · "
                f"[整理结果](../review/cases/{cid}/result.json)","",
                "| 上轮 v030 整理结果 | 本轮 v031：整理未返回时暂显示补漏候选 |","|---|---|",
                f"| [![上轮]({old}/cases/{cid}/current.png)]({old}/cases/{cid}/current.png) | {right} |",""]
        mf=sum(r["missing"]["state"] in TERMINAL for r in records)
        rf=sum(r["review"]["state"] in TERMINAL for r in records)
        E.write(root/"progress.json",dict(updated_at=E.now(),total=len(cases),missing_finished=mf,
            review_finished=rf,finished=rf,cases=records,case_concurrency=cfg["case_workers"],
            api_concurrency=cfg["api_concurrency"]))
        text=["# 24图：补漏协议修复与新增框整理","",
              f"更新 {E.now()}；补漏结束 {mf}/{len(cases)}；整理结束 {rf}/{len(cases)}。","",
              ("这是v031同一轮请求的接收修复回放；补漏不再调用模型，整理优先精确复用同轮回复。原始失败trace完整保留。" if cfg.get('missing_cache_only') else
               "原定位627框及SAM证据复用；补漏和新增框整理全部重新调用。只做定位，不做几何/样式。"),
              f"默认思考；本进程图任务上限{cfg['case_workers']}、API上限{cfg['api_concurrency']}。补漏候选尚未确认完整；整理后才成为已审核新增。",
              "越界bbox按当前输入图0..W-1、0..H-1截断，原始坐标与调整记录保留；其他协议异常候选/操作单独待处理。完成数量和框数不代表视觉准确率。","",
              "[说明](../README.md) · [实际提示词](../PROMPT.md) · [离线回放](../preflight/离线回放.md) · "
              "[补漏进度](../missing/reports/24图_实时对照.md) · [整理前后](../review/reports/24图_整理前后对照.md)",""]+sections
        dest=root/"reports";dest.mkdir(exist_ok=True)
        temp=dest/"24图_实时对照.md.tmp";temp.write_text("\n".join(text)+"\n")
        temp.replace(dest/"24图_实时对照.md")
        (root/"RESULTS.md").write_text(f"# 补漏协议修复24图\n\n更新 {E.now()}，补漏 {mf}/24，整理 {rf}/24。\n\n"
            "[24图对照](reports/24图_实时对照.md) · [离线回放](preflight/离线回放.md)\n\n"
            "本轮几何未执行；视觉质量待逐图查看。历史运行和前面的发现流程未改。\n")

def run(root):
    lock=(root/"run.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,digest in E.read(root/"manifest.json")["files"].items():
        if E.sha256(root/name)!=digest:raise ValueError("Frozen file changed: "+name)
    cfg=E.read(root/"config.json")
    model=Model(cfg,run_dir=root,reuse_from=Path(cfg['reuse_from']) if cfg.get('reuse_from') else None)
    missing_cfg=E.read(root/"missing/config.json")
    review_cfg=E.read(root/"review/config.json")
    sem=threading.Semaphore(1)
    E.write(root/"process.json",dict(pid=os.getpid(),started_at=E.now()))
    E.write(root/"execution.json",dict(state="running",started_at=E.now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    worker=threading.Thread(target=watch,daemon=True);worker.start()
    def one(cid):
        E.run_case(root/"missing",cid,missing_cfg,model,sem)
        status=E.read(root/"missing/cases"/cid/"status.json")
        if status["state"]=="failed":
            E.write(root/"review/cases"/cid/"status.json",
                    dict(state="failed",stage="upstream_failed",updated_at=E.now(),
                         error="Missing stage failed; not treating absent additions as an empty success"))
        else:R.run_case(root/"review",cid,review_cfg,model)
        publish(root)
    try:
        with ThreadPoolExecutor(max_workers=cfg["case_workers"]) as pool:
            if cfg.get('source_live_run'):
                waiting=list(E.read(root/'cases.json'));active={}
                while waiting or active:
                    for cid in list(waiting):
                        if len(active)>=cfg['case_workers']:break
                        status=E.read(Path(cfg['source_live_run'])/'review/cases'/cid/'status.json',{})
                        if status.get('state') in TERMINAL:
                            active[pool.submit(one,cid)]=cid;waiting.remove(cid)
                    for job in list(active):
                        if job.done():job.result();active.pop(job)
                    if waiting or active:time.sleep(5)
            else:
                jobs=[pool.submit(one,cid) for cid in E.read(root/"cases.json")]
                for job in as_completed(jobs):job.result()
        E.write(root/"execution.json",dict(state="finished",finished_at=E.now()))
    except BaseException as exc:
        E.write(root/"execution.json",dict(state="failed",finished_at=E.now(),error=str(exc)))
        raise
    finally:
        stop.set();worker.join();publish(root)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["run","report"])
    ap.add_argument("--run-dir",type=Path,required=True);args=ap.parse_args()
    (run if args.action=="run" else publish)(args.run_dir.resolve())

