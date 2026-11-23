"""Replay archived protocol failures without making API calls."""
import argparse
import json
from collections import Counter
from pathlib import Path
from supplement_protocol import MissingProtocol, ReviewProtocol
from experiment import write

def replay(audit_path, output):
    audit=json.loads(audit_path.read_text())
    events=[];accepted=[];failures=[]
    def contract(entry):
        request=Path(entry["request"])
        prompt=json.loads((request/"prompt.json").read_text())["prompt"]
        size=prompt["input_size"]
        return (MissingProtocol(size) if entry["stage"]=="missing" else
                ReviewProtocol(size,prompt["editable_additions"]))
    for entry in audit["requests"]:
        v=contract(entry);q=Path(entry["request"])
        complete=[c for c in entry["calls"] if c["status"].get("state")=="transport_complete"]
        for error in entry["validation_errors"]:
            idx=int(Path(error["file"]).stem.rsplit("_",1)[1])
            source=q/complete[idx]["name"]/"content.txt"
            try:
                raw=v.parse(source.read_text())
                out=v(raw)
                bad=out["_protocol"]["rejected"]
                if entry["stage"]=="missing":
                    for item,row in zip(out["_protocol"]["accepted"],out["regions"]):
                        original=raw["regions"][item["source_index"]]
                        for key in ("bbox","render_type","grouping","label"):
                            assert original[key]==row[key],(entry["case"],key)
                events.append(dict(case=entry["case"],stage=entry["stage"],source=str(source),
                    old_error=error["error"],outcome="partial_pending" if bad else "accepted",
                    accepted_count=len(out.get("regions",out["_protocol"]["accepted"])),
                    rejected_count=len(bad),pending_ids=out["_protocol"].get("pending_focus_ids",[]),
                    coordinates_changed=False,normalized=out))
            except Exception as exc:
                failures.append(dict(source=str(source),error=repr(exc)))
        if entry["accepted"]:
            raw=json.loads((q/"result.json").read_text())
            out=v(raw)
            assert not out["_protocol"]["rejected"],str(q)
            if entry["stage"]=="missing":
                assert [r["bbox"] for r in out["regions"]]==[r["bbox"] for r in raw["regions"]],str(q)
            else:
                for key in ("keep_ids","replacements","discard"):
                    if key in raw:assert out[key]==raw[key],str(q)
            accepted.append(str(q))
    result=dict(api_calls=0,archived_error_events=len(events),
        outcomes=dict(Counter(x["outcome"] for x in events)),
        original_accepted_requests_regressed=0,
        original_accepted_requests_checked=len(accepted),
        coordinates_changed=False,failures=failures,events=events)
    write(output/"replay.json",result)
    lines=["# 补漏协议离线回放","",
        f"历史报错 {len(events)} 次；结果 {result['outcomes']}；失败 {len(failures)}。",
        f"另回归旧版已接收请求 {len(accepted)} 个，候选坐标和整理操作保持。API调用0。",
        "","partial_pending表示异常候选/整组操作单独保留，未改坐标，其他候选继续。不是视觉准确率。",
        "","|阶段|case|原错误|新结果|待处理数|","|---|---|---|---|---|"]
    lines += [f"|{x['stage']}|{x['case']}|{x['old_error']}|{x['outcome']}|{x['rejected_count']}|" for x in events]
    (output/"离线回放.md").write_text("\n".join(lines)+"\n")
    if failures:raise AssertionError(failures)
    print(json.dumps({k:v for k,v in result.items() if k!="events"},ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--audit",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);replay(args.audit,args.output)

