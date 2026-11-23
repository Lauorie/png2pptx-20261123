"""Four-case scheduler/export smoke test with fake model, zero API calls."""
import copy
import json
import shutil
import threading
from pathlib import Path
from PIL import Image
import experiment as E
import review_additions as R

def run(frozen, output):
    if output.exists():raise FileExistsError(output)
    cfg=E.read(frozen/"config.json")
    ids=E.read(frozen/"cases.json")[:4]
    calls=[]
    class Fake:
        def ask(self,folder,prompt,images,validator,system=None):
            calls.append(str(folder))
            if "editable_additions" in prompt:
                a=dict(decision="edit",keep_ids=[r["id"] for r in prompt["editable_additions"]],
                       replacements=[],discard=[])
            else:
                rows=[]
                for r in prompt["sam_candidates"]:
                    try:E.C.number_box(r["bbox"],prompt["input_size"])
                    except ValueError:continue
                    rows.append(dict(bbox=r["bbox"],render_type="linework",grouping="single",label="offline test candidate"))
                    break
                a={"regions":rows}
            return validator(a)
    for phase in ("missing","review"):
        d=output/phase;d.mkdir(parents=True)
        (d/"inputs").symlink_to(frozen/"inputs",target_is_directory=True)
        E.write(d/"cases.json",ids)
        cc=dict(cfg,source_run=str(output/"missing"))
        E.write(d/"config.json",cc)
    model=Fake()
    for cid in ids:
        d=output/"missing/cases"/cid;d.mkdir(parents=True)
        (output/"review/cases"/cid).mkdir(parents=True)
        for name in ("retained_candidates.json","retained_sam_union.png","compression_audit.json","filter_audit.json"):
            shutil.copy2(frozen/"missing/cases"/cid/name,d/name)
        E.run_case(output/"missing",cid,cfg,model,threading.Semaphore(1))
        assert E.read(d/"status.json")["state"]=="completed",E.read(d/"status.json")
        R.run_case(output/"review",cid,dict(cfg,source_run=str(output/"missing")),model)
        result=E.read(output/"review/cases"/cid/"result.json")
        assert result["baseline_preserved"]
        base=E.read(frozen/"inputs"/cid/"baseline_native.json")
        assert result["native_regions"][:len(base)]==base
        for r in E.read(d/"additions.json"):
            assert r["extent"]=="uncertain" and not r["review_confirmed"]
        for r in E.read(output/"review/cases"/cid/"reviewed_additions.json"):
            assert r["extent"]=="complete" and r["review_confirmed"]
        tasks=E.read(d/"reconstruction_handoff.json")["tasks"][len(base):]
        assert all(not t["render_owner"] and t["requires_review"] for t in tasks)
    summary=dict(cases=ids,fake_calls=len(calls),api_calls=0,
                 baseline_preserved=True,unreviewed_not_exported_as_owner=True)
    E.write(output/"checks.json",summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)

