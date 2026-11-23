"""Prepare one frozen 24-case run; never copy prior GLM supplemental results."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import experiment as E
import review_additions as R
import run_protocol24 as DRIVER
from supplement_protocol import VERSION

def prepare(project, run):
    if run.exists():
        raise FileExistsError("Do not overwrite an existing run: "+str(run))
    src=project/"runs/v029_sam_additive_missing24"
    old_review=project/"runs/v030_sam_additions_review24"
    ids=E.read(old_review/"cases.json")
    cfg=E.read(old_review/"config.json")
    for key in ("source_run","execution_case_ids","linked_case_ids","linked_run"):
        cfg.pop(key,None)
    cfg.update(protocol_version=VERSION,bbox_policy="clip_to_input_pixel_centers",case_workers=16,api_concurrency=128,
        reuse_frozen_sam=True,new_sam_calls=False,new_missing_glm_calls=True,
        new_addition_review_calls=True,new_global_discovery_calls=False,
        max_context_expansions=1,comparison_run=str(old_review),source_evidence_run=str(src),
        supplemental_visual_format_retries=0)
    cfg.pop("reasoning_effort",None)
    run.mkdir(parents=True)
    for sub in ("inputs","missing","review","reports","preflight","constraints_snapshot"):
        (run/sub).mkdir()
    shutil.copytree(Path(__file__).parent,run/"code",
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    preflight=Path(__file__).parent.parent/"preflight"
    if preflight.exists():
        for f in preflight.iterdir():
            if f.is_file():shutil.copy2(f,run/'preflight'/f.name)
    sources=[];baseline_count=0;checks=[]
    for cid in ids:
        inp=src/"inputs"/cid;dest=run/"inputs"/cid
        dest.mkdir()
        for name in ("source.png","baseline_native.json"):
            original=inp/name;target=dest/name
            shutil.copy2(original,target)
            sources.append(dict(case=cid,source=str(original.resolve()),copy=str(target.relative_to(run)),sha256=E.sha256(target)))
        baseline=E.read(dest/"baseline_native.json");baseline_count+=len(baseline)
        for phase in ("missing","review"):
            d=run/phase/"cases"/cid;d.mkdir(parents=True)
            image=E.Image.open(dest/"source.png").convert("RGB")
            E.plot(image,baseline,d/"baseline.png");E.plot(image,baseline,d/"current.png")
            E.write(d/"status.json",dict(state="queued" if phase=="missing" else "waiting_source",
                                        stage="queued" if phase=="missing" else "waiting_source"))
        source_case=src/"cases"/cid
        d=run/"missing/cases"/cid
        names=("retained_candidates.json","retained_sam_union.png","compression_audit.json",
               "filter_audit.json","excluded_candidates.json","compressed_candidates.json")
        for name in names:
            shutil.copy2(source_case/name,d/name)
            sources.append(dict(case=cid,source=str((source_case/name).resolve()),
                copy=str((d/name).relative_to(run)),sha256=E.sha256(d/name)))
        # Actual input drawings/crops must match v029, independent of new prompt.
        image=E.Image.open(dest/"source.png").convert("RGB")
        mask=E.Image.open(d/"retained_sam_union.png").convert("L")
        groups=E.read(d/"retained_candidates.json")
        for tid,view in E.tile_views(image.size,cfg):
            for name,im in [("clean.png",view.image(image)),
                            ("evidence.png",E.annotated(image,view,baseline,groups,mask))]:
                import io
                b=io.BytesIO();im.save(b,format="PNG")
                old=source_case/"proposal_views"/tid/"view_00"/name
                equal=hashlib.sha256(b.getvalue()).hexdigest()==E.sha256(old)
                assert equal,(cid,tid,name)
                checks.append(dict(case=cid,tile=tid,file=name,identical=True))
    assert baseline_count==627,baseline_count
    for phase in ("missing","review"):
        stage=run/phase
        (stage/"inputs").symlink_to("../inputs",target_is_directory=True)
        E.write(stage/"cases.json",ids)
        scfg=dict(cfg)
        if phase=="review":scfg["source_run"]=str(run/"missing")
        E.write(stage/"config.json",scfg)
        (stage/"README.md").write_text("[统一方案](../README.md)\n")
        (stage/"preflight").symlink_to("../preflight",target_is_directory=True)
    E.write(run/"cases.json",ids);E.write(run/"config.json",cfg)
    E.write(run/"preflight/input_checks.json",dict(baseline_count=baseline_count,old_boxes_unchanged=True,
        all_input_pngs_identical=True,input_pngs_checked=len(checks),details=checks))
    E.write(run/"provenance.json",dict(created_at=E.now(),source_code=str(old_review/"code"),
        source_evidence=str(src),no_prior_supplemental_glm_results_reused=True,sources=sources,
        baseline_preserved=True,front_discovery_unchanged=True))
    E.write(run/"execution.json",dict(state="prepared",prepared_at=E.now()))
    E.write(run/"preflight/checks.json",dict(protocol_tests=13,archived_error_events=40,
        offline_replay_outcomes={"accepted":36,"partial_pending":4},old_accepted_requests_checked=182,
        baseline_boxes=627,input_pngs_checked=len(checks),model_calls=0))
    scope=project.parents[3]
    for f in (scope/"communication").glob("*.md"):
        shutil.copy2(f,run/"constraints_snapshot"/f.name)
    for f in (scope/"AGENTS.md",project.parents[1]/"AGENTS.md"):
        shutil.copy2(f,run/"constraints_snapshot"/(f.parent.name+"_AGENTS.md"))
    sample=E.proposal_prompt((1080,720),[],[])
    review=R.review_prompt((1080,720),[],[])
    (run/"PROMPT.md").write_text("# 实际提示词模板\n\n## 补漏\n\n"+json.dumps(sample,ensure_ascii=False,indent=2)+
        "\n\n## 新增框整理\n\n"+json.dumps(review,ensure_ascii=False,indent=2)+"\n")
    (run/"review/PROMPT.md").write_text("[统一提示词](../PROMPT.md)\n")
    (run/"README.md").write_text("""# v031：补漏协议精简与局部容错，24图重跑

本轮只修改SAM补漏及新增框整理。原始定位627框、SAM过滤候选及辅助图全部复用v029；不重新运行前面的全局/局部发现，不调用SAM。补漏与新增框整理全部新请求，不复用v029/v030模型回复。

修改：补漏只要求regions内bbox/type/grouping/label；后台记录LOCAL及未审核状态，去掉complete门槛。解析器按阶段提取唯一JSON，固定元数据可缺省，无害描述保留在audit。候选逐条校验，整理按source_ids关联组原子校验；坏候选/坏操作单独保留，其他输出继续。格式错误最多一次纯文本修复，不重新识图；真正需要上下文时仍由新增整理扩图一次。

坐标仍是当前CLEAN绝对像素中心0..W-1、0..H-1。按用户2026-10-29要求，补漏与新增整理的越界bbox截断到该范围；原始坐标与有效坐标分开保留，调整写入protocol_audit。倒置、非数值和截断后无有效面积的框单独待处理。显式错误坐标系、未知编辑操作不能靠默认值掩盖。新增候选全部extent=uncertain、render_owner=false；keep或合规修框经review后才获得完整性确认。复用旧框字段与顺序原样保留。

并发：16图，共享一个API信号量128。每图补漏6视野/整理4请求；同一图补漏结束即进入整理，无需等24图全部补漏结束。默认thinking enabled，不传reasoning_effort，max_tokens113000；1200秒总时限和网络重试保持。

历史40次协议错误离线回放：36直接适配、4局部保留待处理，0坐标修改、0API调用。旧版182个已接收请求回归无候选坐标或整理操作变化；13项针对原框保护、分组原子性、歧义JSON、非视觉重试的测试通过。新提示词效果仍需真实24图验证。

根目录reports每15秒更新，原图叠加预览随候选/整理返回更新。输出在missing/与review/。最终可审核的交接在review/cases/*/reconstruction_handoff.json。未运行几何/PPT/样式，执行完成不代表视觉准确率。旧运行、主线、前面的发现协议均未修改。

[24图实时对照](reports/24图_实时对照.md) · [离线回放](preflight/离线回放.md) · [提示词](PROMPT.md)
""")
    DRIVER.publish(run)
    E.publish(run/"missing");R.publish(run/"review")
    return cfg

def freeze(run):
    files={}
    for directory in ("code","inputs","constraints_snapshot","preflight"):
        for f in (run/directory).rglob("*"):
            if f.is_file():files[str(f.relative_to(run))]=E.sha256(f)
    for f in [run/"config.json",run/"cases.json",run/"missing/config.json",run/"review/config.json"]:
        files[str(f.relative_to(run))]=E.sha256(f)
    for d in (run/"missing/cases").iterdir():
        for name in ("retained_candidates.json","retained_sam_union.png","compression_audit.json","filter_audit.json"):
            files[str((d/name).relative_to(run))]=E.sha256(d/name)
    E.write(run/"manifest.json",dict(frozen_at=E.now(),files=files))
    print(json.dumps(dict(run=str(run),frozen_files=len(files)),ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--project",type=Path,required=True);ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args();prepare(args.project,args.run);freeze(args.run)

