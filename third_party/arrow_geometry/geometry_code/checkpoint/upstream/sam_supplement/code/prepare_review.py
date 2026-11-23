"""Freeze review of new additions only; feed completed upstream cases without reruns."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image,ImageDraw
import experiment as E
import review_additions as R

WORK=Path(__file__).resolve().parent
PROJECT=WORK.parents[1]
ROOT=PROJECT/'runs/v030_sam_additions_review24'
SOURCE=PROJECT/'runs/v029_sam_additive_missing24'


def prepare():
    if ROOT.exists():raise RuntimeError('Run exists; never overwrite')
    test=subprocess.run([sys.executable,'-B','-m','unittest','test_review_additions'],cwd=WORK,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    cfg=E.read(SOURCE/'config.json')
    for key in ['execution_case_ids','linked_case_ids','linked_run']:
        cfg.pop(key,None)
    cfg.update(source_run=str(SOURCE),api_concurrency=32,review_case_workers=4,review_workers=4,
        review_rounds=2,review_batch_size=6,review_context_scale=1.6,review_context_expansions=1,
        protocol_version='sam-additions-only-review-v1',new_sam_calls=False,new_missing_glm_calls=False,
        new_addition_review_calls=True,baseline_immutable=True,geometry_executed=False)
    # Prioritize the user's example for inspection; this does not change any image rule.
    cases=E.read(SOURCE/'cases.json');cases=['task_0067']+[x for x in cases if x!='task_0067']
    ROOT.mkdir(parents=True);(ROOT/'preflight').mkdir()
    shutil.copytree(WORK,ROOT/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(SOURCE/'inputs',ROOT/'inputs')
    shutil.copytree(SOURCE/'constraints_snapshot',ROOT/'constraints_snapshot')
    E.write(ROOT/'cases.json',cases);E.write(ROOT/'config.json',cfg)
    (ROOT/'preflight/tests.txt').write_text(test.stdout+test.stderr)
    total=0;provenance=[]
    for cid in cases:
        baseline=E.read(ROOT/'inputs'/cid/'baseline_native.json');total+=len(baseline)
        image=Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB')
        final,audit=E.append_only(baseline,[],image.size)
        E.export_additive(cid,final,ROOT/'inputs'/cid/'source.png')
        out=ROOT/'cases'/cid
        for name in ['baseline.png','current.png']:E.plot(image,baseline,out/name)
        E.write(out/'baseline_preservation.json',audit)
        E.write(out/'status.json',dict(state='waiting_source',stage='waiting_upstream_completion',updated_at=E.now()))
        provenance.append(dict(case=cid,baseline_sha256=E.sha256(ROOT/'inputs'/cid/'baseline_native.json'),source=str((SOURCE/'cases'/cid).resolve())))
    # Offline view of the exact crop that the first 0067 review will use.
    cid='task_0067';baseline=E.read(ROOT/'inputs'/cid/'baseline_native.json')
    raw=E.read(SOURCE/'cases'/cid/'additions.json',[]);kept,removed=R.containment_filter(raw,baseline)
    image=Image.open(ROOT/'inputs'/cid/'source.png').convert('RGB')
    for i,focus in enumerate(R.batches(kept,cfg['review_batch_size']),1):
        d=ROOT/'preflight'/cid/f'B{i:03d}';view=R.context_frame(focus,baseline,image.size,cfg['review_context_scale'])
        new,na=R.visual_rows(focus,view);old,oa=R.visual_rows(baseline,view)
        clean=view.image(image);E.save(clean,d/'clean.png');im=clean.copy();draw=ImageDraw.Draw(im)
        for rows,color in [(old,'#d37811'),(new,'#1679d3')]:
            for row in rows:E.corners(draw,row['bbox'],color,row['id'])
        E.save(im,d/'boxes.png');E.write(d/'prompt.json',R.review_prompt(view.input_size,new,old))
        E.write(d/'frame.json',view.metadata(image.size))
    E.write(ROOT/'preflight/checks.json',dict(passed=True,test_log='tests.txt',validated_baseline_cases=len(cases),
        validated_baseline_regions=total,api_requests_sent=0,readonly_originals=True))
    E.write(ROOT/'provenance.json',dict(created_at=E.now(),source_run=str(SOURCE),baseline_sources=provenance,
        spatial_review_source=str(PROJECT/'runs/v022_fan_grouping24/code/local_review.py'),
        source_code_sha256=E.sha256(PROJECT/'runs/v022_fan_grouping24/code/local_review.py'),
        mainline_changed=False,upstream_code_changed=False,source_candidates_frozen_after_case_completion=True))
    (ROOT/'PROMPT.md').write_text('# 新增框整理提示词\n\n'+R.SYSTEM+'\n\n```json\n'+json.dumps(R.review_prompt((1080,720),[],[]),ensure_ascii=False,indent=2)+'\n```\n')
    (ROOT/'README.md').write_text('''# SAM补漏新增框整理：24图

只整理v029产生的新增框，原始定位不修改。已结束图片立即进入整理，未结束的等待原始补漏结束；不重新调用发现或SAM，不改当前运行代码/产物。

## 处理顺序

1. 冻结该图新增快照；在原图坐标下，删除完全落在单个原始框内的新增候选。没有padding、容差或覆盖率阈值；部分伸出保留，多个原框的并集不用于包含判断。保存排除依据。
2. 沿用v022的小批空间调度，每批最多6个新增框，先整理一轮，再对跨批尚未共同审核的重叠新增框复查一轮。空间调度只决定请求分组，不自动决定合框。
3. 每个请求使用重新裁剪的CLEAN和同坐标框图。当前新增框可编辑，原始框只读；观察窗口包进与当前新增框直接相交的原始框完整范围，再加1.6倍上下文，不递归扩散到所有邻框。看不全可扩大一次。
4. 模型只对新增框作keep、替换（修正/拆分/合并）或discard。检查实际笔画、完整可见范围、真实端点、是否原框已覆盖。不给前一步长解释和SAM mask，防止从描述/单个mask推导不存在的连接。
5. 修正结果再做一次严格包含过滤。原始框字段、顺序、原图坐标逐项保持。失败和无法确认的新增候选单独待审，保留JSON，不作为已经审核的新框导出。

没有重新开放旧框修改；原框没框全时，可以保留另一个完整的新框。未运行几何、样式或全局修正。新增数减少不等于召回提高，需看两栏图与trace。

## 配置

24张与上游同一清单，0067优先进入验证；不是针对0067设置算法规则。4图并行，每图最多4个整理请求，API上限32，实际此阶段最多16并发。上游两进程最多96请求，此阶段最多16，合计不超128。

默认thinking enabled、省略reasoning_effort，113000输出上限、1200秒时限；传输与格式重试沿用原client。两轮整理、每批最多一次扩图；剩余跨批歧义记录为issue。旧框保护/只读拒绝、修框、拆合、空框删除、上下文、坐标映射等11项离线测试通过。质量需等实际请求。

[24图整理前后](reports/24图_整理前后对照.md) · [提示词](PROMPT.md) · [配置](config.json) · [检查](preflight/checks.json)
''')
    E.write(ROOT/'execution.json',dict(state='prepared',created_at=E.now()))
    files=[f for part in ['code','inputs','constraints_snapshot'] for f in (ROOT/part).rglob('*') if f.is_file()]
    files += [ROOT/'config.json',ROOT/'cases.json',ROOT/'provenance.json']
    E.write(ROOT/'manifest.json',dict(created_at=E.now(),files={str(f.relative_to(ROOT)):E.sha256(f) for f in files}))
    (ROOT/'launch.sh').write_text(f'''#!/usr/bin/env bash
set -euo pipefail
cd '{ROOT}'
export PYTHONDONTWRITEBYTECODE=1
unset ARROW_API_KEY BOYUE_API_KEY
export ARROW_API_KEY_FILE='/home/yinjinhui/.config/arrow_partition_points/v022_fan_grouping24.key'
exec '{sys.executable}' -B -u '{ROOT}/code/review_additions.py' run --run-dir '{ROOT}' >> '{ROOT}/run.log' 2>&1
''');(ROOT/'launch.sh').chmod(0o750)
    R.publish(ROOT);print(json.dumps(dict(run=str(ROOT),checks=E.read(ROOT/'preflight/checks.json')),ensure_ascii=False))


if __name__=='__main__':prepare()
