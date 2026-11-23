"""Freeze the A geometry implementation and attach reviewed SAM supplement output."""
import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
import run_geometry_dedup24 as Q


def prepare(project,run):
    if run.exists():raise FileExistsError(run)
    work=Path(__file__).resolve().parent.parent
    previous=project/'runs/v010_A_full24_sam_missing4/reconstruction'
    controller=project/'runs/v031_sam_protocol24/receiver_v3_clamp';source=controller/'review'
    cfg=Q.read(previous/'config.json')
    upstream_done=Q.read(controller/'execution.json',{}).get('state')=='finished'
    for key in ('wait_for_geometry_run','source_provider'):cfg.pop(key,None)
    cfg.update(source_localization=str(source),source_controller_run=str(controller),
        bbox_discovery_run=str(source),discovery_provider='bbox_handoff',
        case_workers=16,api_concurrency=128 if upstream_done else 112,task_workers=16,cpu_workers=4,sam_workers=2,
        geometry_critic_enabled=True,geometry_critic_max_refinements=1,
        geometry_preview_width_original_px=2.0,geometry_preview_default_head_size='med/med',
        reconcile_enabled=True,reconcile_max_paths=64,pipeline_mode='geometry',
        coverage_rounds=0,scope_review=False,separate_style=False,
        model_cache_from=None,preview_interval_seconds=10,
        source_provider='v031_reviewed_sam_supplement_clamped',
        run_label='v032 current localization geometry and local model deduplication')
    cfg.pop('reasoning_effort',None)
    assert cfg['thinking']=={'type':'enabled'} and cfg['max_tokens']==113000
    run.mkdir(parents=True);shutil.copytree(work/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(work/'geometry_code',run/'geometry_code',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(controller/'constraints_snapshot',run/'constraints_snapshot',symlinks=True)
    ids=Q.read(controller/'cases.json');Q.write(run/'cases.json',ids);Q.write(run/'config.json',cfg)
    sys.path[:0]=[str(run/'geometry_code'),str(run/'geometry_code/lib')]
    from PIL import Image
    import bbox_inputs,bbox_geometry
    from native_runtime import validated_renderer
    renderer,version=validated_renderer()
    Q.write(run/'preflight/renderer.json',dict(executable=str(renderer),version=version))
    smoke=Q.read(work/'preflight/queue_smoke/checks.json')
    assert smoke['execution_state']=='finished' and smoke['source_background_preserved']
    Q.write(run/'preflight/queue_smoke.json',smoke)
    Q.write(run/'preflight/regression_tests.json',Q.read(work/'preflight/regression_tests.json'))
    evidence=[]
    for cid in ids:
        if not Q.ready(source,cid):
            evidence.append(dict(case=cid,state='waiting_localization'));continue
        out=Q.freeze_case(source,run,cid);clean=Image.open(out/'clean.png').convert('RGB')
        raw=Q.read(out/'reconstruction_handoff.json');plan=bbox_inputs.normalize(raw,clean.size,cid)
        result=Q.read(out/'localization_result.json');native=result['native_regions']
        assert len(raw['tasks'])==len(native),(cid,'candidate count')
        by_id={r['id']:r for r in native}
        for item in raw['tasks']:
            assert item['render_owner'] is True
            assert item['bbox_native']==by_id[item['region_id']]['bbox'],(cid,item['target_id'],'native frame mismatch')
        assert len(plan['tasks'])==len(raw['tasks']) and not plan['issues']
        Q.write(out/'geometry_plan.json',plan)
        evidence.append(dict(case=cid,state='ready',tasks=len(plan['tasks']),
            blocks=sum(t['render_type']=='block_arrow' for t in plan['tasks']),
            native_size=list(clean.size),bbox_native_exact=True,baseline_and_reviewed_only=True))
    Q.write(run/'preflight/handoff_checks.json',dict(cases=evidence,model_calls=0,
        ready_cases=sum(e['state']=='ready' for e in evidence),tasks=sum(e.get('tasks',0) for e in evidence)))
    old=project.parents[3]/'experiments/bbox_discovery_reconstruction_20261025_v1/runs/local_reconcile_full4_v3/code'
    def funcs(path):return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(path.read_text()).body if isinstance(f,ast.FunctionDef)}
    a=funcs(old/'local_reconcile.py');b=funcs(run/'geometry_code/reconcile_geometry.py')
    assert all(a[n]==b[n] for n in ['prompt','validate','transform_document','labeled_current','apply_patch'])
    assert Q.sha(old/'reconcile_scopes.py')==Q.sha(run/'geometry_code/reconcile_scopes.py')
    geometry_files={str(f.relative_to(run/'geometry_code')):Q.sha(f) for f in (run/'geometry_code').rglob('*') if f.is_file()}
    assert all(Q.sha(previous/'geometry_code'/n)==digest for n,digest in geometry_files.items())
    Q.write(run/'provenance.json',dict(prepared_at=Q.now(),geometry_source=str(previous/'geometry_code'),
        geometry_files_identical=len(geometry_files),localization_source=str(source),
        dedup_reference=str(old),dedup_prompt_and_scope_unchanged=True,
        global_document='A version preserves crop-to-native frame conversion and verifies endpoints',
        scope_cap_reference='task0111_postgeometry_reconcile_v1: 64 paths; larger groups deferred intact',
        geometry_results_reused=False,discovery_and_sam_missing_rerun=False))
    Q.write(run/'execution.json',dict(state='prepared',prepared_at=Q.now()))
    (run/'README.md').write_text(f'''# v032：当前定位 → 几何恢复 → 模型联合去重（24图）

输入固定为v031/receiver_v3_clamp的review产物（原始框＋审核保留的补漏框）；尚未结束的case等待完成后冻结。原始定位、SAM补漏不重跑，未审核新增不作为绘制目标。上游issues随case保留。

几何实现逐文件复用v010 A方案：逐框独立GLM点线恢复，普通箭头恢复头型与方向，最多一次几何critic refine。块状箭头走独立shape几何，保留crop SAM证据/轮廓回退；普通线条不重新走SAM身份确认。

全部几何返回后，先保存去重前PPT/原图叠图，再做一次已有局部联合去重：自动几何邻近只决定送审区域，CLEAN原图＋彩色当前路径＋只读周围路径＋点协议交给模型判断整段/部分重复。prompt、自动选区和提交规则沿用local_reconcile_full4_v3；当前A版global_document保留坐标修复。普通线条/箭头参与此去重，块状shape不混进点线协议。沿用后续0111实验的64路径组上限；超大组完整待处理，不截断成员、不程序删除。

几何与去重重新请求，不复用旧重建。默认thinking enabled，不传reasoning_effort；113000输出、1200秒总时限、120秒idle保持。几何refine=1；默认线宽2原图像素、头med/med、实线灰色。没有精细样式、额外补漏或范围审核。

16图并行，本进程API上限{cfg['api_concurrency']}，每图16目标并行，CPU4/SAM2。若源定位尚在运行，预留16API名额，使合计不超过128。LibreOffice26.8和原生头部安全放大修正沿用；叠图以原图为底，不铺白底。

报告每15秒刷新，目标返回后约10秒更新一次轻量预览；去重前后均保留实际PPT渲染。模型失败/不确定时保留原几何。执行完成或路径减少均不等于视觉正确，结果需看图验收。

[24图实时对照](reports/24图_几何去重实时对照.md) · [配置](config.json) · [交接验证](preflight/handoff_checks.json) · [来源](provenance.json)
''')
    Q.publish(run)
    files={}
    for sub in ['code','geometry_code','constraints_snapshot','preflight']:
        for f in (run/sub).rglob('*'):
            if f.is_file():files[str(f.relative_to(run))]=Q.sha(f)
    for name in ['config.json','cases.json','provenance.json']:files[name]=Q.sha(run/name)
    Q.write(run/'manifest.json',dict(frozen_at=Q.now(),files=files))
    print(json.dumps(dict(run=str(run),ready=sum(e['state']=='ready' for e in evidence),
        targets=sum(e.get('tasks',0) for e in evidence),api_concurrency=cfg['api_concurrency'],model_calls=0),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();prepare(a.project.resolve(),a.run.resolve())
