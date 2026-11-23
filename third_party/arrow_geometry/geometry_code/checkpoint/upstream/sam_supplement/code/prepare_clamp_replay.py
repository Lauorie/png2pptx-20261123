"""Same v031 run: bounded-coordinate receiver, exact reply reuse then review."""
import argparse
import shutil
from pathlib import Path
import experiment as E
import prepare_protocol24 as P
import run_protocol24 as D
from supplement_protocol import VERSION
from verify_bbox_clamp import run as verify


def prepare(project, source, run):
    P.prepare(project,run)
    cfg=E.read(run/'config.json')
    cfg.update(protocol_version=VERSION,bbox_policy='clip_to_input_pixel_centers',
        reuse_from=str(source),source_live_run=str(source),missing_cache_only=True,
        new_missing_glm_calls=False,case_workers=4,api_concurrency=32)
    E.write(run/'config.json',cfg)
    for stage in ('missing','review'):
        scfg=dict(cfg)
        if stage=='review':scfg['source_run']=str(run/'missing')
        E.write(run/stage/'config.json',scfg)
    evidence=E.read(run/'provenance.json')
    evidence.update(source_code=str(Path(__file__).parent),source_model_run=str(source),
        no_prior_supplemental_glm_results_reused=False,
        reuse_policy='exact_same_prompt_images_model_parameters_complete_response',
        raw_model_outputs_immutable=True,
        user_authorized_change='Clamp bbox coordinates to the input image lower and upper bounds')
    E.write(run/'provenance.json',evidence)
    checks=verify(source,run/'preflight/clamp_replay')
    E.write(run/'preflight/checks.json',dict(protocol_tests=28,baseline_boxes=627,
        input_pngs_checked=288,model_calls=0,clamp_replay=checks['counts']))
    E.write(run/'constraints_snapshot/bbox_boundary_policy.json',dict(
        user_request='越界把他拉回去，设置一个上下限，你这样直接丢弃的危害更大',
        applies_to=['SAM missing candidates','new-addition review replacements'],
        policy='x in [0,W-1], y in [0,H-1] in actual current CLEAN input pixel centers',
        original_model_reply_preserved=True,baseline_boxes_immutable=True))
    (run/'README.md').write_text('''# v031 receiver_v3_clamp：越界框截断后继续

按用户要求，补漏和新增框整理均将越界坐标截断至当前CLEAN输入图的 `x∈[0,W−1]`、`y∈[0,H−1]`。先截断局部坐标，再按原有变换换算回原图；不改变坐标系、不缩放坐标猜测语义。

原始模型回复及`raw_local_bbox`保持，`effective_local_bbox`记录实际采用坐标，`protocol_audit.bbox_adjustments`记录上下限与改动。倒置、非数值、与图像没有正面积交集的框仍单独待处理；不创造线条。旧框627个保持不变。

这是v031同轮回复的接收修正，不是另一轮定位/SAM实验。补漏仅精确复用父运行完整回复，禁止新补漏调用；4个无正文超时不会被伪装为成功。整理在源图结束后接入；相同请求优先复用，恢复候选造成上下文变化时重新审核。原运行和receiver_v2保留。

本进程4图、API32，整理每图最多4请求；与当前已有进程实际请求总数保持低于128。默认思考，未设置reasoning_effort，输出预算113000保持。未执行几何/PPT/样式。

28项测试通过；本轮完整回复的零API离线回放见下方。回放恢复表示可被流程接收，不等于视觉正确，最终去留由新增整理判断。preflight内旧离线回放是初版协议历史证据，当前验证以clamp_replay为准。

[24图实时结果](reports/24图_实时对照.md) · [越界回复回放](preflight/clamp_replay/越界回放.md) · [实际提示词](PROMPT.md)
''')
    (run/'preflight/离线回放.md').write_text('# 当前边界修正验证\n\n[越界回复回放](clamp_replay/越界回放.md)。28项协议与交接测试通过，原始框保护和输入PNG一致性检查仍通过。\n')
    D.publish(run)
    P.freeze(run)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();prepare(args.project.resolve(),args.source.resolve(),args.run.resolve())
