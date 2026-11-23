"""Geometry-first omission nomination and one-candidate visual checks.

Only additions enter current reconstruction. Global GLM proposals and deferred
localization candidates bypass the localization-box filter; SAM proposals use
initial-geometry task boxes as a cheap scope prefilter.
"""
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bbox_geometry
import coverage_visual
import missing_box_consolidation
import evidence as ev
import protocol
from scene_snapshot import canonical_rows
from transport.common import read_json as read, write_json as write, sha256

SYSTEM = "对照原图与已有重建，判断当前候选是否需要补漏。依据可见箭头和线条判断，只输出指定 JSON。"
CHECK_PROTOCOL = "post_geometry_missing_review.3"
UNRESOLVED_DECISIONS = {"needs_context", "uncertain"}


def clipped_box(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("bbox requires four finite numbers")
    box = [max(0., min(size[i % 2]-1., float(x))) for i, x in enumerate(value)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox has no visible area")
    return box


def map_box(box, matrix):
    return (np.c_[np.asarray(box).reshape(2, 2), np.ones(2)] @
            np.asarray(matrix).T)[:, :2].reshape(-1).tolist()


def proposal_reply(value, size):
    if not isinstance(value, dict) or not isinstance(value.get("regions"), list):
        raise ValueError("Return regions list, or []")
    answer = copy.deepcopy(value)
    for row in answer["regions"]:
        row["bbox"] = clipped_box(row.get("bbox"), size)
    return answer


def check_reply(value, size):
    if not isinstance(value, dict) or value.get("decision") not in {
            "covered", "not_arrow", "missing", "needs_context", "uncertain"}:
        raise ValueError("Use covered/not_arrow/missing/needs_context/uncertain")
    if any(value.get(k) for k in ("replace", "delete", "updates", "nodes", "routes")):
        raise ValueError("This task cannot edit existing geometry or recover paths")
    regions = value.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("regions must be a list")
    if (value["decision"] == "missing") != bool(regions):
        raise ValueError("Only missing has nonempty regions")
    result = proposal_reply({**value, "regions": regions}, size)
    for row in result["regions"]:
        if row.get("render_type") not in ("linework", "block_arrow"):
            raise ValueError("Each missing region needs render_type")
        if row.get("grouping", "single") not in ("single", "group"):
            raise ValueError("grouping must be single/group")
    return result


def _iou(a, b):
    area = lambda v: (v[2]-v[0])*(v[3]-v[1])
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return inter / (area(a)+area(b)-inter) if inter else 0.


def candidate_queue(global_regions, sam_regions, size, deferred_candidates=None):
    """Only nearly identical boxes share a request; no overlap-based bundling."""
    result, aliases = [], []
    for source, rows in (("GLOBAL", global_regions),
                         ("LOCALIZATION_UNCERTAIN", deferred_candidates or []), ("SAM", sam_regions)):
        for i, row in enumerate(rows, 1):
            key = row['target_id'] if source == 'LOCALIZATION_UNCERTAIN' else (row['id'] if source == 'SAM' else f'{i:03d}')
            bbox = row['bbox_native'] if source == 'LOCALIZATION_UNCERTAIN' else row['bbox']
            cid = source + "_" + str(key)
            item = dict(id=cid, source=source, bbox_native=clipped_box(bbox, size),
                        origin=copy.deepcopy(row))
            # Keep global proposals first: a large old localization box never vetoes them.
            duplicate = next((x for x in result if _iou(x["bbox_native"], item["bbox_native"]) >= .9), None)
            if duplicate:
                duplicate.setdefault("aliases", []).append(item)
                aliases.append(dict(candidate=cid, kept=duplicate["id"], rule="bbox_iou>=0.9"))
            else:
                result.append(item)
    return result, aliases


def global_discover(clean, rows, folder, config, model, locks):
    folder = Path(folder)
    with locks["cpu"]:
        cfg = {**config, "context_padding": 0, "min_crop_side": 1,
               "display_long_side_range": config.get("geometry_input_long_side_range", [1080, 1536])}
        crop, frame = ev.frame(clean, [[0, 0, *clean.size]], folder, cfg)
        coverage_visual.render(crop, rows, frame, folder/"current.png", labels=False)
    prompt = protocol.coverage(list(crop.size), input_mode="two_images")
    prompt["instructions"].append(
        "A suspected omission may be inside an existing localization box. Judge actual blue recovered geometry, "
        "not box coverage. The next step checks each proposed bbox independently. Ignore small semantic-equivalent offsets.")
    answer = model.ask(folder/"request", prompt,
                       [("ORIGINAL intact source", folder/"clean.png"),
                        ("RECOVERED blue actual geometry on gray source", folder/"current.png")],
                       lambda x: proposal_reply(x, crop.size),
                       system="Compare original and actual reconstruction. Propose omission regions only, using the requested JSON schema.")
    mapped = [{**r, "bbox": clipped_box(map_box(r["bbox"], frame["view_to_global"]), clean.size)}
              for r in answer["regions"]]
    write(folder/"native_regions.json", mapped)
    return mapped


def _sam_overlay(crop, frame, candidate, sam_case):
    """Project only this candidate's cached masks; no neighboring candidate flood."""
    image = crop.convert("RGBA")
    origins = [candidate, *candidate.get("aliases", [])]
    members = {m for c in origins if c["source"] == "SAM" for m in c["origin"].get("member_ids", [])}
    if not members or sam_case is None:
        return image.convert("RGB")
    raw = {r["id"]: r for r in read(Path(sam_case)/"raw_candidates.json")}
    union = np.zeros((crop.height, crop.width), dtype=bool)
    for mid in sorted(members):
        row = raw[mid]
        meta = read(Path(sam_case)/"sam"/row["view"]/"frame.json")
        x0, y0, x1, y1 = meta["crop_bounds_global_half_open"]
        w, h = meta["actual_crop_input_size_px"]
        sx, sy = w/(x1-x0), h/(y1-y0)
        source_view = dict(global_to_view=[[sx, 0, (-x0+.5)*sx-.5],
                                          [0, sy, (-y0+.5)*sy-.5], [0, 0, 1]])
        path = Path(row["mask_path"])
        if sha256(path) != row["mask_sha256"]:
            raise ValueError("SAM mask changed: " + mid)
        union |= ev.project(dict(view=source_view, path=path), frame)
    tint = Image.new("RGBA", image.size, (30, 175, 210, 0))
    tint.putalpha(Image.fromarray(union.astype("uint8")*40))
    image.alpha_composite(tint)
    return image.convert("RGB")


def check_prompt(candidate, bbox, size, expanded=False, *, frame=None, context=None):
    return dict(
        protocol=CHECK_PROTOCOL,
        task='从当前候选涉及的内容中，提报本阶段应恢复且尚未由蓝色重建表达的箭头或独立线条。只判断是否需要新增，不审核已有几何的精度或样式，不修改已有路径。',
        input_size=list(size), candidate=dict(id=candidate["id"], bbox=bbox),
        images=dict(
            CLEAN="局部原图，唯一外观依据；所有输出 bbox 使用这张图的像素坐标。",
            CURRENT=("与 CLEAN 同范围、同尺寸。这张图由两层组成：灰色底图是完整原图转灰并淡化，仅供位置对照，"
                     "包括尚未重建的箭头、文字和形状；即使灰色箭头的线身和头部完整可见，也不表示已重建或已覆盖。"
                     "叠加的蓝色笔画才是已有重建；判断 covered 必须找到对应的蓝色重建，不能把灰色底图当作覆盖证据。"
                     "已有重建统一用蓝色显示，不会因为样式尚未恢复而变成灰色。蓝色仅用于区分重建与底图，"
                     "其头型、大小、线宽和虚实可能尚未定稿，目标外观仍以 CLEAN 为准。"),
            **({"CONTEXT": "更大范围的原图，用于识别对象归属和可见延续；标记只作定位，不从此图输出坐标或发现无关目标。"}
               if context is not None else
               {"CANDIDATE": "与 CLEAN 同范围；框和青色 SAM 像素只定位当前候选，不证明它是箭头或确有遗漏。"})),
        **({"source_view": dict(
            source_image_size=list(frame["global_size"]),
            source_crop_size=[frame["bbox_global"][2]-frame["bbox_global"][0],
                              frame["bbox_global"][3]-frame["bbox_global"][1]],
            display_scale_xy=[round(v, 3) for v in frame["scale"]])} if frame is not None else {}),
        **({"context_view": context} if context is not None else {}),
        rules=[
            "具有明确箭身、指向或头部特征的箭头属于恢复目标，即使它以字符形式出现在文字或标签中，也按箭头处理。不要求连接两个节点，不因颜色浅、位于容器内、组成循环图标或装饰图案而排除。不具备箭头结构的普通文字笔画、加减乘除等数学符号，以及照片纹理和仅有尖角的非箭头形状不补。",
            '本阶段恢复箭头和独立线条，包括连接线、独立分隔线、轴线、括号和标注引线，不限实线或虚线、有头或无头。这里的 shape 指作为整体表达一个对象、容器或结构的图形，例如闭合的矩形框、虚线框、圆角矩形框、圆等；可以带有填充，也可以没有填充，也可以由多个图元组成。对于非箭头 shape，其围合边框、外轮廓，以及构成其内部结构的网格线、分隔线等，均属于该 shape，不在本阶段恢复，返回 not_arrow。边框是虚线，或某条边、某段网格被单独框选或裁剪，都不使它成为独立线条。独立于这些轮廓和内部结构、单独表达连接、分隔、坐标或标注作用的线条仍需恢复；shape 内部或穿过 shape 的真实箭头、独立连接线也仍需恢复。同一候选中同时存在这些结构线和真实箭头、独立连接线时，只排除结构线，继续审核箭头和连接线，不得将整个候选一并排除。候选只是待审核线索。先依据 CLEAN 和 CONTEXT 判断对象归属，再对范围内目标检查蓝色覆盖；没有蓝色覆盖不等于应补。',
            "默认 render_type=linework，以线身加端点头型表达。只有整体箭身需要独立的填充和描边、普通线条无法保留其外观时才选 block_arrow，例如箭身填充与轮廓不同或空心宽体轮廓。粗、宽、局部放大或实心三角头本身都不是选 block_arrow 的理由。",
            '对属于恢复范围的目标，只判断蓝色重建是否已经表达同一对象或连接。已有表达时不因位置偏差、局部不贴合、头型或样式差异重复新增，也无需量化这些差异；本次无需新增归为 covered。真正未表达的箭头、独立线条或分支应提报为 missing。附近有蓝线或部分目标已恢复，不代表其他目标也已表达，应分别判断。',
            '审核候选框涉及的内容，周围视野仅用于判断对象归属和可见延续。框内各目标分别判断，不能因某个目标已覆盖或不属于范围，就排除其他真实遗漏。只有确认 missing 后才给遗漏目标的 bbox，可沿原图可见笔画调整必要范围；无需新增时 regions=[]。不同目标分别框选，共享网络可 grouping=group。不扫描无关目标，不推测遮挡连接，交叉不自动相连。',
            ("若目标被裁断，或当前视野不足以确认箭头/连接线结构，返回 needs_context 扩大同一候选视野。"
             if not expanded else "已扩大视野；仍无法确认箭头/连接线结构或可见延续则返回 uncertain，不猜测。"),
        ],
        decisions=dict(
            covered="属于恢复范围的目标均已有蓝色重建表达，本次无需新增；不表示几何精度或样式正确。灰色底图不计覆盖。regions=[]。",
            not_arrow="候选涉及的内容中没有本阶段应恢复的箭头或独立线条。regions=[]。",
            missing="存在应恢复而尚未表达的箭头、独立线条或分支；regions只列这些遗漏，不加入已恢复目标或shape结构线。",
            needs_context="需要更大视野判断同一候选，regions=[]。",
            uncertain="扩图后仍无法确定，regions=[]。"),
        output_schema=dict(decision="covered|not_arrow|missing|needs_context|uncertain",
                           reason="简短原图与已有几何证据", regions=[dict(bbox=[0, 0, 10, 10],
                           render_type="linework|block_arrow", grouping="single|group", label="简短描述")]))


def inspect_candidate(clean, rows, candidate, folder, config, model, locks, sam_case=None):
    folder = Path(folder)
    maximum = config.get("bbox_max_expansions", 1)
    for step in range(maximum+1):
        view_dir = folder/f"view_{step:02d}"
        cfg = {**config, "pilot_min_extent": config.get("bbox_min_extent", 25),
               "pilot_display_side": config.get("bbox_display_side", 1080),
               "crop_context_scale": config.get("missing_context_scale", 2.)}
        with locks["cpu"]:
            crop, frame = bbox_geometry.make_frame(clean, candidate["bbox_native"], view_dir, cfg, step)
            coverage_visual.render(crop, rows, frame, view_dir/"current.png", labels=False)
            box = map_box(candidate["bbox_native"], frame["global_to_view"])
            if config.get('missing_review_context_enabled', False):
                q, images = context_inputs(clean, candidate, frame, box, view_dir, step > 0)
            else:
                evidence = _sam_overlay(crop, frame, candidate, sam_case)
                draw = ImageDraw.Draw(evidence)
                draw.rectangle(box, outline="#de6e12", width=2)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                draw.text((box[0], max(0, box[1]-23)), candidate["id"], font=font,
                          fill="#de6e12", stroke_width=2, stroke_fill="white")
                evidence.save(view_dir/"candidate.png")
                q = check_prompt(candidate, box, crop.size, step == maximum, frame=frame)
                images = [("CLEAN", view_dir/"clean.png"), ("CURRENT", view_dir/"current.png"),
                          ("CANDIDATE", view_dir/"candidate.png")]
        answer = model.ask(view_dir/"request", q, images,
                           lambda x: check_reply(x, crop.size), system=SYSTEM)
        write(view_dir/"decision.json", answer)
        if answer["decision"] != "needs_context":
            break
    issues = []
    if answer["decision"] in UNRESOLVED_DECISIONS:
        issues.append(dict(stage="missing_check", candidate=candidate["id"], reason=answer.get("reason", ""),
                           decision=answer["decision"], unresolved=True))
    additions = [{**r, "bbox": clipped_box(map_box(r["bbox"], frame["view_to_global"]), clean.size)}
                 for r in answer["regions"]]
    write(folder/"native_additions.json", additions)
    write(folder/"resolution.json", dict(decision=answer['decision'], reason=answer.get('reason',''),
          view=step, unresolved=answer['decision'] in UNRESOLVED_DECISIONS))
    return additions, issues


def context_inputs(source, candidate, frame, box, view, expanded):
    """The same CLEAN/CURRENT/CONTEXT contract validated in the v063 study."""
    from missing_review_locator import annotate
    view = Path(view)
    context = source.copy(); context.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    sx, sy = context.width / source.width, context.height / source.height
    def xy(bounds):
        return [(float(v)+.5)*(sx if i % 2 == 0 else sy)-.5 for i,v in enumerate(bounds)]
    crop_box = xy(frame['bbox_global']); target_box = xy(candidate['bbox_native'])
    marked, local_locator = annotate(Image.open(view/'current.png').convert('RGB'), box)
    marked.save(view/'current_marked.png')
    context, context_locator = annotate(context, target_box, context=True, crop_box=crop_box)
    context.save(view/'context.png')
    meta = {'size': list(context.size), 'local_crop_box': crop_box, 'candidate_box': target_box,
            'legend': '紫框=CLEAN窗口；橙色Q角标与引线定位当前候选，显示框稍外扩，仅是标记。输出坐标仍用CLEAN，数值candidate.bbox是未外扩候选。'}
    write(view/'context_transform.json', {'source_size': list(source.size), 'scale_xy': [sx,sy], **meta})
    write(view/'locator_annotations.json', {'local':local_locator, 'context':context_locator})
    q = check_prompt(candidate, box, frame['size'], expanded=expanded, frame=frame, context=meta)
    q['candidate']['display_label'] = 'Q'
    q['images']['CURRENT'] += ' 橙色Q标签与外扩角标只定位当前候选，显示框稍外扩，不是原图笔画或已有几何；候选数值坐标不变。'
    return q, [('CLEAN', view/'clean.png'), ('CURRENT', view/'current_marked.png'), ('CONTEXT', view/'context.png')]


def run(clean, results, out, config, model, locks, case_id, notify=None, deferred_candidates=None):
    """Global, deferred localization and SAM -> bbox checks -> append-only geometry."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    original = copy.deepcopy(results)
    rows = canonical_rows(original)
    write(out/"geometry_before.json", original)
    issues = []
    def progress(stage):
        if notify: notify(stage)
    if config.get('resume_before_missing_from'):
        from resume_missing_checkpoint import load
        checkpoint = Path(config['resume_before_missing_from'])/case_id
        frozen, queue, provenance = load(checkpoint)
        if frozen != original:
            raise ValueError('Resume review must use the frozen pre-missing geometry')
        tasks = queue['tasks']; sam_case = None
        if not config.get('post_geometry_global_discovery_enabled', True):
            tasks = [t for t in tasks if t['source'] != 'GLOBAL']
            queue = {**queue, 'tasks': tasks, 'global_candidates': 0,
                     'global_discovery_enabled': False}
        issues.extend(provenance.get('discovery_issues', []))
        write(out/'candidates.json', queue)
        write(out/'discovery_reuse.json', provenance)
        write(out/'deferred_localization_input.json', deferred_candidates or [])
    else:
        global_regions = []
        if config.get('post_geometry_global_discovery_enabled', True):
            progress("global_missing_discovery")
            try:
                global_regions = global_discover(clean, rows, out/"global", config, model, locks)
            except Exception as exc:
                issue = dict(stage="global_missing_discovery", error=str(exc))
                issues.append(issue); write(out/"global/failure.json", issue)
        else:
            write(out/'global/skipped.json', dict(reason='disabled_by_run_config', model_calls=0))
        sam_case = None; sam_regions = []
        if config.get("post_geometry_sam_run"):
            sam_case = Path(config["post_geometry_sam_run"])/"cases"/case_id
            try:
                sam_regions = read(sam_case/"retained_candidates.json")
                scan_status = read(sam_case/"status.json")
                issues.extend(scan_status.get("issues", []))
                if scan_status.get("state") == "failed":
                    issues.append(dict(stage="sam_scan", error=scan_status.get("error", "SAM scan failed")))
            except Exception as exc:
                issue = dict(stage="sam_candidates_unavailable", error=str(exc))
                issues.append(issue); write(out/"sam_failure.json", issue)
        deferred_candidates = copy.deepcopy(deferred_candidates or [])
        write(out/'deferred_localization_input.json', deferred_candidates)
        tasks, aliases = candidate_queue(global_regions, sam_regions, clean.size, deferred_candidates)
        write(out/"candidates.json", dict(tasks=tasks, near_identical_aliases=aliases,
              global_candidates=len(global_regions), sam_candidates=len(sam_regions),
              deferred_localization_candidates=len(deferred_candidates),
              global_proposals_filtered_by_localization=False, one_bbox_per_check=True, auto_bundle=False))
    progress("single_bbox_missing_check")
    checked = {}
    resolutions = {}
    with ThreadPoolExecutor(max_workers=config.get("task_workers", 8)) as pool:
        futures = {pool.submit(inspect_candidate, clean, rows, t, out/"candidates"/t["id"],
                              config, model, locks, sam_case): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                checked[t["id"]], errors = future.result(); issues.extend(errors)
                resolutions[t['id']] = read(out/'candidates'/t['id']/'resolution.json')
            except Exception as exc:
                checked.pop(t['id'], None)
                issue = dict(stage="missing_check", candidate=t["id"], error=str(exc))
                resolutions[t['id']] = dict(decision='failed', unresolved=True, error=str(exc))
                issues.append(issue); write(out/"candidates"/t["id"]/"failure.json", issue)
    write(out/'check_resolutions.json', resolutions)
    write(out/'unresolved_candidates.json', [dict(t,resolution=resolutions[t['id']]) for t in tasks
          if resolutions[t['id']].get('unresolved')])
    write(out/'pending_existing_repairs.json', [dict(t,resolution=resolutions[t['id']]) for t in tasks
          if resolutions[t['id']].get('decision') == 'existing_geometry_issue'])
    additions = []
    for t in tasks:
        for i, r in enumerate(checked.get(t["id"], []), 1):
            additions.append(dict(id=f"MISSING_{t['id']}_{i:03d}", bbox_native=r["bbox"],
                locator=r.get("label", ""), render_type=r["render_type"],
                grouping=r.get("grouping", "single"), readonly=[], missing_only=True,
                source_candidate=t["id"]))
    write(out/"additions_before_consolidation.json", additions)
    consolidation = None
    if config.get("missing_box_consolidation_enabled", False):
        additions, consolidation = missing_box_consolidation.consolidate(
            additions, rows, threshold=config.get("missing_box_iou_threshold", .7),
            filter_existing=config.get("missing_existing_bbox_filter_enabled", True))
        write(out/"box_consolidation.json", consolidation)
    write(out/"additions.json", additions)
    additions = [dict(target, existing_geometry=rows) for target in additions]
    progress("missing_geometry_and_critic")
    new_results = {}
    with ThreadPoolExecutor(max_workers=config.get("task_workers", 8)) as pool:
        futures = {pool.submit(bbox_geometry.recover_target, clean, t, out/"geometry"/t["id"],
                              model, config, locks): t for t in additions}
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                if result.get("discovery_task"):
                    result["discovery_task"] = {k:v for k,v in result["discovery_task"].items() if k != "existing_geometry"}
                new_results[t["id"]] = result
                issues.extend(result.get("issues", []))
            except Exception as exc:
                issue = dict(stage="missing_geometry", candidate=t["id"], error=str(exc))
                issues.append(issue); write(out/"geometry"/t["id"]/"failure.json", issue)
    # No replacement/deletion operation is available to this stage.
    final = original + [new_results[t["id"]] for t in additions if t["id"] in new_results]
    assert final[:len(original)] == original and results == original
    write(out/"geometry_after.json", final)
    queue = read(out/'candidates.json')
    write(out/"audit.json", dict(existing_geometry_unchanged=True, global_candidates=queue.get('global_candidates', sum(t['source']=='GLOBAL' for t in tasks)),
          deferred_localization_candidates=queue.get('deferred_localization_candidates', len(deferred_candidates or [])),
          sam_candidates=queue.get('sam_candidates', sum(t['source']=='SAM' for t in tasks)), check_candidates=len(tasks), new_geometry_targets=len(additions),
          box_consolidation=({k:consolidation[k] for k in ("input_count", "merged_count", "filtered_count", "output_count")} if consolidation else None),
          new_groups=len(final)-len(original), issues=issues))
    return final, issues

