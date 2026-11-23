"""Position observations and atomic model edits; no physical-owner graph."""
import copy
import math

SYSTEM = """Locate visible arrow/linework regions for later geometry recovery.
Each returned region becomes one geometry request. Keep isolated/sparse arrows separate.
Shared branches, regular bundles and manageable local networks may stay together.
Split only when the resulting regions reduce connection ambiguity and recovery burden;
consider whether splitting increases overlapping boxes and confusing context.
Arrow count alone does not decide grouping. A group can contain independent paths.
Discover locations, not path geometry. Concise descriptions never mean fewer regions.
Return concise JSON. Use absolute pixel coordinates in the output image frame
explicitly named by the current request; never convert coordinates yourself."""

RULES = [
    "label只写不依赖坐标系的简短视觉语义。数值位置只能写在结构化bbox字段，label不能写x/y数值、坐标对或像素位置；输入描述若含旧坐标，不复制、不换算、不把它当作端点证据。图中节点名称与层编号可保留。",
    "每个region都会直接交给后续一次几何恢复。目标是找全可见连接并安排可处理的视觉任务，不是按架构、流程链或相似外观概括区域；不输出几何点、精细样式或物理身份关系。",
    "孤立、零星、容易各自恢复的箭头分别框出。被模块/节点隔开的短箭头有各自起终点，即使共线、同色、同向或串成同一流程，也分别定位；流程中的串联不等于共享主干。",
    "共享分支、局部扇入扇出、规则线束及信息量可控的局部网络可以整体恢复，不按箭头数量限制成组。group中允许多条独立路径；合组不意味着同一箭头，交叉不意味着连接。",
    "按实际恢复难度决定粒度：起终点、走向是否容易共同辨认，是否有大量各向异且含糊的连接。只有拆分能降低连接歧义和恢复负担时才拆；同时检查拆后是否增加bbox重叠、上下文干扰和重复恢复。不是框越细越好，也不能仅因同向、同节点或同属网络就合组。",
    "每个框覆盖当前目标实际可见的箭身、头、转弯和末端。到达该箭头可见终点就停止，不把下一个节点另一侧的箭头当作它的延续；原图有遮挡则保留可见片段，不推导隐藏连接。裁剪截断时才借全图找回这条连接的可见延续。",
    "按实际图像逐个定位，不按等距模板复制框、猜数量或把整行/整列一次概括。只缩短label文字，不以少输出框或缩短JSON为目标；不能为了省输出而合并任务。",
    "允许少量重叠观察。矩形重叠或包含不证明同一对象，不证明覆盖完整；不要只因处于已有大框里就忽略真实遗漏。",
    "linework包括普通箭头、无头线、曲线、虚线、括号、轴、引线和框内真实网络。block_arrow是箭身与头构成二维闭合轮廓的块状箭头，实心/空心均包括；普通三角箭头头部仍属linework。",
    "块状箭头与普通线条分别输出区域，即使二者相交。非箭头闭合形状的轮廓、格网边、字形、照片纹理不属于本任务；分子结构只找外部连接。",
    "以干净原图为准，旧框与描述只是待校正的观察。不能推断不存在的连接，不能把裁剪窗口当作恢复对象。",
]


def number_box(value, size):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox needs four GLOBAL pixel coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("bbox needs finite JSON numbers, not bool/string")
    if not (0 <= value[0] < value[2] <= size[0]-1 and
            0 <= value[1] < value[3] <= size[1]-1):
        raise ValueError(f"bbox outside GLOBAL pixel centers 0..{size[0]-1}, 0..{size[1]-1}: {value}")
    return value


REGION = {"id": "R1", "bbox": [0, 0, 10, 10],
          "render_type": "linework|block_arrow",
          "grouping": "single|group", "label": "不含数值坐标的短视觉描述；group简述可见线条及整体恢复理由",
          "extent": "complete|uncertain"}


def regions_valid(rows, size):
    if not isinstance(rows, list):
        raise ValueError("regions must be an array")
    ids = set()
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] or r["id"] in ids:
            raise ValueError("Unique nonempty region IDs required")
        ids.add(r["id"])
        number_box(r.get("bbox"), size)
        if r.get("render_type") not in {"linework", "block_arrow"}:
            raise ValueError("Separate linework and block_arrow regions required")
        if r.get("grouping") not in {"single", "group"}:
            raise ValueError("grouping must be single or group")
        if r.get("extent") not in {"complete", "uncertain"}:
            raise ValueError("extent must be complete or uncertain")
        if not isinstance(r.get("label"), str) or not r["label"].strip():
            raise ValueError("A short visual label is required")
    return rows


def discovery_valid(answer, size):
    if not isinstance(answer, dict) or answer.get("frame") != "GLOBAL":
        raise ValueError("Explicit frame GLOBAL required; never normalize or reinterpret coordinates")
    regions_valid(answer.get("regions"), size)
    return answer


def common(size):
    return {"global_size_px": list(size),
            "coordinate_contract": "所有输出bbox使用GLOBAL实际输入图的绝对像素中心坐标；原点左上，x向右，y向下，范围[0,W-1]×[0,H-1]。LOCAL仅放大观察，不输出LOCAL坐标，不做缩放/平移计算；后台负责换算回原图。",
            "rules": RULES}


def discovery_prompt(size):
    return {**common(size), "task": "扫描GLOBAL找全可见箭头/线条，并安排适合恢复的区域。孤立零星箭头分别定位；规则线束、共享分支和可控局部网络可整体恢复。只在确实降低恢复难度时拆分，不把流程串联误当共享主干。",
            "output_schema": {"frame": "GLOBAL", "regions": [REGION]}}


def local_prompt(size, bounds):
    return {**common(size), "task": "独立局部发现与补漏：LOCAL中只要能看见实际箭头/线条，就结合GLOBAL确认可见范围和适合恢复的区域。仅框相交但没有实际笔画出现在LOCAL的对象不用列举。",
            "local_region_in_global_px": bounds,
            "instructions": [
                "本请求独立看原图，不继承其他视图的框和分组结论。不同视图的重复观察后续统一融合，不为了避免重复而漏掉可见箭头。",
                "LOCAL截断一条连接时，在GLOBAL找回同一条连接的可见范围；到该箭头的实际终点停止，不沿流程继续包入后续节点间的其他箭头。原图被遮挡则保留断开，不清楚就extent=uncertain。",
                "局部看到一些重复单元，不代表要把整列、整行或整个阵列扩成一组。只有满足共同恢复条件的局部group才补全该组范围；不要额外列举与LOCAL无关的目标，也不要保留整组又输出其重复子任务。"],
            "output_schema": {"frame": "GLOBAL", "regions": [REGION]}}


def review_prompt(size, candidates, allow_inspect=True, focused=False):
    return {**common(size), "task": "对照原图校正bbox和恢复区域，并补漏。去掉同一对象的重复观察，与把多条相关线安排到同一个恢复区域，是两种不同操作；都允许。检查过细拆分及过大合组，保留孤立箭头，避免重复恢复交错网络。一次给出修改，不讨论物理身份/父子责任。",
            "candidates": candidates,
            "instructions": [
                "keep_ids原样保留；replacements用source_ids引用被替换观察，给完整新regions。拆分/合并在一次replacement里完成，旧框不再作为额外恢复任务。",
                "每个候选ID恰好属于keep_ids、一次replacement.source_ids或一次discard.ids。没有对应旧观察的遗漏使用source_ids=[]追加。不能静默丢掉候选。",
                "旧group/single标签均可更改。相关分支、共享主干、扇形和信息量可控的小网络可合组；不强制逐线。孤立箭头不机械合并，过大的复杂网络拆成可处理局部。合组不把不同箭头视为同一条，也不在交叉处创造连接。",
                "同一交错区域不要保留整体任务又保留内部残段；按视觉选择整体或几个可处理的小组。空间相交本身不能作为删除理由。",
                ("只处理给定候选附近，允许补附近漏线，LOCAL是新证据；坐标仍全部用GLOBAL。" if focused else "BOXES是候选定位辅助，颜色/框边/编号不是原图；干净GLOBAL是判读依据。"),
                ("需要更清晰局部证据才输出inspect；每项给GLOBAL裁剪范围、需要核对的候选原ID和一句问题。裁剪应含完整相关线条，不自动扩整图。其他候选可先直接决定。" if allow_inspect else "本次不再请求新窗口。不确定区域保留并标extent=uncertain；不要因轮次结束而删除。"),
                "inspect.refs中的候选必须暂时keep；同一候选最多出现在一个窗口。窗口是观察请求，不会直接变成绘制框。"
            ],
            "output_schema": {"frame": "GLOBAL", "keep_ids": ["C1"],
                "replacements": [{"source_ids": ["C2", "C3"], "regions": [REGION],
                                  "reason": "简短可见依据"}],
                "discard": [{"ids": ["C4"], "reason": "明确非范围依据"}],
                "inspect": [{"refs": ["C1"], "bbox": [0, 0, 10, 10], "question": "具体需放大核对的问题"}] if allow_inspect else []}}


def review_valid(answer, size, candidates, allow_inspect=True):
    if not isinstance(answer, dict) or answer.get("frame") != "GLOBAL":
        raise ValueError("Explicit GLOBAL coordinate frame required")
    for name in ("keep_ids", "replacements", "discard", "inspect"):
        if not isinstance(answer.get(name), list):
            raise ValueError(name + " array required")
    expected = {r["id"] for r in candidates}
    assigned = []
    if any(not isinstance(x, str) for x in answer["keep_ids"]):
        raise ValueError("keep_ids must contain string IDs")
    assigned.extend(answer["keep_ids"])
    for patch in answer["replacements"]:
        refs = patch.get("source_ids")
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise ValueError("replacement source_ids must be an array of IDs")
        assigned.extend(refs)
        rows = regions_valid(patch.get("regions"), size)
        if not rows:
            raise ValueError("Use discard with a reason for removal; replacement cannot be empty")
        if not isinstance(patch.get("reason"), str) or not patch["reason"].strip():
            raise ValueError("Replacement needs one short reason")
    for d in answer["discard"]:
        if not isinstance(d.get("ids"), list) or not d["ids"] or any(not isinstance(x, str) for x in d["ids"]):
            raise ValueError("discard ids required")
        assigned.extend(d["ids"])
        if not isinstance(d.get("reason"), str) or not d["reason"].strip():
            raise ValueError("Explicit visual reason required for discard")
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("Every candidate needs exactly one disposition. Missing=" +
                         str(sorted(expected-set(assigned))) + ", unknown=" + str(sorted(set(assigned)-expected)))
    if not allow_inspect and answer["inspect"]:
        raise ValueError("This request cannot schedule more inspections")
    inspect_ids = set()
    for q in answer["inspect"]:
        number_box(q.get("bbox"), size)
        refs = q.get("refs")
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs) or not set(refs) <= set(answer["keep_ids"]):
            raise ValueError("inspect refs must reference retained candidates")
        if len(refs) != len(set(refs)) or inspect_ids.intersection(refs):
            raise ValueError("An observation can be inspected in only one parallel request")
        inspect_ids.update(refs)
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError("Inspection needs a visual question")
    return answer


def apply_review(candidates, answer, namespace):
    """A replacement consumes its old observations atomically, including split parents."""
    index = {r["id"]: r for r in candidates}
    result = [copy.deepcopy(index[k]) for k in answer["keep_ids"]]
    for i, patch in enumerate(answer["replacements"], 1):
        sources = sorted({s for k in patch["source_ids"]
                          for s in index[k].get("observation_ids", [k])})
        for j, r in enumerate(patch["regions"], 1):
            result.append({**copy.deepcopy(r), "id": f"{namespace}_{i}_{j}",
                           "raw_model_id": r["id"],
                           "observation_ids": sources,
                           "model_decision": patch["reason"]})
    return result


def iou(a, b):
    cross = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return cross / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-cross)


def merge_near_identical(observations, threshold=.95):
    """Only high-IoU compression; complete-link prevents transitive giant boxes.

    This combines observations of a region, not physical arrows. The model can
    split it again. No containment, width, mask, density, or coverage filters.
    """
    groups = []
    for r in observations:
        match = next((g for g in groups if all(r["render_type"] == a["render_type"] and
                     iou(r["bbox"], a["bbox"]) >= threshold for a in g)), None)
        if match is None:
            groups.append([r])
        else:
            match.append(r)
    result = []
    for i, g in enumerate(groups, 1):
        b = [min(r["bbox"][k] for r in g) if k < 2 else max(r["bbox"][k] for r in g) for k in range(4)]
        result.append({**copy.deepcopy(g[0]), "id": f"C{i}", "bbox": b,
                       "observation_ids": [r["id"] for r in g],
                       "extent": "uncertain" if any(r["extent"] == "uncertain" for r in g) else "complete",
                       "label": " / ".join(dict.fromkeys(r["label"] for r in g)),
                       "merged_observations": len(g)})
    return result
