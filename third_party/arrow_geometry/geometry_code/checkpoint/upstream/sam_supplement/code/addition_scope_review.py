"""Small scope-only review after SAM addition organization, before geometry.

Only an explicit model reject removes a new candidate. No coordinates, labels,
baseline rows, geometry, or styles are rewritten by this stage.
"""
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import ImageDraw
import experiment as E
from pipeline_frames import Crop, resized_size

SYSTEM = '仅判断一个SAM补漏新增候选是否属于待重建箭头或线条。结合完整原图判断，不修框、不补漏、不恢复几何或样式。只输出JSON。'


def validate(answer):
    if not isinstance(answer, dict) or set(answer) != {'decision', 'reason'}:
        raise ValueError('Return only decision and reason; do not return coordinates or edits')
    if answer['decision'] not in ('keep', 'reject', 'uncertain'):
        raise ValueError('decision must be keep, reject or uncertain')
    if not isinstance(answer['reason'], str) or not answer['reason'].strip():
        raise ValueError('Provide a brief visible reason')
    return answer


def make_prompt(row, global_view, local_view):
    return dict(
        task='只审核target对应的新增笔画是否属于箭头/线条重建范围，回答keep、reject或uncertain。',
        images=['GLOBAL_CLEAN：完整原图，判断目标是否是完整容器/形状的一部分。',
                'LOCAL_CLEAN：目标附近原图，检查实际笔画。',
                'LOCAL_TARGET：与LOCAL_CLEAN同图，蓝色角标及T仅用于定位目标，不是原图笔画。'],
        target=dict(id=row['id'], bbox_global=E.local_box(row['bbox'], global_view),
                    bbox_local=E.local_box(row['bbox'], local_view)),
        image_sizes=dict(global_px=list(global_view.input_size), local_px=list(local_view.input_size)),
        coordinates='输入框是对应图片的绝对像素，仅供定位；不输出坐标。局部裁剪边不代表真实笔画端点。',
        rules=[
            'keep：真实的有头/无头连线、曲线、虚线、共享分支、框内网络、独立分隔线/坐标轴/括号，以及实心或空心块状箭头。没有箭头头部不是删除理由。',
            'reject：目标只是非箭头矩形/圆角/其他闭合形状或容器的边框（含虚线框残边）、文字、纹理或空白误检。对照完整原图确认边框归属，不能把局部看似独立的长直线自动当成连接。',
            '判断蓝色角标指向的候选笔画，不是判断整个crop；旁边或穿过目标的其他真实箭头，不会使容器边框本身变成箭头。',
            '只有能明确排除整个候选时才reject；若候选混合了需要保留的真实连线和边框，或不能确认目标归属，返回uncertain并简述原因。',
            '只基于可见图像，不推导遮挡连线。不要修正范围、拆分合并、检查重复、估算样式，也不要评价几何精度。'
        ],
        output_schema={'decision': 'keep|reject|uncertain', 'reason': '一句简短可见依据'})


def evidence(folder, image, row, cfg):
    """Exact pixel-center transforms; context sizing is not a semantic decision."""
    width, height = image.size
    global_size = resized_size(image.size, min(max(image.size), cfg.get('sam_scope_global_long_edge', 2048)))
    full = Crop(image.size, (0, 0, width, height), global_size)
    b = row['bbox']
    center = [(b[k] + b[k + 2]) / 2 for k in (0, 1)]
    extent = [max(b[k + 2] - b[k], cfg.get('sam_scope_min_context_px', 96)) *
              cfg.get('sam_scope_context_scale', 2.0) for k in (0, 1)]
    bounds = tuple([max(0, math.floor(center[k] - extent[k] / 2)) for k in (0, 1)] +
                   [min(image.size[k], math.ceil(center[k] + extent[k] / 2) + 1) for k in (0, 1)])
    local = Crop(image.size, bounds, (bounds[2] - bounds[0], bounds[3] - bounds[1])).scaled(
        3, max_long_edge=cfg.get('sam_scope_local_long_edge', 1536))
    clean = local.image(image)
    E.save(full.image(image), folder / 'global_clean.png')
    E.save(clean, folder / 'local_clean.png')
    locator = clean.copy()
    E.corners(ImageDraw.Draw(locator), E.local_box(b, local), '#1679d3', 'T')
    E.save(locator, folder / 'local_target.png')
    E.write(folder / 'frames.json', dict(native_bbox=b,
        global_frame=full.metadata(image.size), local_frame=local.metadata(image.size),
        bbox_global=E.local_box(b, full), bbox_local=E.local_box(b, local),
        original_semantic_label_sent=False, output_coordinates=False))
    return make_prompt(row, full, local), [
        ('GLOBAL_CLEAN: full original image', folder / 'global_clean.png'),
        ('LOCAL_CLEAN: original local context', folder / 'local_clean.png'),
        ('LOCAL_TARGET: blue corners locate target T', folder / 'local_target.png')]


def review_one(folder, row, image, model, cfg):
    started = time.monotonic()
    E.write(folder / 'candidate.json', row)
    try:
        prompt, images = evidence(folder, image, row, cfg)
        answer = validate(model.ask(folder / 'request', prompt, images, validate, system=SYSTEM))
        result = dict(id=row['id'], **answer, state='completed')
    except Exception as exc:
        # Failed review is not permission to remove a previously reviewed candidate.
        result = dict(id=row['id'], decision='uncertain', state='failed',
                      reason='Review failed; candidate retained unchanged',
                      error_type=type(exc).__name__, error=str(exc))
    result.update(elapsed_seconds=round(time.monotonic() - started, 3),
                  trace=str(folder.resolve()), coordinates_changed=False)
    E.write(folder / 'result.json', result)
    return result


def review(root, cid, rows, baseline, image, model, cfg):
    """Shared by the live supplement pipeline and replay of completed additions."""
    folder = Path(root) / 'cases' / cid / 'addition_scope_review'
    E.append_only(baseline, rows, image.size)  # Validate no old IDs or mutated inputs.
    original = copy.deepcopy(rows)
    baseline_hash = E.rows_digest(baseline)
    E.write(folder / 'before.json', original)
    if not cfg.get('sam_addition_scope_review_enabled', True):
        audit = dict(enabled=False, before_count=len(rows), after_count=len(rows),
                     baseline_preserved=True, candidates_unchanged=True, issues=[], decisions=[])
        E.write(folder / 'audit.json', audit)
        return original, audit
    answers = {}
    with ThreadPoolExecutor(max_workers=cfg.get('sam_scope_workers', 4)) as pool:
        jobs = {pool.submit(review_one, folder / f'C{i + 1:04d}', row, image, model, cfg): i
                for i, row in enumerate(original)}
        for future in as_completed(jobs):
            answers[jobs[future]] = future.result()
            E.write(folder / 'progress.json', dict(finished=len(answers), total=len(rows), updated_at=E.now()))
    decisions = [answers[i] for i in range(len(original))]
    kept = [r for r, a in zip(original, decisions) if a['decision'] != 'reject']
    rejected = [r for r, a in zip(original, decisions) if a['decision'] == 'reject']
    unresolved = [a for a in decisions if a['decision'] == 'uncertain']
    issues = [dict(stage='sam_addition_scope_review', id=a['id'], state=a['state'],
                   reason=a['reason'], retained_unchanged=True) for a in unresolved]
    if E.rows_digest(baseline) != baseline_hash or rows != original:
        raise ValueError('Scope review mutated source candidates or immutable baseline')
    audit = dict(enabled=True, before_count=len(rows), after_count=len(kept),
        rejected_count=len(rejected), uncertain_count=len(unresolved),
        baseline_preserved=True, coordinates_changed=False,
        kept_rows_unchanged=True, issues=issues, decisions=decisions)
    E.write(folder / 'rejected.json', rejected)
    E.write(folder / 'unresolved.json', unresolved)
    E.write(folder / 'after.json', kept)
    E.write(folder / 'audit.json', audit)
    return kept, audit
