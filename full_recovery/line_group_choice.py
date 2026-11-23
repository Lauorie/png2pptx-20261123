"""Choose a complete competing implementation once, before SOURCE-ID matching.

Geometric nomination is fallible. Unresolved groups retain their union and stay
explicitly unresolved; another existence review cannot override that abstention.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BATCH_SIZE = 6


def eligible(rows, nominations=()):
    # Whole-branch choice is useful for one route versus its complementary
    # pieces. A transitive group containing duplicates within a branch needs
    # an independently editable subset, not all-or-nothing provenance choice.
    from collections import Counter
    counts=Counter(r['layer'] for r in rows)
    layers={r['id']:r['layer'] for r in rows if 'id' in r}
    return (len(counts)==2 and min(counts.values())==1 and
            not any(layers.get(p['a'])==layers.get(p['b'])
                    for p in nominations if p.get('a') in layers and p.get('b') in layers))


def aligned_view(clean, natives, rows, folder):
    """The same display transform for SOURCE and both complete native glyphs."""
    import cv2
    from shape_integration.common import save
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    shape = next(r for r in rows if r['layer'] == 'shape')
    axis = np.asarray(shape['points'][-1] - shape['points'][0], float)
    if np.linalg.norm(axis) < 1:
        return None
    axis /= np.linalg.norm(axis)
    basis = np.stack([axis, [-axis[1], axis[0]]])
    points = np.concatenate([r['points'] for r in rows] +
                            [np.asarray(b) for r in rows for b in r.get('attachment_bounds', [])])
    projected = points @ basis.T
    lo, hi = projected.min(0) - 18, projected.max(0) + 18
    scale = min(8., 1024 / max(hi - lo))
    size = np.maximum(2, np.ceil((hi - lo) * scale).astype(int))
    forward = np.c_[scale * basis, -scale * lo]
    inverse = cv2.invertAffineTransform(forward)
    mapped = points @ forward[:, :2].T + forward[:, 2]
    if (mapped < 0).any() or (mapped >= size).any():
        raise ValueError('Complete native candidate exceeds the evidence viewport')
    w, h = map(int, size)
    sheet = Image.new('RGB', (w, 3 * (h + 24)), '#edf0f3')
    draw = ImageDraw.Draw(sheet)
    for index, im in enumerate([clean, *natives]):
        if im.size != clean.size:
            raise ValueError('SOURCE and native evidence grids differ')
        pixels = cv2.warpAffine(np.asarray(im.convert('RGB')), forward, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
        y = index * (h + 24)
        draw.text((8, y + 5), 'SOURCE' if index == 0 else f'B{index}; complete native implementation', fill='black')
        sheet.paste(Image.fromarray(pixels), (0, y + 24))
    path = folder / 'aligned_candidates.png'; sheet.save(path)
    save(folder / 'display_transform.json', {'source_to_view': forward.tolist(),
         'view_to_source': inverse.tolist(), 'complete_candidates_retained': True,
         'native_geometry_changed': False})
    return path


def validate(raw, prepared):
    from .quality_choice import validate_selection
    # Some providers return the supplied output envelope verbatim. This is
    # only transport normalization; all group/branch constraints still apply.
    if (isinstance(raw, dict) and 'groups' not in raw
            and isinstance(raw.get('output'), dict) and 'groups' in raw['output']):
        raw = raw['output']
    if isinstance(raw, list):
        # Equivalent decision envelope; validate every ID and full branch below.
        raw = {'groups': raw}
    if not isinstance(raw, dict) or not isinstance(raw.get('groups'), list):
        raise ValueError('Return the supplied groups exactly once')
    entries = raw['groups']
    ids = [r.get('id') for r in entries if isinstance(r, dict)]
    if len(ids) != len(entries) or any(not isinstance(k, str) for k in ids) or len(set(ids)) != len(ids) or set(ids) != set(prepared):
        raise ValueError('Every supplied group needs one unique decision')
    result = []
    for row in entries:
        current = prepared[row['id']]
        status, branch, reason = row.get('status'), row.get('keep_branch'), row.get('reason')
        # Some providers quote JSON null. Normalize only an abstention:
        # this retains every candidate and cannot grant deletion authority.
        if status in ('independent', 'unresolved') and branch == 'null':
            branch = None
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Give concrete SOURCE evidence for this group')
        if status == 'selected' and branch in current['implementations']:
            kept = current['implementations'][branch]
            validate_selection({'keep_ids': kept, 'reason': reason}, current['aliases'],
                               required=current['required'], branch_by_id=current['branches'])
        elif status in ('independent', 'unresolved') and branch is None:
            # An assertion of independence is unresolved ownership, never a
            # completed duplicate-removal decision. Retain the union explicitly.
            status = 'unresolved'
            kept = list(current['aliases'])
        else:
            raise ValueError('Choose one complete branch, or independent/unresolved with null keep_branch')
        if 'keep_ids' in row and row['keep_ids'] != kept:
            raise ValueError('A selected group retains the complete branch, never a subset')
        result.append({'id': row['id'], 'status': status, 'keep_branch': branch,
                       'keep_ids': kept, 'reason': reason})
    return {'groups': result}


def validate_batch(raw, prepared):
    """Keep independently valid groups; failed rows have no deletion authority."""
    from collections import Counter
    if isinstance(raw, dict) and 'groups' not in raw and isinstance(raw.get('output'), dict):
        raw = raw['output']
    if isinstance(raw, list):
        raw = {'groups': raw}
    entries = raw.get('groups') if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise ValueError('Return the supplied groups exactly once')
    counts = Counter(row.get('id') for row in entries if isinstance(row, dict)
                     and isinstance(row.get('id'), str))
    by_id = {row['id']: row for row in entries if isinstance(row, dict) and row.get('id') in prepared}
    groups, errors = [], list(raw.get('failed_groups', []))
    for ident, current in prepared.items():
        try:
            if counts[ident] != 1:
                raise ValueError('Missing or repeated group ID')
            checked = validate({'groups': [by_id[ident]]}, {ident: current})['groups'][0]
        except ValueError as error:
            errors.append({'id': ident, 'error': str(error)})
            checked = {'id': ident, 'status': 'unresolved', 'keep_branch': None,
                       'keep_ids': list(current['aliases']),
                       'reason': 'This group has no valid completed decision; retain every native candidate.'}
        groups.append(checked)
    return {'groups': groups, 'failed_groups': errors}


def compare(groups, table, clean, native, evidence, contexts, folder, runtime):
    from shape_integration.common import save
    from .quality_choice import candidate_branch_labels
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    prepared, requests = {}, []
    for batch, nodes in groups:
        ident = batch[0]['id']; rows = [table[n] for n in nodes]
        aliases = {f'C{i+1}': n for i, n in enumerate(nodes)}
        labels = candidate_branch_labels(rows)
        branches = {alias: labels[table[n]['layer']] for alias, n in aliases.items()}
        implementations = {label: [a for a in aliases if branches[a] == label] for label in labels.values()}
        required = [a for a, n in aliases.items() if table[n].get('immutable_border')]
        context_ids = {v['native_key']: k for k, v in contexts[ident].items()}
        images, regions = evidence(batch, nodes)
        aligned = aligned_view(clean, [native[(ident, layer)] for layer in labels], rows, folder / ident)
        if aligned is not None:
            images.append((ident + ' SOURCE and full native candidates; one shared transform', aligned))
        prepared[ident] = {'aliases': aliases, 'branches': branches, 'implementations': implementations,
                           'required': required, 'nodes': nodes, 'pairs': [p['id'] for p in batch], 'images': images}
        requests.append({'id': ident, 'branch_implementations': implementations, 'required': required,
                         'allowed_keep_branches': [b for b, ids in implementations.items() if set(required) <= set(ids)],
                         'source_crop_box_px': list(map(int, regions[0])),
                         'context_objects': {k: {'bbox_2d': v['bbox_2d']} for k, v in contexts[ident].items()},
                         'candidate_geometry': {a: {'source_points': table[n]['points'][
                             np.linspace(0, len(table[n]['points']) - 1, 5).astype(int)].round(2).tolist(),
                             'endpoint_context_ids': [context_ids[k] for k in table[n].get('endpoint_nodes', ())
                                                      if k in context_ids]}
                             for a, n in aliases.items()}})

    batches = []
    for entry in requests:
        # A complete group supplies at most two evidence sheets regardless of
        # its C count. Pack groups, rather than reintroducing per-line limits.
        if not batches or len(batches[-1]) >= BATCH_SIZE:
            batches.append([])
        batches[-1].append(entry)

    def one(item):
        index, batch = item
        stage = runtime.for_stage(reasoning_effort='low', request_timeout_seconds=600,
                                  shape_schema_retries=0)
        try:
            response = stage.call(
            folder / f'batch_{index:03d}', 'review',
            '对照SOURCE，为每组重复线路选择更贴近原图的一侧。只返回JSON。',
            {'task': '每组一次比较SOURCE、B1和B2，选择本组保留的原生实现。',
             'rules': [
                 'SOURCE、B1和B2使用相同裁剪与坐标变换。C是实际原生候选；B面板展示该分支全部C共同绘制的结果。context_objects及共同节点、填充面提供只读上下文。',
                 '在SOURCE追踪本组各候选对应的完整路径，包括可见端点、转折和接入位置。candidate_geometry和endpoint_context_ids是拟合提示；以SOURCE中的实际路径确定同一对象，恢复偏移和粗细差异不代表新的源对象。',
                 '两侧对应同一组源路径时选selected：先比较路径完整性、端点和头部，再比较线型、颜色和粗细；两侧都有误差时仍选更贴近SOURCE的一侧，叠加不能修复这些误差。',
                 'keep_branch取allowed_keep_branches中的一个值，保留其全部C并删除另一侧全部C。required是需要保留的原生轮廓。',
                 '两侧分别包含独立源路径或互补区段、删除任一侧会丢失独有内容时，保留两侧并返回unresolved；源路径身份无法确认时也返回unresolved。说明具体路径或区段。'],
             'groups': batch,
             'output': {'groups': [{'id': '输入组ID', 'status': 'selected|unresolved',
                                    'keep_branch': 'B1 或 B2；unresolved为null', 'reason': '具体原图证据'}]}},
            [(ident + ': ' + label, path) for entry in batch for ident in [entry['id']]
             for label, path in prepared[ident]['images']],
                lambda raw: validate_batch(raw, {r['id']: prepared[r['id']] for r in batch}), 16384)
            failure = ({'groups': response['failed_groups'], 'error_type': 'invalid_group_rows'}
                       if response.get('failed_groups') else None)
            return response['groups'], failure
        except (ValueError, RuntimeError) as error:
            # A failed decision has no deletion authority. Complete groups in
            # other batches keep their successful result; do not rerun them or
            # add another existence review for this group.
            failure = {'groups': [entry['id'] for entry in batch],
                       'error_type': type(error).__name__, 'error': str(error)}
            retained = [{'id': entry['id'], 'status': 'unresolved', 'keep_branch': None,
                         'keep_ids': list(prepared[entry['id']]['aliases']),
                         'reason': 'Comparison did not complete; retain every native candidate.'}
                        for entry in batch]
            return retained, failure

    workers = max(1, min(2, int(getattr(runtime, 'config', {}).get('mapping_workers', 2))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(one, enumerate(batches)))
    choices, unresolved, failures = {}, [], []
    for response, failure in responses:
        if failure is not None:
            failures.append(failure)
        for row in response:
            current = prepared[row['id']]
            if row['status'] == 'unresolved':
                unresolved.append(row)
            choices[row['id']] = {'keep_ids': [current['aliases'][a] for a in row['keep_ids']],
                'candidate_ids': current['nodes'], 'pairs': current['pairs'], 'reason': row['reason'],
                'selection_unit': 'minimal_competitor_group' if row['status'] == 'selected' else 'unresolved_competitor_group',
                'source_relation': row['status'], 'whole_group_branch': row['keep_branch']}
    calls = len(batches)
    save(folder / 'summary.json', {'groups': len(prepared), 'model_calls': calls,
         'selected': sum(c['source_relation'] == 'selected' for c in choices.values()),
         'unresolved': unresolved, 'unresolved_action': 'retain_union_no_additional_review',
         'source_inventory_requests': 0, 'failed_batches': failures,
         'ownership_complete': not unresolved})
    return choices, calls
