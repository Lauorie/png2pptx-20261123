"""Small crop-based model edits. Spatial scheduling never decides arrow identity."""
import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import contracts as C
import draft_prompts as Draft
from semantic_hint_contract import clean_prompt
from frames import crop_from_global, global_frame, visible_global_bounds
from presentation import compact, overlay, latest
from transport.common import write_json as write
from review_type_policy import inherit as inherit_types


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def signature(row):
    return hashlib.sha256(json.dumps(compact([row])[0], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pair_key(a, b):
    return tuple(sorted((signature(a), signature(b))))


def unresolved_pairs(rows, checked):
    return [(a['id'], b['id']) for i, a in enumerate(rows) for b in rows[i + 1:]
            if overlap(a['bbox'], b['bbox']) > 0 and pair_key(a, b) not in checked]


def batches(rows, limit, checked=None):
    """Disjoint editable IDs per wave; bbox overlap only chooses who sees whom."""
    if limit < 2:
        raise ValueError('Review batch must permit at least two candidates')
    index = {r['id']: r for r in rows}
    pending = set(index)
    pairs = None if checked is None else unresolved_pairs(rows, checked)
    result = []
    while pending:
        if pairs is None:
            seed = max(pending, key=lambda k: (area(index[k]['bbox']), k))
            neighbors = [k for k in pending - {seed} if overlap(index[k]['bbox'], index[seed]['bbox']) > 0]
        else:
            active = [(a, b) for a, b in pairs if a in pending and b in pending]
            if not active:
                break
            degrees = {k: sum(k in edge for edge in active) for k in pending}
            seed = max(pending, key=lambda k: (degrees[k], area(index[k]['bbox']), k))
            neighbors = list({b if a == seed else a for a, b in active if seed in (a, b)})
        neighbors.sort(key=lambda k: (-overlap(index[seed]['bbox'], index[k]['bbox']) /
                                     min(area(index[seed]['bbox']), area(index[k]['bbox'])), k))
        chosen = [seed] + neighbors[:limit - 1]
        result.append([index[k] for k in chosen]); pending.difference_update(chosen)
    return result


def frame_for(rows, view, context_scale, expansion=0):
    boxes = [r['bbox'] for r in rows]
    box = [min(b[k] for b in boxes) if k < 2 else max(b[k] for b in boxes) for k in range(4)]
    center = [(box[k] + box[k + 2]) / 2 for k in (0, 1)]
    # Min context is expressed in original image pixels, not resized display pixels.
    minimum = [25 * view.input_size[k] / view.global_size[k] for k in (0, 1)]
    extent = [max(box[k + 2] - box[k], minimum[k]) * context_scale * (1.6 ** expansion) for k in (0, 1)]
    window = [max(0., center[k] - extent[k] / 2) for k in (0, 1)] + [
        min(view.input_size[k] - 1., center[k] + extent[k] / 2) for k in (0, 1)]
    return crop_from_global(window, view, min_edge=1080, max_edge=1536)


def change_frame(box, source_view, target_view):
    return target_view.to_local(source_view.to_global(box[:2])) + target_view.to_local(source_view.to_global(box[2:]))


def intersect_centers(box, size):
    return [max(0., min(size[k % 2] - 1., v)) for k, v in enumerate(box)]


def local_rows(rows, view, frame):
    converted, audits = [], []
    for row in rows:
        raw = change_frame(row['bbox'], view, frame)
        shown = intersect_centers(raw, frame.input_size)
        converted.append({**row, 'bbox': shown})
        audits.append({'id': row['id'], 'global_bbox': row['bbox'], 'local_exact': raw,
                       'local_visible': shown, 'boundary_intersection': raw != shown})
    C.regions_valid(converted, frame.input_size)
    return converted, audits


def prompt(size, rows, boundary_pass=False, omit_candidate_labels=False):
    candidates = compact(rows)
    if omit_candidate_labels:
        candidates = [{k: v for k, v in row.items() if k != 'label'} for row in candidates]
    return Draft.review_prompt(size, candidates, boundary_pass)


def validate(answer, size, rows, inherit_existing_type=False):
    if not isinstance(answer, dict) or answer.get('frame') != 'LOCAL':
        raise ValueError('Explicit LOCAL pixel frame required')
    if answer.get('decision') == 'expand':
        if not isinstance(answer.get('reason'), str) or not answer['reason'].strip():
            raise ValueError('Context expansion requires a reason')
        # An expansion is a control decision, never a partial geometry commit.
        # Ignore speculative edits rather than spend another visual inference
        # repairing a reply that already told us the necessary next action.
        return {k: answer[k] for k in ('frame', 'decision', 'reason')}
    if answer.get('decision') != 'edit':
        raise ValueError('Use edit or expand')
    checked_answer = inherit_types(answer, rows)[0] if inherit_existing_type else answer
    C.review_valid({**checked_answer, 'frame': 'GLOBAL'}, size, rows, allow_inspect=False)
    # Store the unmodified model JSON in request/result.json. Inheritance is
    # applied separately below, so resume produces the same auditable decision.
    return answer


def global_answer(answer, frame, view):
    converted = copy.deepcopy(answer); converted['frame'] = 'GLOBAL'
    audits = []
    for patch in converted['replacements']:
        for row in patch['regions']:
            local = list(row['bbox']); exact = change_frame(local, frame, view)
            visible = intersect_centers(exact, view.input_size)
            row['bbox'] = visible
            audits.append({'raw_local_bbox': local, 'mapped_global_exact': exact,
                           'global_visible': visible, 'boundary_intersection': exact != visible})
    return converted, audits


def run(root, cid, rows, model, cfg, state):
    if cfg.get('adaptive_review', False):
        from scoped_review import run as adaptive_run
        return adaptive_run(root, cid, rows, model, cfg, state)
    out = root / 'cases' / cid / 'local_review'
    source = Image.open(root / 'inputs' / cid / 'source.png').convert('RGB')
    view = global_frame(source.size)
    checked, issues, history = set(), [], []
    current = copy.deepcopy(rows)
    total_calls = 0
    for round_index in range(cfg.get('local_review_rounds', 3)):
        groups = batches(current, cfg.get('local_review_batch_size', 6), None if round_index == 0 else checked)
        if not groups:
            break
        state(f'local_bbox_review_round_{round_index + 1}')
        folder = out / f'round_{round_index + 1:02d}'
        write(folder / 'plan.json', {'groups': [[r['id'] for r in g] for g in groups],
            'input_count': len(current), 'policy': 'spatial request scheduling only; all edits are model decisions'})

        def review_one(index, focus, part=''):
            base = folder / (f'B{index:03d}' + part)
            try:
                for expansion in range(cfg.get('local_review_expansions', 1) + 1):
                    target = base / f'view_{expansion:02d}'
                    target.mkdir(parents=True, exist_ok=True)
                    frame = frame_for(focus, view, cfg.get('local_review_context_scale', 1.25), expansion)
                    local, before = local_rows(focus, view, frame)
                    frame.image(source).save(target / 'clean.png')
                    overlay(target / 'clean.png', local, target / 'boxes.png')
                    write(target / 'frame.json', {'crop': frame.metadata(source.size),
                        'global_view': view.metadata(source.size), 'local_bounds_in_global': visible_global_bounds(frame, view),
                        'input_conversion': before})
                    clean_rows, hint_changes = clean_prompt(compact(local))
                    if hint_changes:
                        write(target / 'semantic_hint_input.json', {'raw_candidates': compact(local),
                            'changes': hint_changes, 'structured_coordinates_changed': False})
                    omit_labels = cfg.get('review_omit_candidate_labels', False)
                    if omit_labels:
                        write(target / 'candidate_label_input_audit.json', {
                            'omitted_input_field': 'label', 'candidate_ids': [r['id'] for r in clean_rows],
                            'backend_candidates_unchanged': True, 'output_labels_allowed': True})
                    answer = model.ask(target / 'request', prompt(frame.input_size, clean_rows, round_index > 0, omit_labels),
                        [('CLEAN: sole LOCAL output coordinates', target / 'clean.png'),
                         ('BOXES: same LOCAL frame, fallible annotations', target / 'boxes.png')],
                        lambda a: validate(a, frame.input_size, local, cfg.get('review_inherit_existing_type', False)),
                        system=Draft.REVIEW_SYSTEM)
                    if answer['decision'] == 'expand':
                        continue
                    type_issue = None
                    if cfg.get('review_inherit_existing_type', False):
                        write(target / 'accepted_model_answer.json', answer)
                        answer, type_audit = inherit_types(answer, local)
                        write(target / 'type_inheritance_audit.json', type_audit)
                        write(target / 'typed_local_answer.json', answer)
                        if type_audit['blocked_mixed_type_patches']:
                            type_issue = {'stage': 'local_bbox_type_inheritance',
                                          'reason': 'Cross-type replacement retained original candidates',
                                          'blocked_patches': type_audit['blocked_mixed_type_patches'],
                                          'retained_pending': True}
                    mapped, after = global_answer(answer, frame, view)
                    C.review_valid(mapped, view.input_size, focus, False)
                    write(target / 'mapped_answer.json', mapped)
                    write(target / 'output_conversion.json', after)
                    replacement = C.apply_review(focus, mapped, f'LR{round_index+1}B{index}{part}')
                    return {'focus': focus, 'replacement': replacement, 'folder': str(base), 'issue': type_issue}
                raise ValueError('Model still requests more context after permitted local expansion')
            except Exception as exc:
                write(base / 'failure.json', {'error': str(exc), 'focus_ids': [r['id'] for r in focus]})
                # Failed local edits are retried with fewer IDs, not handed off as the whole raw inventory.
                if len(focus) > 1 and not part:
                    mid = len(focus) // 2
                    a = review_one(index, focus[:mid], 'a'); b = review_one(index, focus[mid:], 'b')
                    return {'focus': focus, 'replacement': a['replacement'] + b['replacement'], 'folder': str(base),
                            'issue': a['issue'] or b['issue'], 'recovered_by_smaller_requests': not a['issue'] and not b['issue']}
                return {'focus': focus, 'replacement': [{**r, 'extent': 'uncertain'} for r in focus],
                        'folder': str(base), 'issue': {'stage': 'local_bbox_review', 'error': str(exc),
                            'focus_ids': [r['id'] for r in focus], 'retained_pending': True}}

        answers = {}
        with ThreadPoolExecutor(max_workers=cfg.get('local_workers', 4)) as pool:
            jobs = {pool.submit(review_one, i, g): i for i, g in enumerate(groups, 1)}
            for future in as_completed(jobs):
                answers[jobs[future]] = future.result()
                partial = copy.deepcopy(current)
                for key in sorted(answers):
                    done = answers[key]
                    consumed = {r['id'] for r in done['focus']}
                    partial = [r for r in partial if r['id'] not in consumed] + done['replacement']
                latest(root, cid, partial, f'local_review_{round_index + 1:02d}_partial')
        for index in sorted(answers):
            answer = answers[index]
            ids = {r['id'] for r in answer['focus']}
            current = [r for r in current if r['id'] not in ids] + answer['replacement']
            if answer['issue']:
                issues.append(answer['issue'])
            else:
                # A split retry did not compare the two subrequests, so leave their cross-pairs pending.
                if not answer.get('recovered_by_smaller_requests'):
                    for members in (answer['focus'], answer['replacement']):
                        checked.update(pair_key(a, b) for i, a in enumerate(members) for b in members[i + 1:])
            history.append({'round': round_index + 1, 'batch': index, 'focus_ids': sorted(ids),
                            'output_ids': [r['id'] for r in answer['replacement']], 'issue': answer['issue']})
        total_calls += len(groups)
        C.regions_valid(current, view.input_size)
        write(folder / 'result.json', {'regions': current, 'unchecked_overlaps': unresolved_pairs(current, checked)})
        latest(root, cid, current, f'local_review_{round_index + 1:02d}')
    write(out / 'audit.json', {'history': history, 'scheduled_batches': total_calls,
        'unchecked_overlaps': unresolved_pairs(current, checked), 'issues': issues,
        'all_geometry_decisions_by_model': True, 'global_correction_calls': 0})
    return current, issues
