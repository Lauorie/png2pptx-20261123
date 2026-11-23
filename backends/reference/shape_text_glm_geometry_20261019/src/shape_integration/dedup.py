"""Select shape/image conflicts by IoU, then optionally review them in batches."""
from __future__ import annotations
import copy
import math
from typing import Literal, Protocol, TypedDict


class ReviewDecision(TypedDict):
    shape_id: str
    decision: Literal['delete', 'keep', 'defer']
    image_id: str | None
    reason: str


class ConflictReviewer(Protocol):
    def __call__(self, requests: list[dict]) -> list[ReviewDecision]:
        """Judge only rule-selected candidates together; do not classify all objects."""
        ...


def validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Expected four bbox numbers')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError('BBox numbers must be finite')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('BBox must have positive area')


def intersection_over_union(a, b):
    """Continuous axis-aligned rectangle IoU, with no rounding, padding, or +1."""
    validate_box(a); validate_box(b)
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union


def filter_candidates(shapes, images, threshold=.85, *, mode='rule_only', reviewer: ConflictReviewer | None = None,
                      source=None, review_batch_size=32):
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one')
    if mode not in ('rule_only', 'model_review'):
        raise ValueError('Unknown decision mode')
    if mode == 'rule_only' and reviewer is not None:
        raise ValueError('Rule-only mode never calls a reviewer')
    if mode == 'model_review' and reviewer is None:
        raise ValueError('Model-review mode requires a batch reviewer')
    if type(review_batch_size) is not int or review_batch_size < 1:
        raise ValueError('Review batch size must be a positive integer')
    for rows in (shapes, images):
        ids = [o['id'] for o in rows]
        if len(set(ids)) != len(ids): raise ValueError('Duplicate IDs within inventory')
        for o in rows: validate_box(o['bbox_2d'])
    pairs, decisions, removed, kept, packets = [], [], [], [], []
    reviewer_calls = 0
    for shape in shapes:
        matches = []
        for asset in images:
            score = intersection_over_union(shape['bbox_2d'], asset['bbox_2d'])
            row = {'shape_id': shape['id'], 'image_id': asset['id'], 'iou': score,
                   'above_threshold': score > threshold}
            pairs.append(row)
            matches.append((score, asset))
        matches.sort(key=lambda item: (-item[0], item[1]['id']))
        maximum, best = matches[0] if matches else (0., None)
        proposal = maximum > threshold
        packet = {'schema_version': 'shape_image_conflict_review.2', 'source': source,
                  'coordinate_system': 'original_image_pixel_centers', 'shape': copy.deepcopy(shape),
                  'overlapping_images': [{'asset': copy.deepcopy(asset), 'iou': score}
                                         for score, asset in matches if score > threshold],
                  'rule': {'metric': 'IoU', 'comparison': '>', 'threshold': threshold,
                           'selected_for_review': proposal},
                  'response_schema': {'decision': 'delete|keep|defer', 'reason': 'visible evidence'},
                  'execution_status': 'not_called'}
        if proposal: packets.append(packet)
        action = 'delete' if proposal and mode == 'rule_only' else 'keep'
        reason = 'max_image_iou_above_threshold' if proposal else 'no_image_iou_above_threshold'
        record = {'shape_id': shape['id'], 'shape_label': shape.get('label'), 'max_iou': maximum,
                  'best_image_id': best['id'] if best else None,
                  'best_image_target': best.get('target') if best else None,
                  'matching_image_ids': [asset['id'] for score, asset in matches if score > threshold],
                  'rule_selected': proposal, 'rule_proposed_delete': proposal and mode == 'rule_only',
                  'decision': action, 'reason': reason, 'decision_source': 'rule',
                  'shape_bbox_original': list(shape['bbox_2d']),
                  'best_image_bbox_original': list(best['bbox_2d']) if best else None}
        decisions.append(record)
    by_id = {row['shape_id']: row for row in decisions}
    if mode == 'model_review':
        for start in range(0, len(packets), review_batch_size):
            chunk = packets[start:start+review_batch_size]
            answers = reviewer(copy.deepcopy(chunk)); reviewer_calls += 1
            expected = {p['shape']['id']: p for p in chunk}
            if not isinstance(answers, list) or any(not isinstance(a, dict) for a in answers):
                raise ValueError('Reviewer must return one decision per selected shape')
            ids = [a.get('shape_id') for a in answers]
            if any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or set(ids) != set(expected):
                raise ValueError('Reviewer omitted, repeated or invented a selected shape ID')
            for answer in answers:
                packet = expected[answer['shape_id']]
                allowed = {m['asset']['id'] for m in packet['overlapping_images']}
                action = answer.get('decision'); image_id = answer.get('image_id')
                if (action not in ('delete', 'keep', 'defer') or not isinstance(answer.get('reason'), str)
                        or not answer['reason'].strip() or (action == 'delete' and image_id not in allowed)
                        or (action != 'delete' and image_id is not None)):
                    raise ValueError('Invalid reviewer decision or matched image reference')
                packet['execution_status'] = 'reviewed'; packet['response'] = answer
                by_id[answer['shape_id']].update(decision=action, reason=answer['reason'],
                    decision_source='reviewer', matched_image_id=image_id)
    for shape in shapes:
        (removed if by_id[shape['id']]['decision'] == 'delete' else kept).append(copy.deepcopy(shape))
    return {'kept': kept, 'removed': removed, 'decisions': decisions, 'pairs': pairs,
            'review_requests': packets, 'reviewer_calls': reviewer_calls,
            'review_candidates': len(packets),
            'deferred_ids': [r['shape_id'] for r in decisions if r['decision'] == 'defer'],
            'threshold': threshold, 'mode': mode}
