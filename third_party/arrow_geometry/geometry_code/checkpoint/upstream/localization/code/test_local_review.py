import copy

import pytest
import contracts as C
from frames import global_frame
from local_review import batches, change_frame, frame_for, global_answer, local_rows, pair_key, unresolved_pairs, validate


def region(key, box):
    return {'id': key, 'bbox': box, 'render_type': 'linework', 'grouping': 'single', 'label': key, 'extent': 'complete'}


@pytest.mark.parametrize('native_size,box', [
    ((1080, 720), [400, 300, 650, 800]),
    ((721, 1081), [0, 0, 400, 700]),
    ((1080, 720), [1700, 1000, 2047, 1364]),
])
def test_actual_crop_coordinates_round_trip_without_model_conversion(native_size, box):
    view = global_frame(native_size)
    rows = [region('C1', box)]
    frame = frame_for(rows, view, 1.25)
    local, audit = local_rows(rows, view, frame)
    exact = audit[0]['local_exact']
    assert change_frame(exact, frame, view) == pytest.approx(box, abs=1e-9)
    answer = {'frame': 'LOCAL', 'decision': 'edit', 'keep_ids': [], 'discard': [], 'inspect': [],
              'replacements': [{'source_ids': ['C1'], 'regions': local, 'reason': 'same visible line'}]}
    validate(answer, frame.input_size, local)
    mapped, after = global_answer(answer, frame, view)
    C.review_valid(mapped, view.input_size, rows, False)
    assert len(after) == 1
    assert after[0]['raw_local_bbox'] == local[0]['bbox']


def test_request_batches_do_not_merge_or_delete_candidate_geometry():
    rows = [region('large', [0, 0, 100, 100])] + [region(str(i), [i * 5 + 1, 10, i * 5 + 4, 20]) for i in range(15)]
    original = copy.deepcopy(rows)
    groups = batches(rows, 6)
    assigned = [r['id'] for group in groups for r in group]
    assert sorted(assigned) == sorted(r['id'] for r in rows)
    assert max(map(len, groups)) <= 6
    assert rows == original


def test_cross_batch_overlap_stays_pending_until_both_seen_together():
    rows = [region(str(i), [i, 0, 100 + i, 100]) for i in range(8)]
    groups = batches(rows, 4)
    checked = {pair_key(a, b) for group in groups for i, a in enumerate(group) for b in group[i + 1:]}
    assert unresolved_pairs(rows, checked)
    boundary = batches(rows, 4, checked)
    assert boundary
    ids = [r['id'] for group in boundary for r in group]
    assert len(ids) == len(set(ids))
    edited = copy.deepcopy(rows[0]); edited['bbox'][2] += 1
    assert pair_key(edited, rows[1]) != pair_key(rows[0], rows[1])


def test_local_protocol_rejects_wrong_frame_and_never_commits_expansion_edits():
    assert 'GLOBAL image pixels only' not in C.SYSTEM
    rows = [region('C1', [10, 10, 50, 50])]
    answer = {'frame': 'LOCAL', 'decision': 'edit', 'keep_ids': ['C1'], 'replacements': [], 'discard': [], 'inspect': []}
    validate(answer, (100, 100), rows)
    with pytest.raises(ValueError, match='LOCAL'):
        validate({**answer, 'frame': 'GLOBAL'}, (100, 100), rows)
    mixed = {**answer, 'decision': 'expand', 'reason': 'tail clipped',
             'replacements': [{'source_ids': ['C1'], 'regions': [{'bbox': [-9, 0, 999, 999]}]}]}
    original = copy.deepcopy(mixed)
    assert validate(mixed, (100, 100), rows) == {'frame': 'LOCAL', 'decision': 'expand', 'reason': 'tail clipped'}
    assert mixed == original  # Raw trace remains intact; edits are ignored, not repaired.
    with pytest.raises(ValueError, match='reason'):
        validate({'frame': 'LOCAL', 'decision': 'expand', 'reason': ''}, (100, 100), rows)
    assert validate({'frame': 'LOCAL', 'decision': 'expand', 'reason': 'tail clipped'}, (100, 100), rows)


def test_atomic_split_preserves_other_candidate_and_removes_old_group():
    rows = [region('C1', [10, 10, 90, 90]), region('C2', [5, 2, 8, 5])]
    answer = {'frame': 'GLOBAL', 'keep_ids': ['C2'], 'discard': [], 'inspect': [],
              'replacements': [{'source_ids': ['C1'], 'regions': [region('a', [10, 10, 20, 20]), region('b', [80, 80, 90, 90])], 'reason': 'independent arrows'}]}
    C.review_valid(answer, (100, 100), rows, False)
    result = C.apply_review(rows, answer, 'batch')
    assert result[0] == rows[1]
    assert 'C1' not in [r['id'] for r in result]
    assert len(result) == 3
