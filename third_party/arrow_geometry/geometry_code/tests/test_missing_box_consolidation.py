import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from missing_box_consolidation import consolidate, nonzero_bbox, path_bounds


def target(name, box, kind='linework'):
    return dict(id=name, bbox_native=box, render_type=kind, grouping='single',
                source_candidate='same_review', locator='one diagonal')


def test_crossing_targets_merge_scope_without_inheriting_one_description():
    original = [target('a', [0, 0, 100, 100]), target('b', [2, 0, 102, 100])]
    before = copy.deepcopy(original)
    kept, audit = consolidate(original, [])
    assert len(kept) == 1 and kept[0]['bbox_native'] == [0, 0, 102, 100]
    assert kept[0]['locator'] == '' and kept[0]['members'] == before
    assert kept[0]['member_ids'] == ['a', 'b'] and original == before
    assert consolidate(list(reversed(original)), []) == (kept, audit)


def test_no_containment_deletion_no_transitive_chain_and_separate_types():
    rows = [target('a', [0, 0, 100, 10]), target('b', [15, 0, 115, 10]),
            target('c', [30, 0, 130, 10]), target('d', [50, 3, 55, 6]),
            target('e', [0, 0, 100, 10], 'block_arrow')]
    kept, _ = consolidate(rows, [])
    assert len(kept) == 4
    assert sorted(m for t in kept for m in t['member_ids']) == ['a', 'b', 'c', 'd', 'e']
    assert next(t for t in kept if t['id'] == 'd')['locator'] == 'one diagonal'


def test_existing_filter_requires_all_original_members_to_match_one_path():
    old = [dict(id='old', kind='straight', start=[0, 0], end=[100, 100])]
    before = copy.deepcopy(old)
    rows = [target('a', [0, 0, 100, 100]), target('b', [15, 0, 115, 100]),
            target('small', [20, 20, 30, 30]), target('shape', [0, 0, 100, 100], 'block_arrow')]
    kept, audit = consolidate(rows, old)
    assert {t['id'] for t in kept} == {'small', 'shape'} and audit['filtered_count'] == 1
    rows[1]['bbox_native'] = [30, 0, 130, 100]
    kept, audit = consolidate(rows, old)
    assert 'b' in {t['id'] for t in kept} and old == before
    kept, _ = consolidate(rows, old, filter_existing=False)
    assert 'a' in {t['id'] for t in kept}


def test_vector_bounds_include_curve_extrema_and_minimum_native_extent():
    assert path_bounds(dict(beziers=[[[0, 0], [0, 10], [10, 10], [10, 0]]])) == [0, 0, 10, 7.5]
    assert nonzero_bbox([0, 2, 10, 2]) == [0, 1.5, 10, 2.5]
    assert path_bounds(dict(start=[0, 0], via=[[30, -5]], end=[10, 10])) == [0, -5, 30, 10]
