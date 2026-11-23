import copy
import pytest
import contracts as C
from local_review import validate
from review_type_policy import inherit


def row(key, kind='linework', box=None):
    return dict(id=key, bbox=box or [10, 10, 30, 30], render_type=kind,
                grouping='single', label='visible target', extent='complete')


def answer(refs, regions, keep=None):
    return dict(frame='LOCAL', decision='edit', keep_ids=keep or [], discard=[], inspect=[],
                replacements=[dict(source_ids=refs, regions=regions, reason='adjust visible extent')])


def test_type_only_change_preserves_raw_answer_and_all_geometric_fields():
    source = [row('C1')]
    child = row('R1', 'block_arrow', [11.25, 9.5, 32.75, 38])
    raw = answer(['C1'], [child]); saved = copy.deepcopy(raw)
    assert validate(raw, (100, 100), source, True) == saved
    typed, audit = inherit(raw, source)
    assert raw == saved
    assert typed['replacements'][0]['regions'][0] == {**child, 'render_type': 'linework'}
    assert audit['inherited'][0]['model_render_type'] == 'block_arrow'
    assert inherit(raw, source) == (typed, audit)  # Identical when replaying the cached raw result.


def test_split_and_same_type_merge_preserve_true_block_representation():
    source = [row('C1', 'block_arrow'), row('C2', 'block_arrow')]
    children = [row('R1'), row('R2')]
    for r in children:
        r.pop('render_type')
    raw = answer(['C1', 'C2'], children)
    validate(raw, (100, 100), source, True)
    typed, _ = inherit(raw, source)
    assert [r['render_type'] for r in typed['replacements'][0]['regions']] == ['block_arrow', 'block_arrow']
    rebuilt = C.apply_review(source, {**typed, 'frame': 'GLOBAL'}, 'review')
    assert len(rebuilt) == 2 and all(r['observation_ids'] == ['C1', 'C2'] for r in rebuilt)


def test_mixed_patch_preserves_inputs_without_discarding_other_valid_edit():
    source = [row('C1'), row('C2', 'block_arrow'), row('C3')]
    raw = answer(['C1', 'C2'], [row('mixed')])
    raw['replacements'].append(dict(source_ids=['C3'], regions=[row('new3', box=[20, 20, 40, 40])], reason='tighten'))
    validate(raw, (100, 100), source, True)
    typed, audit = inherit(raw, source)
    actual = C.apply_review(source, {**typed, 'frame': 'GLOBAL'}, 'review')
    assert actual[:2] == source[:2]
    assert actual[2]['bbox'] == [20, 20, 40, 40]
    assert len(audit['blocked_mixed_type_patches']) == 1


def test_new_candidates_require_their_own_type_and_do_not_inherit_a_neighbor():
    source = [row('C1')]
    raw = answer([], [row('new', 'block_arrow')], keep=['C1'])
    typed, audit = inherit(raw, source)
    validate(raw, (100, 100), source, True)
    assert typed['replacements'][0]['regions'][0]['render_type'] == 'block_arrow'
    assert audit['new_region_ids'] == ['new']
    raw['replacements'][0]['regions'][0].pop('render_type')
    with pytest.raises(ValueError, match='linework'):
        validate(raw, (100, 100), source, True)


def test_inheritance_cannot_hide_missing_unknown_or_duplicated_source_ids():
    source = [row('C1'), row('C2', 'block_arrow')]
    for refs, keep in [(['C1', 'unknown'], []), (['C1', 'C2'], ['C1']), (['C1'], [])]:
        with pytest.raises(ValueError, match='disposition'):
            validate(answer(refs, [row('new')], keep), (100, 100), source, True)


def test_expand_still_ignores_speculative_edits():
    raw = dict(frame='LOCAL', decision='expand', reason='target continues outside crop',
               replacements=[dict(source_ids=['C1'], regions=[row('bad', 'block_arrow')])])
    assert validate(raw, (100, 100), [row('C1')], True) == {
        'frame': 'LOCAL', 'decision': 'expand', 'reason': 'target continues outside crop'}
