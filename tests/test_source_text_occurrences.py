from copy import deepcopy

import pytest

from full_recovery.text_inventory import apply_reply, duplicate_pairs, install
from shape_integration.texts import options


FRAME = {'box': [0, 0, 400, 200], 'scale': 1, 'offset': [0, 0],
         'source_size': [400, 200], 'size': [400, 200]}


def row(ident, text, box):
    return {'id': ident, 'text': text, 'bbox_2d': box, 'kind': 'text',
            'runs': [{'text': text, 'script': 'normal', 'color': '#000000'}],
            'font': 'Arial', 'color': '#000000', 'rotation_deg': 0,
            'bold': False, 'italic': False, 'underline': False,
            'visual_line_count': 1, 'legibility': 'clear'}


def example():
    rows = [row('A', '18 units', [10, 10, 100, 30]),
            row('B', '16 units', [10, 10, 100, 30]),
            row('C', '18 units', [10, 70, 100, 90])]
    keeper = {**deepcopy(rows[1]), 'replaces': ['B']}
    reply = {'texts': [keeper], 'duplicate_ids': [
        {'id': 'A', 'duplicate_of': 'B', 'reason': 'Source has one baseline reading 16 units'}]}
    return rows, reply


def test_colliding_spellings_only_nominate_and_preserve_distinct_repeats():
    rows, _ = example()
    before = deepcopy(rows)
    assert duplicate_pairs(rows) == [['A', 'B']]
    assert apply_reply(rows, {'texts': [], 'duplicate_ids': []}, FRAME, [], options({})) == before
    assert rows == before


def test_explicit_source_reread_retires_false_spelling_not_real_repeated_label():
    rows, reply = example()
    before = deepcopy(rows)
    history = []
    selected = apply_reply(rows, reply, FRAME, history, options({}))
    assert [(r['id'], r['text']) for r in selected] == [('B', '16 units'), ('C', '18 units')]
    assert rows == before
    assert history[-1]['action'] == 'retire_source_confirmed_duplicate'


def test_missing_uncertain_or_different_occurrence_keeper_cannot_delete():
    rows, reply = example()
    for changed in ([], [{**reply['texts'][0], 'legibility': 'partial'}],
                    [{**reply['texts'][0], 'bbox_2d': [10, 70, 100, 90]}]):
        history = []
        proposal = {**deepcopy(reply), 'texts': changed}
        assert apply_reply(rows, proposal, FRAME, history, options({})) == rows
        assert proposal['duplicate_ids'] == [] and proposal['held_duplicate_ids']
        assert all(item['action'] == 'hold_source_alias' for item in history)


def test_partial_crop_does_not_authorize_retirement():
    rows, reply = example()
    selected = apply_reply(rows, reply, {**FRAME, 'box': [20, 0, 400, 200]}, [], options({}))
    assert selected == rows and not reply['duplicate_ids']
    assert 'crosses' in reply['held_duplicate_ids'][0]['hold_reason']


def test_one_unproved_alias_does_not_rollback_independent_source_proof():
    rows, reply = example()
    rows += [row('D', 'wrong', [200, 10, 260, 30]), row('E', 'right', [200, 10, 260, 30])]
    reply['texts'].append({**row('E', 'right', [200, 100, 260, 120]), 'replaces': ['E']})
    reply['duplicate_ids'].append({'id': 'D', 'duplicate_of': 'E', 'reason': 'Unproved location'})
    selected = apply_reply(rows, reply, FRAME, [], options({}))
    assert [r['id'] for r in selected] == ['B', 'C', 'D', 'E']
    assert next(r for r in selected if r['id'] == 'E')['bbox_2d'] == [200, 10, 260, 30]
    assert [r['id'] for r in reply['duplicate_ids']] == ['A']
    assert [r['id'] for r in reply['held_duplicate_ids']] == ['D']


def test_formula_scripts_and_different_orientations_are_not_plain_aliases():
    rows, _ = example()
    for protected in ({'kind': 'formula'}, {'rotation_deg': 90},
                      {'runs': [{'text': '2', 'script': 'super'}]}):
        assert not duplicate_pairs([rows[0], {**rows[1], **protected}])


def test_uncertainty_normalization_keeps_uncertainty_and_cycle_guards():
    install()
    from shape_integration.texts.inventory_audit import validate_reply
    rows, _ = example()
    entries = [{**rows[0], 'replaces': ['A'], 'legibility': 'cut'},
               {**rows[1], 'replaces': ['B'], 'legibility': 'unclear'}]
    checked = validate_reply({'texts': entries}, FRAME['size'], options({}), ['A', 'B'], [['A', 'B']])
    assert [r['legibility'] for r in checked['texts']] == ['partial', 'unreadable']
    assert not checked['duplicate_ids']
    for deletions in ([{'id': 'OTHER', 'duplicate_of': 'B', 'reason': 'unknown'}],
                      [{'id': 'A', 'duplicate_of': 'B', 'reason': 'same'},
                       {'id': 'B', 'duplicate_of': 'A', 'reason': 'same'}]):
        with pytest.raises(ValueError):
            validate_reply({'texts': [], 'duplicate_ids': deletions}, FRAME['size'],
                           options({}), ['A', 'B'], [['A', 'B']])
