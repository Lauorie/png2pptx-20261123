from copy import deepcopy

from PIL import Image
import pytest

from shape_integration.common import read, save, sha
from shape_integration.texts import options
from shape_integration.texts.controller import mapped_box
from shape_integration.texts.inventory_audit import (
    apply_reply, audit, duplicate_pairs, recall_regions, validate_reply)
from shape_integration.texts.layout_integrity import new_conflicts
from test_two_stage_crop import original


def row(ident, text, box):
    return {**original(), 'id': ident, 'text': text, 'bbox_2d': box,
            'runs': [{'text': text, 'script': 'normal', 'color': '#000000'}]}


def local(row, replaces):
    return {**deepcopy(row), 'replaces': replaces}


FRAME = {'box': [0, 0, 400, 200], 'scale': 1, 'offset': [0, 0],
         'source_size': [400, 200], 'size': [400, 200]}


def test_containment_only_nominates_same_physical_plain_text():
    a = row('T0001', '(1) input', [10, 10, 100, 30])
    b = row('T0002', 'input', [40, 10, 100, 30])
    before = deepcopy([a, b])
    assert duplicate_pairs([a, b]) == [['T0001', 'T0002']]
    assert [a, b] == before
    assert not duplicate_pairs([a, {**b, 'bbox_2d': [40, 50, 100, 70]}])
    assert not duplicate_pairs([a, {**b, 'kind': 'formula'}])
    assert not duplicate_pairs([a, {**b, 'rotation_deg': 90}])
    assert not duplicate_pairs([row('A', 'data', [0, 0, 20, 10]),
                                row('B', 'a', [5, 0, 10, 10])])


def test_duplicate_review_can_separate_a_list_number_without_losing_text():
    rows = [row('T0001', '(1) input', [10, 10, 100, 30]),
            row('T0002', 'input', [40, 10, 100, 30])]
    raw = {'texts': [local(row('T0001', '(1)', [10, 30, 30, 50]), ['T0001']),
                     local(row('T0002', 'input', [40, 10, 100, 30]), ['T0002'])]}
    reply = validate_reply(raw, FRAME['size'], options({}), ['T0001', 'T0002'],
                           duplicate_pairs(rows))
    result = apply_reply(rows, reply, FRAME, [], options({}))
    assert [r['text'] for r in result] == ['(1)', 'input']
    assert not duplicate_pairs(result)
    assert rows[0]['text'] == '(1) input'


def test_retirement_requires_reread_keeper_preserving_removed_ink():
    rows = [row('T0001', '(1) input', [10, 10, 100, 30]),
            row('T0002', 'input', [40, 10, 100, 30])]
    raw = {'texts': [local(rows[0], ['T0001'])], 'duplicate_ids': [
        {'id': 'T0002', 'duplicate_of': 'T0001', 'reason': 'Same five visible glyphs'}]}
    reply = validate_reply(raw, FRAME['size'], options({}), ['T0001', 'T0002'],
                           duplicate_pairs(rows))
    history = []
    assert len(apply_reply(rows, reply, FRAME, history, options({}))) == 1
    assert history[-1]['action'] == 'retire_source_confirmed_duplicate'
    for bad in ({**reply, 'texts': []},
                {**reply, 'texts': [local(row('T0001', '(1)', [10, 10, 30, 30]), ['T0001'])]},
                {**reply, 'texts': [local(row('T0001', '(1) input', [200, 100, 300, 130]), ['T0001'])]}):
        before = deepcopy(history)
        with pytest.raises(ValueError):
            apply_reply(rows, bad, FRAME, history, options({}))
        assert history == before and len(rows) == 2


def test_unknown_deletion_and_cycles_are_rejected():
    ids = ['T0001', 'T0002']
    for removals in ([{'id': 'OTHER', 'duplicate_of': 'T0001', 'reason': 'wrong'}],
                     [{'id': 'T0001', 'duplicate_of': 'T0002', 'reason': 'same'},
                      {'id': 'T0002', 'duplicate_of': 'T0001', 'reason': 'same'}]):
        with pytest.raises(ValueError):
            validate_reply({'texts': [], 'duplicate_ids': removals}, FRAME['size'],
                           options({}), ids, [ids])


def test_additions_never_reuse_retired_or_still_active_ids():
    from shape_integration.texts.controller import merge
    rows = [row('T0001', 'A', [10, 10, 20, 20]),
            row('T0003', 'B', [30, 10, 40, 20])]
    history = [{'action': 'retire_source_confirmed_duplicate', 'id': 'T0005'}]
    merge(rows, [local(row('', 'C', [50, 10, 60, 20]), [])], FRAME, 'audit', history, options({}))
    assert rows[-1]['id'] == 'T0006'


def test_dense_recall_does_not_depend_on_model_regions_or_omit_empty_tiles():
    rows = [row(f'T{i:04d}', 'label', [10, 10, 30, 20]) for i in range(50)]
    boxes = recall_regions(rows, [1600, 800])
    assert len(boxes) == 4 and boxes[-1][2:] == [1599, 799]
    assert recall_regions(rows, [1600, 800], 2) == boxes[:2]
    assert recall_regions([row('A', 'Title', [10, 10, 100, 70])], [300, 200]) == []


def test_empty_inventory_still_gets_a_recall_view():
    assert recall_regions([], [300, 200]) == [[0, 0, 299, 199]]


def test_recall_can_add_white_rotated_text_without_existing_id(tmp_path, monkeypatch):
    from shape_integration.texts import model
    image = tmp_path/'input.png'
    Image.new('RGB', (400, 200), '#4060A0').save(image)
    rows = [row(f'T{i:04d}', 'label', [10 + i * 5, 10, 25 + i * 5, 20]) for i in range(1, 17)]
    source = {'texts': rows, 'source_sha256': sha(image), 'merge_history': [],
              'regions_needing_review': []}
    calls = []
    def ask(runtime, folder, prompt, images, validator, opt):
        calls.append(prompt)
        entry = local(row('', 'Missing label', [40, 40, 60, 150]), [])
        entry.update(rotation_deg=-90, color='#FFFFFF',
                     runs=[{'text': 'Missing label', 'script': 'normal', 'color': '#FFFFFF'}])
        return validator({'texts': [entry] if len(calls) == 1 else [], 'regions': []})
    monkeypatch.setattr(model, 'ask', ask)
    result = audit(source, image, tmp_path/'audit', object(), options({}))
    assert len(result['texts']) == 17 and len(source['texts']) == 16
    assert result['texts'][-1]['color'] == '#FFFFFF'
    assert len(calls) == 2
    assert read(tmp_path/'audit/summary.json')['regions'] == 2


def test_audit_error_never_drops_inventory(tmp_path, monkeypatch):
    from shape_integration.texts import model
    image = tmp_path/'input.png';Image.new('RGB', (400, 200)).save(image)
    rows = [row('T0001', '(1) input', [10, 10, 100, 30]),
            row('T0002', 'input', [40, 10, 100, 30])]
    def fail(*args, **kwargs):
        raise RuntimeError('upstream malformed response')
    monkeypatch.setattr(model, 'ask', fail)
    source = {'texts': rows}
    result = audit(source, image, tmp_path/'audit', object(), options({}))
    assert result['texts'] == rows and 'regions_needing_review' not in source
    assert result['regions_needing_review']


def test_new_conflict_reports_the_unmodified_bad_baseline_blocker():
    a = row('A', 'Container', [700, 100, 760, 120])
    b = row('B', 'Quantum', [500, 150, 560, 170])
    events = new_conflicts([a, b], {'B': [{**b, 'bbox_2d': a['bbox_2d']}]})
    assert events[0]['parents'] == ['A', 'B']


def test_failed_local_crop_and_bad_blocker_recover_before_rollback(tmp_path, monkeypatch):
    from shape_integration.texts import two_stage_crop as pipeline
    image = tmp_path/'input.png';Image.new('RGB', (1000, 400), 'white').save(image)
    a = row('A', 'Container', [700, 100, 760, 120])
    b = row('B', 'Quantum', [650, 160, 710, 180])
    correct = {'A': [100, 100, 160, 120], 'B': [700, 100, 760, 120]}
    baseline = tmp_path/'baseline';save(baseline/'scene.json', {'texts': [a, b]})
    Image.new('RGB', (1000, 400), 'white').save(baseline/'native.png')
    output = tmp_path/'run';output.mkdir()
    attempts = []
    class Runtime:
        def for_stage(self, **kwargs):
            return self
        def call(self, folder, tag, system, prompt, images, validator, budget):
            wide = 'recovery' in folder.parts
            attempts.append((prompt['stage'], wide))
            items = []
            f = read(folder.parent/'input/frame.json')
            for target in prompt['targets']:
                ident = target['id']
                if prompt['stage'] == '1_bbox_only':
                    item = {'id': ident, 'status': 'ambiguous', 'units': [], 'reason': 'Outside first crop'}
                    if wide or ident == 'B':
                        item.update(status='resolved', units=[{'bbox_2d': mapped_box(correct[ident], f)}])
                else:
                    unit = {**deepcopy(a if ident == 'A' else b), 'visual_line_count': 1,
                            'direction_status': 'confirmed', 'reason': 'Visible unique label'}
                    item = {'id': ident, 'status': 'resolved', 'units': [unit]}
                items.append(item)
            return validator({'items': items})
    def fit(rows, *args, **kwargs):
        return [{**r, 'font_size_px': 18} for r in rows], []
    monkeypatch.setattr(pipeline.layout, 'fit_all', fit)
    monkeypatch.setattr(pipeline.export, 'final', lambda rows, source, folder, *args:
        Image.new('RGB', (1000, 400), 'white').save(folder/'native.png'))
    result = pipeline.finish(image, output, baseline, options({}), Runtime())
    selected = read(output/'final/scene.json')['texts']
    assert {r['id'] for r in selected} == set(correct)
    for selected_row in selected:
        assert selected_row['bbox_2d'] == pytest.approx(correct[selected_row['id']])
    assert set(result['accepted_ids']) == {'A', 'B'} and not result['unresolved_ids']
    recovery = read(output/'two_stage_crop/recovery_summary.json')
    assert recovery['recovered_ids'] == ['A', 'B'] and recovery['conflicts_before_recovery']
    assert any(stage == '2_attributes_at_locked_boxes' and wide for stage, wide in attempts)
