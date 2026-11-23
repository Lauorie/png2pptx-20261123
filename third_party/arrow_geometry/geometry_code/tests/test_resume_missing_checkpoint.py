import copy
import json
from pathlib import Path

import pytest
from PIL import Image

import resume_missing_checkpoint as checkpoint
import post_geometry_missing as missing
from transport.common import write_json as write, read_json as read
from test_post_geometry_missing import row, FakeModel
from test_bbox_pipeline import BASE, locks


def frozen(tmp_path):
    old = tmp_path/'old'; root = tmp_path/'new'; cid = 'case'
    source = old/'geometry/cases'/cid; source.mkdir(parents=True)
    Image.new('RGB', (100,80), 'white').save(source/'clean.png')
    image = root/'localization/inputs'/cid/'source.png'; image.parent.mkdir(parents=True)
    image.write_bytes((source/'clean.png').read_bytes())
    write(source/'reconstruction_handoff.json', {'tasks':[]})
    write(root/'localization/cases'/cid/'reconstruction_handoff.json', {'tasks':[]})
    groups = [dict(group='old', state='completed', rows=[row()], issues=[])]
    write(source/'initial_geometry_results.json', groups)
    write(source/'post_geometry_missing/geometry_before.json', groups)
    write(source/'post_geometry_missing/deferred_localization_input.json', [])
    write(source/'post_geometry_missing/candidates.json', {'tasks':[
        dict(id='GLOBAL_001', source='GLOBAL', bbox_native=[30,40,80,60], origin={})]})
    write(source/'post_geometry_missing/audit.json', {'issues':[
        dict(stage='global_missing_discovery', error='previous failure'),
        dict(stage='missing_check', error='old review must not propagate')]})
    return old, root, source, groups


def test_frozen_queue_replays_only_review_and_addition_geometry(tmp_path, monkeypatch):
    old, root, source, groups = frozen(tmp_path)
    checkpoint.freeze(old, root, ['case'])
    initial, queue, provenance = checkpoint.load(root/'resume_missing/case')
    monkeypatch.setattr(missing, 'global_discover', lambda *a: pytest.fail('discovery reran'))
    cfg = {**BASE, 'resume_before_missing_from':str(root/'resume_missing'),
           'missing_review_context_enabled':True}
    model = FakeModel(); before = copy.deepcopy(initial)
    final, issues = missing.run(Image.open(source/'clean.png'), initial, root/'output', cfg,
                                model, locks(), 'case')
    checks = [c for c in model.calls if 'candidate' in c[1]]
    assert len(checks) == 1
    assert [x[0] for x in checks[0][2]] == ['CLEAN','CURRENT','CONTEXT']
    assert '引线' not in checks[0][1]['images']['CURRENT']
    assert not read(root/'output/candidates/GLOBAL_001/view_00/locator_annotations.json')['local']['leader']
    assert initial == before and final[0] == initial[0] and len(final) == 2
    assert [x['stage'] for x in issues] == ['global_missing_discovery']
    assert all('candidate' in c[1] or 'targets' in c[1] for c in model.calls)


def test_resume_rejects_geometry_after_omissions(tmp_path):
    old, root, source, groups = frozen(tmp_path)
    write(source/'post_geometry_missing/geometry_before.json', [])
    with pytest.raises(ValueError, match='differs'):
        checkpoint.freeze(old, root, ['case'])


def test_resume_detects_changed_frozen_candidates(tmp_path):
    old, root, source, groups = frozen(tmp_path)
    checkpoint.freeze(old, root, ['case'])
    write(root/'resume_missing/case/post_geometry_missing/candidates.json', {'tasks':[]})
    with pytest.raises(ValueError, match='changed'):
        checkpoint.load(root/'resume_missing/case')
