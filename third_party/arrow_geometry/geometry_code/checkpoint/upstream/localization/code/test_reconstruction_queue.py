import json
from pathlib import Path

import pytest

from reconstruction_queue import budget_available, freeze_case, ready


def make_source(tmp_path):
    source = tmp_path / 'localization'
    case = source / 'cases' / 'sample'
    case.mkdir(parents=True)
    image = source / 'inputs' / 'sample' / 'source.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'unchanged source image')
    (case / 'result.json').write_text('{"state":"completed"}')
    (case / 'reconstruction_handoff.json').write_text('{"case_id":"sample","tasks":[]}')
    return source, case


def test_no_dispatch_of_partially_written_localization(tmp_path):
    source, case = make_source(tmp_path)
    assert not ready(source, 'sample')
    with pytest.raises(ValueError, match='not finished'):
        freeze_case(source, tmp_path / 'geometry', 'sample')
    (case / 'finished.json').write_text('{}')
    assert ready(source, 'sample')
    copied = freeze_case(source, tmp_path / 'geometry', 'sample')
    assert (copied / 'clean.png').read_bytes() == b'unchanged source image'
    assert len(json.loads((copied / 'input_provenance.json').read_text())) == 4


def test_resume_cannot_mix_changed_localization_and_old_geometry(tmp_path):
    source, case = make_source(tmp_path)
    (case / 'finished.json').write_text('{}')
    dest = freeze_case(source, tmp_path / 'geometry', 'sample')
    before = (dest / 'reconstruction_handoff.json').read_bytes()
    (case / 'reconstruction_handoff.json').write_text('{"case_id":"sample","tasks":[{"target_id":"changed"}]}')
    with pytest.raises(ValueError, match='frozen localization changed'):
        freeze_case(source, tmp_path / 'geometry', 'sample')
    assert (dest / 'reconstruction_handoff.json').read_bytes() == before


def test_old_run_progress_does_not_release_api_budget(tmp_path):
    previous = tmp_path / 'old_geometry'
    previous.mkdir()
    cfg = {'wait_for_geometry_run': str(previous)}
    (previous / 'progress.json').write_text('{"stage":"running"}')
    assert not budget_available(cfg)
    (previous / 'geometry_finished.json').write_text('{"total":14}')
    assert budget_available(cfg)
