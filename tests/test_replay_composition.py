"""Replaying only the composition of a finished run, on its frozen branch products."""
import hashlib
import json
from pathlib import Path

import pytest

from full_recovery.replay_composition import reuse_arguments, reuse_report, stage


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def world(tmp_path: Path) -> dict:
    project = tmp_path / 'project'
    _write(project / 'full_recovery/merge_existing.py', 'CURRENT = 1\n')
    _write(project / 'scripts/lo_guard.py', '# guard\n')
    _write(project / 'src/shape.py', 'CURRENT = 1\n')
    source = tmp_path / 'jobs/done'
    _write(source / 'settings.json', json.dumps({
        'source_project': str(project), 'workflow_scope': 'fresh_full',
        'case_workers': 6, 'reuse_request_roots': []}))
    _write(source / 'sources.json', json.dumps({'ready': [{'item': {'id': 'a'}}], 'missing': []}))
    _write(source / 'source_manifest.json', json.dumps({'images': [{'id': 'a'}]}))
    _write(source / 'shape/pipeline_config.json', '{"model": "m"}')
    old = _write(source / 'shape/code/src/shape.py', 'OLD = 1\n')
    only = _write(source / 'shape/code/config.json', '{}')
    _write(source / 'shape/code_sha256.json', json.dumps(
        {'src/shape.py': _sha(old), 'config.json': _sha(only)}))
    _write(source / 'backends/text/config.json', '{}')
    _write(source / 'renderer_runtime/lo_cpu.py', '#\n')
    (tmp_path / 'jobs/done.input').mkdir()
    return {'project': project, 'source': source, 'target': tmp_path / 'jobs/replay'}


def test_staged_run_composes_only_and_reuses_the_finished_runs_decisions(world: dict) -> None:
    stage(world['source'], world['target'], {})
    settings = json.loads((world['target'] / 'settings.json').read_text())
    assert settings['workflow_scope'] == 'composition'
    assert settings['reuse_request_roots'] == [str(world['source'])]
    assert settings['reuse_assets_from'] == str(world['source'])


def test_overrides_reach_the_staged_settings_but_not_the_finished_run(world: dict) -> None:
    before = (world['source'] / 'settings.json').read_text()
    stage(world['source'], world['target'], {'source_residual_pictures': False})
    settings = json.loads((world['target'] / 'settings.json').read_text())
    assert settings['source_residual_pictures'] is False
    assert (world['source'] / 'settings.json').read_text() == before


def test_staged_code_is_the_projects_current_code(world: dict) -> None:
    stage(world['source'], world['target'], {})
    target = world['target']
    assert (target / 'code/full_recovery/merge_existing.py').read_text() == 'CURRENT = 1\n'
    assert (target / 'shape/code/src/shape.py').read_text() == 'CURRENT = 1\n'
    assert (target / 'shape/code/config.json').read_text() == '{}'


def test_staged_manifests_match_the_staged_files(world: dict) -> None:
    stage(world['source'], world['target'], {})
    target = world['target']
    for prefix, name in ((target, 'code_manifest.json'), (target / 'shape/code', 'shape/code_sha256.json')):
        manifest = json.loads((target / name).read_text())
        assert manifest
        for relative, digest in manifest.items():
            assert _sha(prefix / relative) == digest


def test_branch_products_are_bound_exactly_as_the_finished_run_bound_them(world: dict) -> None:
    stage(world['source'], world['target'], {})
    for name in ('sources.json', 'source_manifest.json'):
        assert (world['target'] / name).read_bytes() == (world['source'] / name).read_bytes()
    assert (world['target'] / 'backends').resolve() == (world['source'] / 'backends').resolve()
    assert Path(str(world['target']) + '.input').resolve() == Path(str(world['source']) + '.input').resolve()


def test_an_existing_directory_is_never_reused(world: dict) -> None:
    world['target'].mkdir(parents=True)
    with pytest.raises(ValueError):
        stage(world['source'], world['target'], {})


def test_report_separates_replayed_decisions_from_new_model_calls(tmp_path: Path) -> None:
    for name, reused in (('a/x/calls/r_00', True), ('a/y/calls/r_00', True), ('b/x/calls/r_00', False)):
        folder = tmp_path / 'cases' / name
        _write(folder / 'identity.json', '{}')
        _write(folder / 'result.json', '{}')
        if reused:
            _write(folder / 'reuse.json', '{}')
    assert reuse_report(tmp_path) == {'a': {'reused': 2, 'fresh': 0}, 'b': {'reused': 0, 'fresh': 1}}


def test_text_painting_is_told_where_earlier_decisions_live(tmp_path: Path) -> None:
    assert reuse_arguments({}, tmp_path) == []
    assert reuse_arguments({'reuse_request_roots': []}, tmp_path) == []
    assert reuse_arguments({'reuse_request_roots': ['/r1', '/r2']}, tmp_path) == [
        '--reuse-output-root', str(tmp_path), '--reuse-request-root', '/r1', '--reuse-request-root', '/r2']
