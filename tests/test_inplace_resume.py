import json
from pathlib import Path

import pytest

from full_recovery.inplace_resume import recover_transport_cache, input_reader, verify_spec, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_resume_retries_only_transient_failures_preserving_all_evidence(tmp_path):
    errors = {'failed': 'Model transport failed after 1 attempt(s): HTTPError HTTP 502',
              'schema': 'Invalid shape schema',
              'auth': 'Model transport failed after 1 attempt(s): HTTPError HTTP 401',
              'success': 'Model transport failed after 1 attempt(s): HTTPError HTTP 502'}
    for name, error in errors.items():
        write(tmp_path / name / 'error_0.json', {'error': error})
        write(tmp_path / name / 'identity.json', {'request': name})
    write(tmp_path / 'success/result.json', {'ok': True})
    moved = recover_transport_cache(tmp_path)
    assert len(moved) == 1 and not (tmp_path / 'failed').exists()
    assert json.loads((Path(moved[0]['archive']) / 'identity.json').read_text()) == {'request': 'failed'}
    assert all((tmp_path / name).exists() for name in ['schema', 'auth', 'success'])
    assert recover_transport_cache(tmp_path) == []


def test_sidecar_changes_runtime_inputs_without_rewriting_original_proofs(tmp_path):
    original, replacement, spec_path = [tmp_path / n for n in ['settings.json', 'next.json', 'spec.json']]
    write(original, {'sam': 'old'})
    write(replacement, {'sam': 'new'})
    spec = {'inputs': {str(original): str(replacement)},
            'artifacts': {str(original): sha(original), str(replacement): sha(replacement)}}
    write(spec_path, spec)
    assert input_reader(verify_spec(spec_path))(original) == {'sam': 'new'}
    assert json.loads(original.read_text()) == {'sam': 'old'}
    write(replacement, {'sam': 'tampered'})
    with pytest.raises(ValueError, match='Continuation artifact changed'):
        verify_spec(spec_path)
