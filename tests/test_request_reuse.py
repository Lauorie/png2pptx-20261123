"""Answering a request from the reply an earlier run recorded for the same request."""
import json
from pathlib import Path
from types import SimpleNamespace

from full_recovery.request_reuse import install, seed


def _record(folder: Path, identity: dict, result: dict) -> None:
    folder.mkdir(parents=True)
    (folder / 'identity.json').write_text(json.dumps(identity))
    (folder / 'result.json').write_text(json.dumps(result))


def test_identical_request_is_seeded_with_the_recorded_reply(tmp_path: Path) -> None:
    _record(tmp_path / 'old/cases/a/calls/r_00', {'prompt': 1}, {'answer': 7})
    folder = tmp_path / 'new/cases/a/calls/r_00'
    assert seed(folder, tmp_path / 'new', [tmp_path / 'old'], {'prompt': 1})
    assert json.loads((folder / 'result.json').read_text()) == {'answer': 7}
    assert json.loads((folder / 'identity.json').read_text()) == {'prompt': 1}
    assert (folder / 'reuse.json').is_file()


def test_a_different_request_is_left_for_the_model(tmp_path: Path) -> None:
    _record(tmp_path / 'old/cases/a/calls/r_00', {'prompt': 1}, {'answer': 7})
    folder = tmp_path / 'new/cases/a/calls/r_00'
    assert not seed(folder, tmp_path / 'new', [tmp_path / 'old'], {'prompt': 2})
    assert not folder.exists()


def test_a_reply_recorded_as_partly_failed_is_not_replayed(tmp_path: Path) -> None:
    _record(tmp_path / 'old/c/r_00', {'prompt': 1}, {'failed_groups': ['g']})
    assert not seed(tmp_path / 'new/c/r_00', tmp_path / 'new', [tmp_path / 'old'], {'prompt': 1})


def test_a_folder_outside_the_run_is_never_seeded(tmp_path: Path) -> None:
    _record(tmp_path / 'old/c/r_00', {'prompt': 1}, {'answer': 7})
    assert not seed(tmp_path / 'elsewhere/c/r_00', tmp_path / 'new', [tmp_path / 'old'], {'prompt': 1})


class _Client:
    def request_parameters(self) -> dict:
        return {'max_tokens': 5}


def _runtime_class(calls: list) -> type:
    class Runtime:
        def __init__(self, config: dict) -> None:
            self.config = config
            self.base = 'http://endpoint'

        def client(self, root: Path, budget: int) -> _Client:
            return _Client()

        def call(self, root, tag, system, prompt, images, validator, budget, *, retry_builder=None):
            folder = Path(root) / (tag + '_00')
            if (folder / 'result.json').exists():
                return validator(json.loads((folder / 'result.json').read_text()))
            calls.append(tag)
            return validator({'answer': 'model'})
    return Runtime


def _identity(prompt: dict) -> dict:
    return {'system': 's', 'prompt': prompt, 'images': [], 'parameters': {'max_tokens': 5},
            'endpoint': 'http://endpoint', 'request_timeout_seconds': 360,
            'timeout_policy': 'discard_no_retry', 'actual_prompt': prompt}


def test_installed_runtime_replays_a_recorded_request_without_calling_the_model(tmp_path: Path) -> None:
    calls: list = []
    common = SimpleNamespace(Runtime=_runtime_class(calls))
    _record(tmp_path / 'old/c/review_00', _identity({'q': 1}), {'answer': 'recorded'})
    install(common)
    runtime = common.Runtime({'request_timeout_seconds': 360, 'reuse_output_root': str(tmp_path / 'new'),
                              'reuse_request_roots': [str(tmp_path / 'old')]})
    assert runtime.call(tmp_path / 'new/c', 'review', 's', {'q': 1}, [], lambda r: r, 100) == {'answer': 'recorded'}
    assert runtime.call(tmp_path / 'new/c', 'other', 's', {'q': 2}, [], lambda r: r, 100) == {'answer': 'model'}
    assert calls == ['other']


def test_installed_runtime_is_unchanged_without_reference_runs(tmp_path: Path) -> None:
    calls: list = []
    common = SimpleNamespace(Runtime=_runtime_class(calls))
    install(common)
    runtime = common.Runtime({'request_timeout_seconds': 360})
    assert runtime.call(tmp_path / 'c', 'review', 's', {'q': 1}, [], lambda r: r, 100) == {'answer': 'model'}
    assert not (tmp_path / 'c').exists()
