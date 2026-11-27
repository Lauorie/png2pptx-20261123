"""Answer a request from the reply an earlier run recorded for the same request.

The text backend is a frozen copy whose runtime only remembers requests inside
one run directory. A replay of a finished run sits in a new directory, so each
request folder is seeded here from the finished run when, and only when, the
request is identical; the backend then finds its own folder already answered.
A request that differs in any way is left alone and goes to the model.
"""
import copy
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

REASON = 'Identical system/prompt/images/parameters/endpoint; no new decision'


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(folder: Path, output_root: Path, reference_roots: Iterable[Path], identity: Mapping[str, Any]) -> bool:
    """Copy the recorded reply for an identical request into a new request folder.

    Args:
        folder: The request folder about to be used; nothing is done if it exists.
        output_root: The run directory that contains ``folder``.
        reference_roots: Finished runs with the same folder layout.
        identity: Everything that defines the request.

    Returns:
        True when a recorded reply was put in place.
    """
    folder = Path(folder)
    output_root = Path(output_root)
    if folder.exists() or not folder.is_relative_to(output_root):
        return False
    for root in reference_roots:
        reference = Path(root) / folder.relative_to(output_root)
        if not (reference / 'identity.json').is_file() or not (reference / 'result.json').is_file():
            continue
        if _read(reference / 'identity.json') != identity:
            continue
        result = _read(reference / 'result.json')
        if isinstance(result, dict) and result.get('failed_groups'):
            continue
        folder.mkdir(parents=True)
        shutil.copy2(reference / 'identity.json', folder / 'identity.json')
        shutil.copy2(reference / 'result.json', folder / 'result.json')
        (folder / 'reuse.json').write_text(json.dumps({
            'source': str(reference), 'identity_sha256': _sha(reference / 'identity.json'),
            'result_sha256': _sha(reference / 'result.json'), 'reason': REASON}, indent=2) + '\n')
        return True
    return False


def install(common: Any) -> None:
    """Make the backend's runtime consult reference runs before its first attempt.

    Args:
        common: The backend's ``shape_integration.common`` module; its ``Runtime``
            class is replaced by a subclass. Runtimes whose config names no
            reference runs behave exactly as before.
    """
    original = common.Runtime

    class ReusingRuntime(original):
        def call(self, root, tag, system, prompt, images, validator, budget, *, retry_builder=None):
            roots = self.config.get('reuse_request_roots') or []
            output_root = self.config.get('reuse_output_root')
            if roots and output_root:
                identity = {
                    'system': system, 'prompt': prompt,
                    'images': [{'label': label, 'sha256': _sha(path)} for label, path in images],
                    'parameters': self.client(root, budget).request_parameters(), 'endpoint': self.base,
                    'request_timeout_seconds': self.config.get('request_timeout_seconds', 600),
                    'timeout_policy': 'discard_no_retry', 'actual_prompt': copy.deepcopy(prompt)}
                if retry_builder is not None:
                    identity.update(actual_system=system, actual_images=identity['images'])
                seed(Path(root) / (tag + '_00'), Path(output_root), roots, identity)
            return super().call(root, tag, system, prompt, images, validator, budget, retry_builder=retry_builder)

    common.Runtime = ReusingRuntime
