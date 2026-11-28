"""Compose a finished run again from its frozen branch products.

The three branches (shapes, text, arrows) are where the model's run-to-run
variation enters. A replay binds the very same branch products, runs today's
composition code on them and answers every request the finished run already
made from its recorded reply. Two replays that differ in one setting therefore
differ only by that setting.

    python -m full_recovery.replay_composition FINISHED_RUN NEW_DIR [--set key=json ...]
    python NEW_DIR/code/full_recovery/api_worker.py --run-dir NEW_DIR
"""
import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

FROZEN_LINKS = ('backends', 'renderer_runtime')
WORK_FOLDERS = ('logs', 'jobs', 'render_locks', 'delivery', 'cases')
BOUND_DOCUMENTS = ('sources.json', 'source_manifest.json')
ENTRY = 'from full_recovery.merge_existing import main\nif __name__ == "__main__": main()\n'


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_code(project: Path, target: Path) -> None:
    """Freeze today's composition code, in the layout the worker verifies."""
    code = target / 'code'
    shutil.copytree(project / 'full_recovery', code / 'full_recovery',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (code / 'merge_existing.py').write_text(ENTRY)
    shutil.copy2(project / 'scripts/lo_guard.py', code / 'lo_guard.py')
    _save(target / 'code_manifest.json',
          {str(p.relative_to(target)): _sha(p) for p in sorted(code.rglob('*')) if p.is_file()})


def _stage_shape_code(project: Path, source: Path, target: Path) -> None:
    """The finished run's shape code tree, with every file the project still has taken from it."""
    old = source / 'shape/code'
    new = target / 'shape/code'
    shutil.copytree(old, new, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {}
    for relative in _read(source / 'shape/code_sha256.json'):
        current = project / relative
        if current.is_file():
            shutil.copy2(current, new / relative)
        manifest[relative] = _sha(new / relative)
    _save(target / 'shape/code_sha256.json', manifest)
    shutil.copy2(source / 'shape/pipeline_config.json', target / 'shape/pipeline_config.json')


def stage(source: Path, target: Path, overrides: Mapping[str, Any]) -> Path:
    """Prepare a composition-only run over a finished run's branch products.

    Args:
        source: A finished run directory; it is only read.
        target: The replay directory; must not exist.
        overrides: Settings that differ from the finished run.

    Returns:
        The replay directory, ready for ``api_worker.py --run-dir``.

    Raises:
        ValueError: When the target already exists.
    """
    source = Path(source).resolve()
    target = Path(target).resolve()
    if target.exists():
        raise ValueError('Use a new replay directory: ' + str(target))
    settings = _read(source / 'settings.json')
    project = Path(settings['source_project'])
    target.mkdir(parents=True)
    _stage_code(project, target)
    _stage_shape_code(project, source, target)
    for name in BOUND_DOCUMENTS:
        shutil.copy2(source / name, target / name)
    for name in FROZEN_LINKS:
        (target / name).symlink_to(source / name, target_is_directory=True)
    for name in WORK_FOLDERS:
        (target / name).mkdir()
    inputs = Path(str(source) + '.input')
    if inputs.exists():
        Path(str(target) + '.input').symlink_to(inputs, target_is_directory=True)
    _save(target / 'settings.json', {
        **settings, 'workflow_scope': 'composition', 'replay_of': str(source),
        'reuse_request_roots': [str(source)], 'reuse_assets_from': str(source), **overrides})
    logger.info('Staged composition replay of %s at %s', source, target)
    return target


def reuse_arguments(settings: Mapping[str, Any], run: Path) -> List[str]:
    """Arguments that let the text painting step answer from earlier runs' replies."""
    roots = settings.get('reuse_request_roots') or []
    if not roots:
        return []
    arguments = ['--reuse-output-root', str(run)]
    for root in roots:
        arguments += ['--reuse-request-root', str(root)]
    return arguments


def reuse_report(run: Path) -> Dict[str, Dict[str, int]]:
    """Per case: requests answered from a recorded reply, and requests the model answered anew."""
    report: Dict[str, Dict[str, int]] = {}
    cases = Path(run) / 'cases'
    for identity in sorted(cases.rglob('identity.json')):
        folder = identity.parent
        row = report.setdefault(folder.relative_to(cases).parts[0], {'reused': 0, 'fresh': 0})
        row['reused' if (folder / 'reuse.json').is_file() else 'fresh'] += 1
    return report


def _override(text: str) -> tuple:
    key, _, value = text.partition('=')
    if not key or not value:
        raise argparse.ArgumentTypeError('Expected key=json, got ' + text)
    return key, json.loads(value)


def main() -> None:
    """Stage a replay, or with --report print how much of one was replayed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path, nargs='?')
    parser.add_argument('--set', dest='overrides', type=_override, action='append', default=[])
    parser.add_argument('--report', action='store_true', help='Summarise reused and fresh requests of SOURCE')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.report:
        report = reuse_report(args.source)
        for case, row in report.items():
            logger.info('%-18s reused %4d  fresh %4d', case, row['reused'], row['fresh'])
        logger.info('%-18s reused %4d  fresh %4d', 'TOTAL', sum(r['reused'] for r in report.values()),
                    sum(r['fresh'] for r in report.values()))
        return
    if args.target is None:
        parser.error('A new replay directory is required')
    stage(args.source, args.target, dict(args.overrides))


if __name__ == '__main__':
    main()
