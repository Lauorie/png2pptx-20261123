"""Explicit sidecar inputs for continuing an immutable run in its original directory.

The submitted snapshot and its proofs remain unchanged. A separate receipt pins
the adapter and replacement input documents; the original worker still verifies
every original snapshot hash before execution.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_spec(path):
    spec = read(path)
    for name, digest in spec['artifacts'].items():
        if sha(name) != digest:
            raise ValueError('Continuation artifact changed: ' + name)
    for name in spec['inputs'].values():
        if str(Path(name).resolve()) not in spec['artifacts']:
            raise ValueError('Unpinned continuation input: ' + name)
    return spec


def input_reader(spec):
    mapping = {str(Path(k).resolve()): Path(v) for k, v in spec['inputs'].items()}

    def load(path):
        return read(mapping.get(str(Path(path).resolve()), path))
    return load


def recover_transport_cache(root):
    """Archive only failed transient image calls, never successful/schema calls.

    The detector otherwise rethrows persisted errors without issuing a request.
    Renaming alongside the original attempt keeps all evidence in the same case.
    This runs once on explicit continuation while the case writer owns the run.
    """
    moved = []
    for path in sorted(Path(root).glob('*/error_0.json')):
        folder = path.parent
        if '.transport_failed_' in folder.name or (folder / 'result.json').exists():
            continue
        error = read(path).get('error', '')
        transient = ('Incomplete model transport' in error or
                     ('Model transport failed' in error and
                      re.search(r'HTTP (408|425|429|500|502|503|504)\b', error)))
        if not transient:
            continue
        destination = folder.with_name(folder.name + '.transport_failed_' + str(time.time_ns()))
        folder.rename(destination)
        moved.append({'source': str(folder), 'archive': str(destination), 'error': error})
    return moved


def load_module(name, path):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    sys.modules[name] = module
    definition.loader.exec_module(module)
    return module


def main():
    Path('/proc/self/oom_score_adj').write_text('0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--stage', choices=['worker', 'pipeline'], default='worker')
    args = parser.parse_args()
    spec = verify_spec(args.spec)
    run = Path(spec['run'])
    sys.path.insert(0, str(run / 'code'))
    reader = input_reader(spec)
    if args.stage == 'worker':
        worker = load_module('full_recovery.continuation_api_worker', spec['worker_code'])
        worker.read = reader
        sys.argv = [sys.argv[0], '--run-dir', str(run)]
        worker.main(pipeline_argv=[str(Path(__file__).resolve()), '--spec', str(args.spec.resolve()),
                                   '--stage', 'pipeline'])
    elif spec['role'] == 'shape':
        if os.environ.get('STYLE_CLUSTER_WORKER') != '1':
            raise RuntimeError('Use the CPU worker')
        selection = reader(run / 'resume_selection.json')
        moved = []
        for cid in selection['pending_shape']:
            moved.extend(recover_transport_cache(Path(selection['source_run']) /
                         'shape/localization/cases' / cid / 'images/calls/detector'))
        receipt = run / ('transport_cache_recovery_' + str(time.time_ns()) + '.json')
        receipt.write_text(json.dumps({'requests': moved, 'successful_results_preserved': True}, indent=2))
        branch = load_module('full_recovery.continuation_branches', spec['branch_code'])
        branch.read = reader
        branch.execute(run, shape_resume_settings=spec['shape_resume_settings'],
                       shape_pending_ids=spec['shape_pending_ids'])
    elif spec['role'] == 'merge':
        merge = importlib.import_module('full_recovery.merge_existing')
        merge.read = reader
        merge.execute(run)
    else:
        raise ValueError('Unknown continuation role')


if __name__ == '__main__':
    main()
