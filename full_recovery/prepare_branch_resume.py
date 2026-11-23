"""Freeze a branch-only continuation without rewriting historical code proofs."""
import argparse
import fcntl
from pathlib import Path
import shutil
import time

from .prepare import read, save, sha
from .text_product_validation import verify


def prepare(source, run, key_file):
    if run.exists():
        raise ValueError('Use a new continuation directory')
    project = Path(__file__).resolve().parents[1]
    items = read(source / 'source_manifest.json')['images']
    old = read(source / 'settings.json')
    pending_text, pending_shape, preserved = [], [], {}
    runtime_metadata_missing = []
    with (source / 'worker.lock').open('r') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for prefix, manifest in ((source, source / 'code_manifest.json'),
                                 (source / 'shape/code', source / 'shape/code_sha256.json')):
            for relative, digest in read(manifest).items():
                if (not (prefix / relative).exists() and relative in {
                        'code/full_recovery/assets/math_font/.uuid',
                        'code/full_recovery/assets/text_symbol_fonts/.uuid'}):
                    runtime_metadata_missing.append(relative)
                    continue
                if sha(prefix / relative) != digest:
                    raise ValueError('Historical frozen input changed: ' + relative)
        for item in items:
            cid = item['id']
            text = source / 'texts/cases' / cid
            shape = source / 'shape/geometry/cases' / cid / 'complete.json'
            record = {}
            if (text / 'summary.json').is_file():
                record['text'] = verify(item, text)
            else:
                pending_text.append(cid)
            if shape.is_file():
                row = read(shape)
                if sha(row['pptx']) != row['pptx_sha256']:
                    raise ValueError('Completed shape changed: ' + cid)
                record['shape'] = row
            else:
                pending_shape.append(cid)
            if record:
                preserved[cid] = record
        run.mkdir(parents=True)
        for folder in ('jobs', 'logs', 'render_locks', 'texts/cases', 'case_slots', 'budget'):
            (run / folder).mkdir(parents=True, exist_ok=True)
        private = run / 'private'
        private.mkdir(mode=0o700)
        key = private / 'api_key'
        shutil.copyfile(key_file, key)
        key.chmod(0o600)
        shutil.copytree(source / 'code', run / 'code', symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(source / 'shape/code', run / 'shape/code', symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('case_slots.py', 'branches_only.py', 'api_worker.py', 'atomic_json.py',
                     'api_budget.py', 'api_gateway.py'):
            shutil.copy2(project / 'full_recovery' / name, run / 'code/full_recovery' / name)
        shutil.copy2(project / 'run_shapes.py', run / 'shape/code/run_shapes.py')
        shutil.copy2(project / 'full_recovery/case_slots.py', run / 'shape/code/full_recovery/case_slots.py')
        (run / 'code/branches_only.py').write_text(
            'from full_recovery.branches_only import main\nif __name__ == "__main__": main()\n')
        for name in ('backends', 'renderer_runtime'):
            (run / name).symlink_to(source / name, target_is_directory=True)
        for item in items:
            target = source / 'texts/cases' / item['id']
            (run / 'texts/cases' / item['id']).symlink_to(target, target_is_directory=True)
        slots = {'root': str(run / 'case_slots'), 'total': 24, 'shape_reserved': 16}
        settings = {**old, 'workflow_scope': 'shape_text', 'composition_enabled': False,
                    'arrow_inference_enabled': False, 'case_workers': 24, 'text_workers': 24,
                    'api_key_file': str(key), 'api_key_files': [str(key)],
                    'glm_api_key_file': str(key), 'api_credential_source_files': [str(key_file)],
                    'api_global_concurrency': 96, 'case_slot_config': slots,
                    'api_budget_config': str(run / 'budget/config.json')}
        settings.pop('api_relays', None)
        save(run / 'settings.json', settings)
        save(run / 'shape_resume_settings.json', {'case_slot_config': slots, 'case_workers': 24,
             'sam_url': settings['sam_url'], 'sam_urls': settings['sam_urls'],
             'resume_finished_geometry': True, 'resume_transport_cases_once': True})
        save(run / 'source_manifest.json', {'images': items})
        save(run / 'shape_pending_ids.json', pending_shape)
        save(run / 'resume_selection.json', {'source_run': str(source), 'pending_shape': pending_shape,
             'pending_text': pending_text, 'preserved': preserved, 'at': time.time(),
             'missing_fontconfig_runtime_metadata': runtime_metadata_missing,
             'code_policy': 'Original recognition/critic backends retained; scheduling only changed',
             'completed_products_reused': True, 'composition_enabled': False})
        freeze(run)
        save(run / 'status.json', {'state': 'prepared_waiting_for_valid_endpoint_and_budget_monitor',
             'pending_shape': len(pending_shape), 'pending_text': len(pending_text), 'at': time.time()})
        return {'run': str(run), 'pending_shape': len(pending_shape), 'pending_text': len(pending_text)}


def freeze(run):
    # This is used only before first launch. Never re-sign a submitted snapshot.
    if (run / 'jobs/job.id').exists():
        raise ValueError('Submitted snapshots are immutable')
    paths = [p for p in (run / 'code').rglob('*') if p.is_file() and p.suffix != '.pyc']
    paths += [run / name for name in ('settings.json', 'source_manifest.json',
              'resume_selection.json', 'shape_resume_settings.json', 'shape_pending_ids.json')]
    save(run / 'code_manifest.json', {str(p.relative_to(run)): sha(p) for p in paths})
    root = run / 'shape/code'
    save(run / 'shape/code_sha256.json', {str(p.relative_to(root)): sha(p)
         for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix != '.pyc'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source_run.resolve(), args.run_dir.resolve(), args.key_file.resolve()))


if __name__ == '__main__':
    main()
