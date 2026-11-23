"""Import immutable prepared inputs/geometry, never historical critic decisions."""
import hashlib
import json
from pathlib import Path
import shutil

from ..common import read, save, sha
from .contracts import MODEL_VERSION
from .observation import SYSTEM, make_prompt


def revision_seed(folder):
    record = read(folder/'record.json')
    selected = record.get('selected_revision')
    prepared = sorted(folder.glob('revision_*/job.json'))
    revision = folder/selected if selected else prepared[-1].parent if prepared else None
    if revision is None:
        raise ValueError('No prepared input for '+str(folder))
    job = read(revision/'job.json')
    names = set(job['input_sha256']) | {'job.json', 'native.png', 'status.json', 'observation.json',
             'fit_local.json', 'fit_original.json', 'comparison.png', 'description_audit.json',
             'observation_raw.json', 'observation_normalization.json'}
    hashes = {name: sha(revision/name) for name in sorted(names) if (revision/name).is_file()}
    for name, digest in job['input_sha256'].items():
        if hashes.get(name) != digest:
            raise ValueError('Seed input changed: '+str(revision/name))
    status = read(revision/'status.json') if (revision/'status.json').exists() else {}
    reusable = (status.get('status') == 'finished' and status.get('part_counts', {}).get('succeeded', 0) > 0
                and all(name in hashes for name in ['fit_local.json', 'fit_original.json', 'observation.json']))
    return {'path': str(revision.resolve()), 'sha256': hashes, 'fit_reused': reusable,
            'source_record_sha256': sha(folder/'record.json'), 'source_state': record['state']}


def import_seed(seed, folder, candidate, item):
    source = Path(seed['path'])
    for name, digest in seed['sha256'].items():
        if sha(source/name) != digest:
            raise ValueError('Seed changed during import: '+str(source/name))
    job = read(source/'job.json')
    if (job['candidate'] != candidate or job['frame']['original_image_size_px'] != item['source_size']
            or job['frame']['global_image_size_px'] != item['input_size']):
        raise ValueError('Seed identity/frame mismatch')
    if not seed['fit_reused']:
        # A failed historical observation needs the current protocol. Keep its
        # crop/identity frozen, preserve the source prompt and change only the
        # new destination's request; never edit a historical run in place.
        request = {'system': SYSTEM, 'prompt': make_prompt(job, job.get('identity_descriptions', ()))}
        encoded = json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        job['input_sha256']['prompt.json'] = hashlib.sha256(encoded.encode()).hexdigest()
        job['observation_protocol'] = MODEL_VERSION
    if not folder.exists():
        stage = folder.with_name(folder.name+'.preparing')
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        for name in seed['sha256']:
            if not seed['fit_reused'] and name in {'status.json', 'fit_local.json', 'fit_original.json', 'observation.json', 'comparison.png', 'observation_raw.json', 'observation_normalization.json'}:
                continue
            target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, target)
        if not seed['fit_reused']:
            shutil.copy2(stage/'prompt.json', stage/'source_prompt.json')
            save(stage/'prompt.json', request)
            save(stage/'job.json', job)
        save(stage/'reused_revision.json', seed)
        stage.rename(folder)
    elif read(folder/'reused_revision.json') != seed:
        raise ValueError('Reused revision identity changed')
    if read(folder/'job.json') != job:
        raise ValueError('Imported job changed')
    for name, digest in job['input_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('Imported prepared input changed')
    if seed['fit_reused']:
        for name in ['status.json', 'observation.json', 'fit_local.json', 'fit_original.json']:
            if sha(folder/name) != seed['sha256'][name]:
                raise ValueError('Imported geometry changed')
    return job
