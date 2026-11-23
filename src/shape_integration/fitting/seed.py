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
    # Numerical success is not visual acceptance. A rejected or unresolved
    # decomposition must be observed afresh, not locked into a partial edit.
    available = (status.get('status') == 'finished' and status.get('part_counts', {}).get('succeeded', 0) > 0
                and all(name in hashes for name in ['fit_local.json', 'fit_original.json', 'observation.json']))
    return {'path': str(revision.resolve()), 'sha256': hashes,
            'fit_reused': available and record.get('state') == 'accepted_by_model',
            'fallback_fit_available': available,
            'source_record_sha256': sha(folder/'record.json'), 'source_state': record['state']}


def reviewed_addition_seed(seed, review):
    """An explicit current fit review may repair missing-only geometry in place.

    The old run remains unresolved. This only preserves its existing contours
    while the ordinary controller adds and reviews missing source parts.
    """
    if (not seed or seed.get('fit_reused') or not seed.get('fallback_fit_available') or
            review.get('decision')!='revise' or
            not review.get('protocol_version','').startswith('shape_integration.fit_check.') or
            not review.get('issues') or any(issue.get('kind')!='missing' or
                issue.get('parts') or issue.get('part_ids') for issue in review['issues'])):
        return seed
    local=read(Path(seed['path'])/'fit_local.json')
    if (local['fit_status']!='ok' or local['observation_status']!='ok' or
            set(review.get('part_id_map',{}).values())!={p['id'] for p in local['parts']} or
            not any(row.get('source_relation')=='supported' for row in review.get('source_observations',[]))):
        return seed
    return {**seed,'fit_reused':True,'reviewed_edit_scope':'current_source_missing_only',
            'historical_state_unchanged':seed.get('source_state')}


def reviewed_coverage_recovery(history, selected):
    """A residual local defect must not erase source-confirmed recovered parts."""
    entry=next((row for row in history if row['directory']==selected),{})
    status=entry.get('observation',{});review=entry.get('review',{})
    coverage=[r for r in review.get('source_observations',[])
              if r['id'].startswith('coverage_request_')]
    return bool(coverage and all(r['source_relation']=='supported' for r in coverage)
        and status.get('fit_status')=='ok' and status.get('observation_status')=='ok'
        and review.get('decision')=='revise' and review.get('issues')
        and all(r['kind'] in ('geometry','topology','occlusion') and r.get('part_ids')
                for r in review['issues']))


def checked_job(seed,candidate,item):
    source = Path(seed['path'])
    for name, digest in seed['sha256'].items():
        if sha(source/name) != digest:
            raise ValueError('Seed changed during import: '+str(source/name))
    job = read(source/'job.json')
    if (job['candidate'] != candidate or job['frame']['original_image_size_px'] != item['source_size']
            or job['frame']['global_image_size_px'] != item['input_size']):
        raise ValueError('Seed identity/frame mismatch')
    return job


def import_seed(seed, folder, candidate, item, *, input_context=None):
    source=Path(seed['path']);job=checked_job(seed,candidate,item)
    refresh_context = input_context is not None
    if refresh_context:
        job['input_context'] = input_context
        job['identity_descriptions'] = input_context.get('identity_descriptions', [])
        job['revision_feedback'] = []
        job.pop('revision_context', None)
    if not seed['fit_reused']:
        # A failed historical observation needs the current protocol. Keep its
        # crop/identity frozen, preserve the source prompt and change only the
        # new destination's request; never edit a historical run in place.
        # A fresh observation must not inherit an old failed critic's claims.
        # Keep them in the immutable seed for audit, not in the new request.
        job['revision_feedback'] = []
        job.pop('revision_context', None)
    if not seed['fit_reused'] or refresh_context:
        from .observation import system_for
        request = {'system': system_for(job), 'prompt': make_prompt(job, job.get('identity_descriptions', ()))}
        encoded = json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        job['input_sha256']['prompt.json'] = hashlib.sha256(encoded.encode()).hexdigest()
        job['observation_protocol'] = MODEL_VERSION
    locator_bytes=None
    if refresh_context and input_context.get('coverage_source_anchors'):
        from io import BytesIO
        from PIL import Image,ImageDraw,ImageFont
        from .views import FONT
        from .coverage_evidence import annotate
        with Image.open(source/'clean.png') as image:locator=image.convert('RGB')
        draw=ImageDraw.Draw(locator);edge=max(locator.size);box=job['identity_bbox_local']
        font=ImageFont.truetype(FONT,max(14,edge//70))
        draw.rectangle(box,outline='#e00090',width=max(2,edge//600))
        draw.text((max(0,box[0]),max(0,box[1]-edge//45)),candidate['id'],font=font,
                  fill='#e00090',stroke_width=1,stroke_fill='white')
        annotate(locator,job)
        buffer=BytesIO();locator.save(buffer,format='PNG');locator_bytes=buffer.getvalue()
        job['input_sha256']['overlay.png']=hashlib.sha256(locator_bytes).hexdigest()
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
        if not seed['fit_reused'] or refresh_context:
            shutil.copy2(stage/'prompt.json', stage/'source_prompt.json')
            if locator_bytes is not None:
                shutil.copy2(stage/'overlay.png',stage/'source_locator.png')
                (stage/'overlay.png').write_bytes(locator_bytes)
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
