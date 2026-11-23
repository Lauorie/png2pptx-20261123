"""Freeze initial geometry and discovery queues; rerun review and all downstream stages."""
import copy
import hashlib
import json
from pathlib import Path

from checked_copy import copy_file, digest
from transport.common import read_json as read, write_json as write


def freeze(source_run, root, ids, *, fresh_style_calls=True):
    source_run = Path(source_run).resolve(); root = Path(root).resolve(); files = []
    from PIL import Image
    for cid in ids:
        source = source_run/'geometry/cases'/cid
        destination = root/'resume_missing'/cid
        destination.mkdir(parents=True)
        for old, new in [(source/'clean.png', root/'localization/inputs'/cid/'source.png'),
                         (source/'reconstruction_handoff.json', root/'localization/cases'/cid/'reconstruction_handoff.json')]:
            if digest(old) != digest(new):
                raise ValueError('Resume image or handoff mismatch: '+cid)
        initial = read(source/'initial_geometry_results.json')
        if initial != read(source/'post_geometry_missing/geometry_before.json'):
            raise ValueError('Initial geometry differs from omission-review input: '+cid)
        records = []
        def save(src, dst):
            before = digest(src); dst.parent.mkdir(parents=True, exist_ok=True)
            copy_file(src, dst)
            if before != digest(dst) or before != digest(src):
                raise ValueError('Resume input changed: '+str(src))
            records.append(dict(source=str(src), path=str(dst.relative_to(root)), sha256=before))
            files.append(dst)
        for name in ('initial_geometry_results.json', 'post_geometry_missing/geometry_before.json',
                     'post_geometry_missing/candidates.json', 'post_geometry_missing/deferred_localization_input.json'):
            save(source/name, destination/name)
        for name in ('global/native_regions.json', 'global/failure.json'):
            if (source/'post_geometry_missing'/name).exists():
                save(source/'post_geometry_missing'/name, destination/'post_geometry_missing'/name)
        old_sam = source_run/'sam/missing/cases'/cid
        for name in ('retained_candidates.json', 'status.json'):
            if (old_sam/name).exists(): save(old_sam/name, destination/'sam'/name)
        # Keep prior discovery failures, but never carry old review/addition decisions forward.
        audit = read(source/'post_geometry_missing/audit.json')
        discovery_issues = [x for x in audit.get('issues', []) if x.get('stage') in
                            ('global_missing_discovery', 'sam_scan', 'sam_candidates_unavailable')]
        effective = copy.deepcopy(initial); resources = {}
        def resource(path):
            original = Path(path).resolve()
            if str(original) not in resources:
                folder = destination/'resources'/hashlib.sha256(str(original).encode()).hexdigest()[:16]
                target = folder/original.name; save(original, target)
                with Image.open(target) as im: im.load()
                transform = original.parent/'transform.json'
                if transform.exists(): save(transform, folder/'transform.json')
                resources[str(original)] = str(target)
            return resources[str(original)]
        for group in effective:
            for row in group.get('rows', []):
                for key in ('local_input', 'context_image'):
                    if row.get(key): row[key] = resource(row[key])
                row['source_mask_paths'] = [resource(p) for p in row.get('source_mask_paths', [])]
        path = destination/'effective_initial_geometry_results.json'
        write(path, effective); files.append(path)
        provenance = destination/'provenance.json'
        write(provenance, dict(source_run=str(source_run), case=cid, resume_at='single_bbox_missing_check',
              fresh_initial_geometry_calls=False, fresh_discovery_calls=False, fresh_review_calls=True,
              fresh_addition_geometry_calls=True, fresh_reconciliation_calls=True, fresh_style_calls=fresh_style_calls,
              candidates=len(read(source/'post_geometry_missing/candidates.json')['tasks']),
              discovery_issues=discovery_issues, files=records,
              effective_sha256=digest(path)))
        files.append(provenance)
    return files


def load(folder):
    folder = Path(folder); provenance = read(folder/'provenance.json')
    root = folder.parent.parent
    for record in provenance['files']:
        if digest(root/record['path']) != record['sha256']:
            raise ValueError('Frozen missing-review input changed: '+record['path'])
    effective = folder/'effective_initial_geometry_results.json'
    if digest(effective) != provenance['effective_sha256']:
        raise ValueError('Frozen initial geometry changed')
    return read(effective), read(folder/'post_geometry_missing/candidates.json'), provenance
