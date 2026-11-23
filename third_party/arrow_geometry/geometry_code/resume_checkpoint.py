"""Freeze completed geometry and its crop frames for a style-only continuation."""
import copy
import hashlib
import json
from pathlib import Path

from checked_copy import copy_file, digest


def read(path):
    return json.loads(Path(path).read_text())


def freeze(source_run, root, ids):
    source_run=Path(source_run).resolve();root=Path(root).resolve();files=[]
    from PIL import Image
    for cid in ids:
        source=source_run/'geometry/cases'/cid
        destination=root/'resume_geometry'/cid;destination.mkdir(parents=True)
        if digest(source/'clean.png')!=digest(root/'localization/inputs'/cid/'source.png'):
            raise ValueError('Resume source image mismatch: '+cid)
        if digest(source/'reconstruction_handoff.json')!=digest(root/'localization/cases'/cid/'reconstruction_handoff.json'):
            raise ValueError('Resume candidate handoff mismatch: '+cid)
        records=[]
        def save(src,dst):
            before=digest(src);dst.parent.mkdir(parents=True,exist_ok=True);copy_file(src,dst)
            if before!=digest(dst) or before!=digest(src):raise ValueError('Resume input changed: '+str(src))
            records.append(dict(source=str(src),path=str(dst.relative_to(root)),sha256=before))
            files.append(dst)
        for name in ('initial_geometry_results.json','before_reconciliation.json','geometry_results.json',
                     'post_geometry_missing/audit.json','reconciliation/audit.json'):
            save(source/name,destination/name)
        for name in ('fatal.json','before_dedup_render_error.json'):
            if (source/name).exists():save(source/name,destination/'previous_failure'/name)
        final=copy.deepcopy(read(destination/'geometry_results.json'));crops={}
        for group in final:
            for row in group.get('rows',[]):
                if not row.get('local_input'):continue
                original=Path(row['local_input']).resolve()
                if str(original) not in crops:
                    folder=destination/'crops'/hashlib.sha256(str(original).encode()).hexdigest()[:16]
                    target=folder/'clean.png';save(original,target)
                    with Image.open(target) as im:im.load()
                    transform=original.parent/'transform.json'
                    if transform.exists():save(transform,folder/'transform.json')
                    elif not row.get('geometry_frame'):raise ValueError('Missing crop transform: '+str(original))
                    crops[str(original)]=str(target)
                row['local_input']=crops[str(original)]
        effective=destination/'effective_geometry_results.json'
        effective.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n');files.append(effective)
        provenance=destination/'provenance.json'
        provenance.write_text(json.dumps(dict(source_run=str(source_run),case=cid,
            resume_at='after_dedup_render',fresh_geometry_calls=False,fresh_missing_calls=False,
            fresh_reconciliation_calls=False,paths=sum(len(g.get('rows',[])) for g in final),files=records),
            ensure_ascii=False,indent=2)+'\n');files.append(provenance)
    return files


def load(folder):
    folder=Path(folder)
    initial=read(folder/'initial_geometry_results.json')
    before=read(folder/'before_reconciliation.json')
    final=read(folder/'effective_geometry_results.json')
    issues=[]
    for group in initial:issues.extend(group.get('issues',[]))
    for name in ('post_geometry_missing/audit.json','reconciliation/audit.json'):
        issues.extend(read(folder/name).get('issues',[]))
    return initial,before,final,issues
