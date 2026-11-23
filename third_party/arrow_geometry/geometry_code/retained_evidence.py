"""Reuse attributed LOCAL SAM evidence when a repair crop has no candidates.

Projection is only a coordinate conversion. It neither creates identities nor
changes model geometry. Missing/mismatched source frames are reported, not guessed.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import evidence as ev
from transport.common import read_json as read,write_json as write,sha256 as sha


def load(rows,view,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grouped={};issues=[]
    for row in rows:
        local=row.get('local_input')
        if not local:continue
        frame=Path(local).with_name('transform.json')
        for source in row.get('source_mask_paths',[]):
            path=Path(source)
            try:
                old=read(frame)
                if old['global_size']!=view['global_size']:raise ValueError('Global coordinate frame sizes differ')
                with Image.open(path) as im:
                    if list(im.size)!=old['size']:raise ValueError('Native mask dimensions differ from its recorded local frame')
                key=(sha(path),sha(frame))
                if key not in grouped:grouped[key]=dict(path=path,frame=frame,view=old,rows=[],short_ids=[])
                g=grouped[key]
                if row['id'] not in g['rows']:g['rows'].append(row['id'])
                sid=row.get('short_id',row['id'])
                if sid not in g['short_ids']:g['short_ids'].append(sid)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                issues.append(dict(route=row['id'],path=str(path),frame=str(frame),error=str(exc)))
    result=[];records=[]
    for (digest,frame_digest),g in grouped.items():
        try:
            m=ev.project(dict(path=g['path'],view=g['view']),view)
            if not m.any():continue
            mid=f'M{len(result)+1}';dest=out/(mid+'.png')
            Image.fromarray(m.astype('uint8')*255).save(dest)
            origin=dict(source_path=str(g['path']),source_sha256=digest,source_frame=str(g['frame']),source_frame_sha256=frame_digest,
                        existing_ids=g['rows'],existing_short_ids=g['short_ids'])
            result.append(dict(id=mid,mask=m,path=str(dest.resolve()),origin='retained_local_sam',native_area=int(m.sum()),scores={},retained_from=origin))
            records.append(dict(id=mid,path=str(dest.resolve()),sha256=sha(dest),projected_pixels=int(m.sum()),**origin))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            issues.append(dict(path=str(g['path']),error=str(exc)))
    write(out/'manifest.json',dict(reason='current repair SAM has no selected mask',view=view,masks=records,issues=issues,
                                  creates_new_targets=False,new_sam_calls=0,geometry_modified=False))
    return result
