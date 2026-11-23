"""Compare native-rendered font candidates against source ink, not OCR labels."""
from copy import deepcopy
from PIL import Image
from ..common import save


def select(rows, source, folder, slots, opt):
    from . import FONTS
    from .layout import native_metrics, resolve_font
    from .registration import register
    candidates=[];groups={}
    for row in rows:
        seen=set();group=[]
        for family in (row['font'], *FONTS):
            resolved=resolve_font(family,row['bold'],row['italic'])['path']
            if resolved in seen:continue
            seen.add(resolved)
            candidate={**deepcopy(row),'font':family,'id':f'{row["id"]}.F{len(group):02d}'}
            group.append(candidate);candidates.append(candidate)
        groups[row['id']]=group
    metrics=native_metrics(candidates,folder/'candidates',slots,opt)
    selected=[];measurements={};audit=[]
    with Image.open(source) as image:
        for row in rows:
            scores=[]
            for candidate in groups[row['id']]:
                _,_,record=register(candidate,metrics[candidate['id']],image)
                score=max((c['score'] for c in record.get('candidates',[])),default=0.)
                scores.append({'font':candidate['font'],'id':candidate['id'],
                               'score':score,'status':record['status']})
            best=max(scores,key=lambda r:r['score']);chosen=scores[0]
            # An improvement must be both absolute and relative. Ambiguous,
            # tiny, low-contrast and structurally mismatched crops retain OCR.
            if (best['status']=='accepted' and best['score']>=.65
                    and best['score']>=chosen['score']+.04):chosen=best
            selected.append({**row,'font':chosen['font']})
            measurements[row['id']]=metrics[chosen['id']]
            audit.append({'id':row['id'],'before':row['font'],'after':chosen['font'],
                          'candidates':scores,'changed':row['font']!=chosen['font']})
    save(folder/'selection.json',audit)
    return selected,measurements
