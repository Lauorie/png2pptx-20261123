"""Conservative physical-occurrence deduplication, never text-only dedup."""
from copy import deepcopy
import unicodedata

from ..common import intersect


def key(row):
    return ''.join(unicodedata.normalize('NFKC', row['text']).split())


def same_occurrence(a, b, threshold=.65):
    if abs((a.get('rotation_deg', 0)-b.get('rotation_deg', 0)+180)%360-180)>5:
        return False
    ka,kb=key(a),key(b)
    if not ka or not kb:return False
    boxes=[a['bbox_2d'], b['bbox_2d']]
    areas=[max(1e-9,(v[2]-v[0])*(v[3]-v[1])) for v in boxes]
    overlap=intersect(*boxes)
    iou=overlap/max(1e-9,sum(areas)-overlap)
    if a.get('kind')=='formula' or b.get('kind')=='formula':
        return (a.get('kind')==b.get('kind') and a.get('latex')==b.get('latex')
                and ka==kb and iou>=threshold)
    if ka==kb:return iou>=threshold or (overlap/min(areas)>=.95 and overlap/max(areas)>=.5)
    # Only a long literal underscore transcription variation is tolerated.
    # A prose label may encode the same underscore as a subscript run. Only
    # long multiword labels qualify; never merge mathematical expressions,
    # superscripts, short labels, changed values, or substrings.
    compatible=lambda r:all(v.get('script','normal') in ('normal','sub') for v in r.get('runs',[]))
    return (compatible(a) and compatible(b) and len(a['text'].split())>=2
            and len(b['text'].split())>=2 and min(len(ka),len(kb))>=8
            and ka.replace('_','')==kb.replace('_','')
            and overlap/min(areas)>=.95 and overlap/max(areas)>=.5)


def deduplicate(rows, threshold=.65):
    output=[];audit=[]
    for incoming in rows:
        row=deepcopy(incoming)
        matches=[i for i,old in enumerate(output) if same_occurrence(old,row,threshold)]
        if len(matches)!=1:
            output.append(row);continue
        index=matches[0];old=output[index]
        # More recent local reviews supersede global evidence. On a tie, the
        # later entry is authoritative; preserve the first stable occurrence ID.
        winner=row if len(row.get('observations',[]))>=len(old.get('observations',[])) else old
        chosen=deepcopy(winner);chosen['id']=old['id']
        chosen['observations']=old.get('observations',[])+row.get('observations',[])
        chosen['deduplicated_ids']=list(dict.fromkeys(old.get('deduplicated_ids',[])+row.get('deduplicated_ids',[])+[row['id']]))
        output[index]=chosen
        audit.append({'kept_id':old['id'],'removed_id':row['id'],
                      'selected_evidence_id':winner['id'],'reason':'same_physical_text_occurrence'})
    return output,audit
