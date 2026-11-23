"""Optional pixel evidence and parallel physical-target geometry requests.

Evidence retrieval never assigns mask ownership or changes fitted points.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from transport.common import read_json as read, write_json as write


def related_masks(route, candidates, include_network=False):
    if route.get('identity_kind')=='network' and not include_network:return []
    boxes=[route['target_bbox'],*[b['bbox'] for b in route.get('source_boxes',[])]]
    found=[]
    for c in candidates:
        h,w=c['mask'].shape
        if any(c['mask'][max(0,int(np.floor(b[1]))):min(h,int(np.ceil(b[3]))),
                         max(0,int(np.floor(b[0]))):min(w,int(np.ceil(b[2])))].any() for b in boxes):
            found.append(c)
    return found


def independent(answer, proposals, candidates, gid, crop, out, model, locks, config, groups):
    """Keep semantically connected components together; fit others concurrently."""
    import geometry_routing
    import mask_vectors
    out=Path(out)
    def work(item):
        i,group=item
        folder=out/'geometry_components'/f'C{i:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'input').exists():(folder/'input').symlink_to((out/'input').resolve(),target_is_directory=True)
        sub=copy.deepcopy(answer);sub['routes']=copy.deepcopy(group)
        cfg={**config,'bbox_geometry_batch':'joint','geometry_namespace':f'C{i:03d}:'}
        result=geometry_routing.recover(sub,proposals,candidates,gid,crop,folder,model,locks,cfg)
        return result,read(folder/'routing.json'),read(folder/'head_calls.json')['calls']
    with ThreadPoolExecutor(max_workers=min(len(groups),config.get('task_workers',8))) as pool:
        results=list(pool.map(work,enumerate(groups,1)))
    rows=[];supports={};issues=[];calls=0;records=[];heads=0
    indices={gid+':'+r['id']:i for i,r in enumerate(answer['routes'],1)}
    for (rr,ss,ii,cc),routing,hh in results:
        for row in rr:row['identity_index']=indices[row.get('geometry_owner_id',row['id'])]
        rows.extend(rr);supports.update(ss);issues.extend(ii);calls+=cc;records.extend(routing);heads+=hh
    rows.sort(key=lambda r:r['identity_index']);mask_vectors.assign_parents(rows,gid)
    write(out/'routing.json',records)
    write(out/'head_calls.json',dict(calls=heads,ordinary_mask_extra_calls=0))
    write(out/'geometry_partition.json',dict(mode='independent_components',
        components=[[r['id'] for r in group] for group in groups],parallel=True))
    return rows,supports,issues,calls
