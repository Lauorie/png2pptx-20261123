"""Shared native-mask evidence, rendered once as a contextual coverage layer."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev
from transport.common import write_json as write,sha256 as sha


def build(observations,out):
    rows=[]
    for obs in sorted(observations,key=lambda x:x['group']):
        for c in obs['candidates']:
            if c.get('native_area',0)<=0 or 'bbox_global' not in c:continue
            rows.append(dict(observation=obs['group'],id=c['id'],path=c['path'],sha256=sha(c['path']),
                view=obs['view'],bbox_global=c['bbox_global'],owner=c.get('owner')))
    write(Path(out)/'shared_observations.json',dict(masks=rows,creates_targets=False))
    return rows


def overlay(crop,view,selected,gid,pool,out,layer='proposals.png'):
    out=Path(out);coverage=np.zeros((crop.height,crop.width),bool);used=[]
    owned_ids={r['id'] for r in selected}
    for r in pool:
        if r['observation']==gid and r['id'] in owned_ids:continue
        if ev.gap(r['bbox_global'],view['bbox_global'])>0:continue
        m=ev.project(r,view)
        if m.any():coverage|=m;used.append({k:v for k,v in r.items() if k!='view'})
    owned=np.logical_or.reduce([r['mask'] for r in selected]) if selected else np.zeros_like(coverage)
    extra=coverage&~owned
    # A pixel is blended once regardless of how many overlapping observations
    # contain it. Own M/D labels and model geometry remain unchanged.
    original=Image.open(out/'input'/layer).convert('RGB');array=np.asarray(original).copy()
    array[extra]=np.rint(.65*array[extra]+.35*np.array([0,185,215])).astype('uint8')
    original.save(out/'input'/('own_'+layer));Image.fromarray(array).save(out/'input'/layer)
    Image.fromarray(extra.astype('uint8')*255).save(out/'input/context_coverage.png')
    record=dict(observations=len(used),additional_pixels=int(extra.sum()),sources=used,
        source_geometry_unchanged=True,creates_targets=False)
    write(out/'shared_evidence.json',record);return record


def overview(view,pool,out):
    """Ablation: expose full-figure SAM coverage without declaring recovery."""
    out=Path(out);image=Image.open(out/'input/context.png').convert('RGB');image.save(out/'input/own_context.png')
    sx=image.width/view['global_size'][0];sy=image.height/view['global_size'][1]
    matrix=np.array([[sx,0,.5*sx-.5],[0,sy,.5*sy-.5],[0,0,1.]])
    small={**ev.identity_view(view['global_size']),'size':list(image.size),'global_to_view':matrix.tolist(),'view_to_global':np.linalg.inv(matrix).tolist()}
    coverage=np.zeros((image.height,image.width),bool)
    for r in pool:coverage|=ev.project(r,small)
    array=np.asarray(image).copy();array[coverage]=np.rint(.65*array[coverage]+.35*np.array([0,185,215])).astype('uint8')
    result=Image.fromarray(array);ImageDraw.Draw(result).rectangle(ev.crop_in_context(view,result.size),outline='#c00080',width=3)
    result.save(out/'input/context.png')
    write(out/'shared_context_overview.json',dict(pixel_coverage=int(coverage.sum()),creates_targets=False,coverage_is_not_recovery=True))
