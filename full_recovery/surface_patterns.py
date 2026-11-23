"""Source-verified face patterns have one native owner, outside line voting.

Source support AND actual native visibility are required. A container bbox or
network prior cannot protect a pattern or delete an arrow on its own.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from .handoff import strokes,resample,json_part_id
from .source_line_coverage import visible_points,coverage


def resolve(scene,styles,rows,source,shape_pptx):
    if shape_pptx is None:return {'pattern_keys':[],'drop_arrow_names':[],'receipts':[]}
    preview=Path(shape_pptx).parent/'native.png'
    if not preview.exists():return {'pattern_keys':[],'drop_arrow_names':[],'receipts':[]}
    with Image.open(preview) as im:native=np.asarray(im.convert('RGB'))
    rgb=np.asarray(source.convert('RGB'))
    if native.shape!=rgb.shape:return {'pattern_keys':[],'drop_arrow_names':[],'receipts':[]}
    from shape_integration.styling.geometry import footprint
    from .compound_routes import recognize
    geoms={};patterns=[]
    faces=[k for k,p in scene['parts'].items() if p.get('status')=='ok' and p.get('kind') in ('rectangle','rounded_rectangle','ellipse')
           and styles.get(k,{}).get('fill') is not None and styles[k].get('fill_opacity',1)==1]
    for key,p in scene['parts'].items():
        paint=styles.get(key,{})
        commands=p.get('commands',[])
        if (p.get('status')!='ok' or p.get('kind')!='path' or paint.get('fill') or not paint.get('stroke')
                or any(c[0]=='Z' for c in commands)):continue
        paths=strokes(commands)
        if len(paths)<3 or recognize(paths,paint.get('width_px') or 1.) is not None:continue
        vectors=np.concatenate([np.diff(line,axis=0) for line in paths])
        lengths=np.linalg.norm(vectors,axis=1);valid=lengths>1e-6
        if not valid.any():continue
        vectors,lengths=vectors[valid],lengths[valid]
        angles=np.arctan2(vectors[:,1],vectors[:,0]);anchor=angles[np.argmax(lengths)]
        delta=np.abs(np.sin(2*(angles-anchor)))
        axial_support=float(lengths[delta<np.sin(np.deg2rad(20))].sum()/lengths.sum())
        if axial_support<.9:continue
        geom,_=footprint(p);owners=[]
        for face in faces:
            if scene['parts'][face]['object_id']!=p['object_id']:continue
            if face not in geoms:geoms[face]=footprint(scene['parts'][face])[0]
            if geoms[face].area>0 and geoms[face].buffer(.5).covers(geom):owners.append(face)
        if not owners:continue
        face=min(owners,key=lambda k:geoms[k].area);surface=geoms[face]
        nodes=[]
        for k,node in scene['parts'].items():
            if k==face or node.get('status')!='ok' or node.get('kind') not in ('ellipse','rectangle','rounded_rectangle'):continue
            b=np.asarray(node['bbox_2d'],float)
            if np.prod(b[2:]-b[:2])<.2*surface.area:nodes.append(b)
        ends=np.array([[line[0],line[-1]] for line in paths])
        if nodes:
            boxes=np.asarray(nodes);attached=np.any(np.all((ends[:,:,None]>=boxes[None,None,:,:2]-2)&
                (ends[:,:,None]<=boxes[None,None,:,2:]+2),axis=3),axis=2)
            if attached.any(1).mean()>=.25:continue
        points=resample(paths,count=12)
        if not len(points):continue
        colour=np.array([int(paint['stroke'][i:i+2],16) for i in (1,3,5)])
        radius=max(1,min(3,int(np.ceil((paint.get('width_px') or 1)/2+1))))
        offsets=np.array([(a,b) for a in range(-radius,radius+1) for b in range(-radius,radius+1)])
        xy=np.rint(points[:,None]+offsets).astype(int)
        xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
        original=(np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]].astype(float)-colour,axis=2)<85).any(1)
        visible=(np.linalg.norm(native[xy[:,:,1],xy[:,:,0]].astype(float)-colour,axis=2)<85).any(1)
        if original.mean()<.85 or visible.mean()<.85:continue
        patterns.append({'key':key,'face':face,'paths':paths,'style':paint,'points':points,
            'source_support':float(original.mean()),'native_visibility':float(visible.mean()),'axial_support':axial_support})
    dropped=[];receipts=[]
    for row in rows:
        if row['layer']!='arrows' or any(h not in (None,'none') for h in row.get('heads',[])):continue
        points=visible_points(row,rgb)
        if len(points)<max(8,.35*len(row['points'])):continue
        for pattern in patterns:
            p=row['points'];q=scene['parts'][pattern['face']]['bbox_2d']
            if np.any(p.min(0)<np.array(q[:2])-1) or np.any(p.max(0)>np.array(q[2:])+1):continue
            supported=coverage(points,pattern,rgb)
            if supported.mean()<.95:continue
            # Validate visibility in the delivered native render as well.
            xy=np.rint(points).astype(int);colour=np.array([int(pattern['style']['stroke'][i:i+2],16) for i in (1,3,5)])
            if (np.linalg.norm(native[xy[:,1],xy[:,0]].astype(float)-colour,axis=1)<100).mean()<.85:continue
            dropped.append(row['key']);receipts.append({'arrow_id':row['key'],'native_pattern_key':pattern['key'],
                'source_visible_samples':len(points),'replacement_coverage':float(supported.mean()),
                'source_support':pattern['source_support'],'native_visibility':pattern['native_visibility']})
            break
    return {'pattern_keys':[p['key'] for p in patterns],'drop_arrow_names':sorted(set(dropped)),'receipts':receipts}
