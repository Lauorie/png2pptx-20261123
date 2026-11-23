"""A complete source-measured node can own a duplicate round connector cap.

Only native oval caps touching one already recovered opaque ellipse qualify.
Source node pixels and independently measured geometry are required. Shaft,
directional heads, arrow algorithms and other nodes are outside this decision.
"""
import numpy as np
from PIL import Image
from shapely.geometry import Point,Polygon
from .handoff import strokes


def observe(scene,styles,arrows,source,excluded=()):
    with Image.open(source)as image:rgb=np.asarray(image.convert('RGB'))
    nodes=[]
    for key,p in scene['parts'].items():
        m=p.get('diagnostics_local',{}).get('source_primitive_registration',{})
        s=styles.get(key,{})
        if p.get('kind')!='ellipse' or m.get('perimeter_support',0)<.8 or not isinstance(s.get('fill'),str)or s.get('fill_opacity',1)!=1:continue
        ls=strokes(p['commands'])
        if len(ls)!=1:continue
        geom=Polygon(ls[0]);inner=geom.buffer(-max(1.,(s.get('width_px')or 0)/2))
        if inner.is_empty:continue
        b=np.asarray(inner.bounds);lo=np.maximum(0,np.floor(b[:2])).astype(int);hi=np.minimum(rgb.shape[1::-1],np.ceil(b[2:])).astype(int)
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];mask=np.array([inner.contains(Point(x,y))for x,y in zip(xx.ravel(),yy.ravel())]).reshape(xx.shape)
        pixels=rgb[lo[1]:hi[1],lo[0]:hi[0]][mask].astype(float)
        fill=np.array([int(s['fill'][i:i+2],16)for i in (1,3,5)])
        if len(pixels)<12 or np.mean(np.linalg.norm(pixels-fill,axis=1)<20)<.9:continue
        nodes.append((key,p,geom,fill))
    selections={};audit=[]
    for row in arrows:
        if row['id']in excluded:continue
        width=row.get('style',{}).get('width_px')or 1.
        for side,name in enumerate(('start','end')):
            head=(row.get('style',{}).get(name)or{}).get('type')or(row.get('heads')or['none','none'])[side]
            if head!='oval' or name not in row:continue
            endpoint=Point(row[name]);matches=[]
            for key,p,geom,fill in nodes:
                diameter=min(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])
                if diameter<4*width or geom.distance(endpoint)>min(3*width,.3*diameter):continue
                pigment=row.get('style',{}).get('color')
                if not isinstance(pigment,str):continue
                color=np.array([int(pigment[i:i+2],16)for i in (1,3,5)])
                if np.linalg.norm(color-fill)>80:continue
                matches.append((key,p))
            if len(matches)!=1:continue
            key,p=matches[0];selections.setdefault(row['id'],[]).append(side)
            audit.append({'arrow':row['id'],'side':side,'head':'oval','shape_part':key,
                'node_ref':p['object_id']+'/'+p['id'],'reason':'complete_source_node_owns_round_terminal',
                'shaft_preserved':True,'directional_head_unchanged':True})
    return selections,audit
