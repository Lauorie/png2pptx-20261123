"""Fallible endpoint diagnostics shared by geometry review and composition."""
from collections import defaultdict
from pathlib import Path
import numpy as np


def endpoint_node(point, nodes, fallback_nodes=()):
    """Resolve one endpoint; an ambiguous other end must not erase this link."""
    for group in (nodes, fallback_nodes):
        scores=[]
        for key,center,radius in group:
            score=float(np.linalg.norm((np.asarray(point)-center)/radius))
            if score<=1.65:scores.append((score,key))
        if scores:break
    scores.sort()
    if not scores or (len(scores)>1 and scores[1][0]-scores[0][0]<.2):return None
    return scores[0][1]


def node_pair(points, nodes, fallback_nodes=()):
    result=[endpoint_node(point,nodes,fallback_nodes) for point in np.asarray(points)[[0,-1]]]
    return tuple(sorted(result)) if all(result) and result[0]!=result[1] else None


def node_catalog(parts):
    nodes=[];boxes=[]
    for key,part in parts.items():
        bounds=np.asarray(part.get('bbox_2d',[]),float);kind=part.get('kind')
        if part.get('status')!='ok' or kind not in ('ellipse','rectangle','rounded_rectangle') or bounds.shape!=(4,):continue
        radius=(bounds[2:]-bounds[:2])/2
        if min(radius)<=2 or (kind=='ellipse' and max(radius)>1.6*min(radius)):continue
        (nodes if kind=='ellipse' else boxes).append((key,(bounds[:2]+bounds[2:])/2,radius))
    # A panel enclosing multiple distinct nodes is context, not an endpoint.
    # Keep concentric inner/outer outlines of one node together: independently
    # fitted branches may attach to either border of that same node.
    leaf_boxes=[]
    for key,center,radius in boxes:
        children=[c for k,c,r in nodes+boxes if k!=key and
                  np.prod(radius)>np.prod(r) and np.all(abs(c-center)+r<=radius)]
        if len(children)>=2 and any(np.linalg.norm(a-b)>2 for a in children for b in children):continue
        leaf_boxes.append((key,center,radius))
    return nodes,leaf_boxes


def fitted_connections(parts):
    """Inspect actual endpoints, never semantic part names; do not delete."""
    nodes,boxes=node_catalog(parts);routes=[];repeated=defaultdict(list)
    for key,part in parts.items():
        commands=part.get('commands',[])
        if part.get('status')!='ok' or not commands or any(c[0] not in ('M','L') for c in commands):continue
        strokes=[]
        for command in commands:
            if command[0]=='M':strokes.append([])
            if strokes:strokes[-1].append(command[1:3])
        for index,stroke in enumerate(strokes):
            if len(stroke)<2:continue
            points=np.asarray(stroke,float);delta=points[-1]-points[0];length=np.linalg.norm(delta)
            if length<=0:continue
            normal=np.array([-delta[1],delta[0]])/length
            if np.max(abs((points-points[0])@normal))>.5:continue
            pair=node_pair(points,nodes,boxes)
            if not pair:continue
            routes.append({'part':key,'stroke':index,'nodes':list(pair)})
            repeated[pair].append({'part':key,'stroke':index})
    return {'routes':routes,'same_endpoint_routes':[
        {'nodes':list(pair),'routes':rows} for pair,rows in repeated.items() if len(rows)>1]}


def connection_views(local,job,folder,provider=None):
    """Magnify ambiguous routes inside the existing critic call; no new agent.

    SAM masks for individual nodes are selected using the existing quality
    checks; matching the whole compound would hide this small-node evidence.
    """
    from PIL import Image,ImageDraw,ImageFont
    from ..common import read,save
    from .views import FONT
    folder=Path(folder);parts={f'F{i+1}':p for i,p in enumerate(local['parts'])}
    diagnostics=fitted_connections(parts);images=[];records=[]
    original={p['id']:p for p in read(folder/'fit_original.json')['parts']}
    with Image.open(folder/'clean.png') as im:clean=im.convert('RGB')
    font=ImageFont.truetype(FONT,max(14,clean.width//80))
    for index,group in enumerate(diagnostics['same_endpoint_routes'][:3]):
        bounds=np.asarray([parts[n]['bbox_2d'] for n in group['nodes']])
        pad=max((bounds[:,2:]-bounds[:,:2]).max(),20)
        lo=np.maximum(0,np.floor(bounds[:,:2].min(0)-pad)).astype(int)
        hi=np.minimum(clean.size,np.ceil(bounds[:,2:].max(0)+pad)).astype(int)
        box=(*lo,*hi);fit=clean.copy();draw=ImageDraw.Draw(fit)
        for route_index,(route,color) in enumerate(zip(group['routes'],['#d10070','#007ac2','#b75b00']*len(group['routes']))):
            part=parts[route['part']];strokes=[]
            for command in part['commands']:
                if command[0]=='M':strokes.append([])
                strokes[-1].append(tuple(command[1:3]))
            points=strokes[route['stroke']]
            draw.line(points,fill=color,width=max(2,clean.width//900))
            t=(route_index+1)/(len(group['routes'])+1)
            anchor=(1-t)*np.asarray(points[0])+t*np.asarray(points[-1])
            draw.text(tuple(anchor),route['part'],font=font,fill=color,stroke_width=1,stroke_fill='white')
        nearby={k:p for k,p in parts.items() if p.get('status')=='ok'
                and p.get('kind') in ('ellipse','rectangle','rounded_rectangle')
                and np.all(np.asarray(p['bbox_2d'])[2:]>lo) and np.all(np.asarray(p['bbox_2d'])[:2]<hi)}
        for label,part in nearby.items():
            b=part['bbox_2d'];draw.text((b[0],b[1]-font.size),label,font=font,fill='#00702a',stroke_width=1,stroke_fill='white')
        panels=[('CLEAN source',clean),('fitted route annotations',fit)];sam=[]
        if provider is not None and hasattr(provider,'select_object'):
            selected={}
            for part in nearby.values():
                source=original.get(part['id'])
                if source:
                    for g in provider.select_object({'id':job['object_id'],'bbox_2d':source['bbox_2d']}):selected[g['id']]=g
            if selected:
                path=folder/f'connection_{index}_sam_full.png'
                sam=provider.render(folder/'clean.png',list(selected.values()),job['frame']['source_crop_half_open'],path,masks=True)
                with Image.open(path) as im:panels.append(('SAM node contours, fallible',im.convert('RGB')))
        for label,panel in panels:
            tile=panel.crop(box);scale=min(2.,900/max(tile.size))
            tile=tile.resize(tuple(max(1,round(v*scale)) for v in tile.size),Image.Resampling.LANCZOS)
            path=folder/f'connection_{index}_{len(images)}.png';tile.save(path)
            images.append((f'Connection detail {index+1}: {label}',path))
        records.append({'routes':group,'crop':list(map(int,box)),'sam':sam})
    save(folder/'connection_evidence.json',{'diagnostics':diagnostics,'details':records,
                                          'extra_model_calls':0,'automatic_geometry_changes':False})
    return images
