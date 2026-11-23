"""Bounded source/native comparison inside composition ownership reconciliation.

Geometry only nominates candidates. Source evidence decides complete identity
and which existing native object survives. No geometry or head edits occur.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageChops
from shapely.geometry import Polygon, LineString

from .compose import selection_subset, subset, merge
from .handoff import strokes, resample, distances, arrow_strokes, json_part_id


def filled_terminals(scene,styles,key,ends,width,pigments):
    """Bind round terminal paint to its same-owner solid shaft for review."""
    owner=scene['parts'][key]['object_id'];found=[]
    for other,part in scene['parts'].items():
        paint=styles.get(other,{}).get('fill')
        if (other==key or part.get('object_id')!=owner or part.get('status')!='ok'
                or part.get('kind')!='ellipse' or not isinstance(paint,str)
                or min(np.linalg.norm(colour(paint)-colour(c)) for c in pigments)>55):continue
        b=np.asarray(part['bbox_2d'],float);diameter=max(b[2:]-b[:2]);center=(b[:2]+b[2:])/2
        if not 2<=diameter<=max(10.,3*width):continue
        side=int(np.linalg.norm(center-ends[0])>np.linalg.norm(center-ends[1]))
        if np.linalg.norm(center-ends[side])<=max(2.,.75*diameter):found.append((side,other,b))
    return [r for r in found if sum(q[0]==r[0] for q in found)==1]


def filled_routes(scene, styles, arrow_rows, claimed):
    """Nominate filled shafts/silhouettes only against a real matching arrow.

    Ordinary faces stay outside line ownership. A small polygon alone is not
    an arrow: its colour, extent and painted corridor must match an existing
    straight arrow. Native geometry is never converted to this comparison axis.
    """
    result=[]
    for key,p in scene['parts'].items():
        s=styles.get(key,{})
        fill=s.get('fill')
        pigments=[fill] if isinstance(fill,str) else [fill.get('start_color'),fill.get('end_color')] if isinstance(fill,dict) else []
        pigments=[v for v in pigments if isinstance(v,str)]
        if key in claimed or p.get('status')!='ok' or p.get('kind') not in ('path','rectangle') or not pigments:continue
        if p.get('native_parameters_original') and p.get('kind')!='rectangle':continue
        cs=p.get('commands',[])
        if not cs or cs[-1][0]!='Z' or any(c[0] not in ('M','L','Z') for c in cs):continue
        vertices=np.asarray([c[1:3] for c in cs if c[0] in ('M','L')],float)
        if len(vertices)>1 and np.linalg.norm(vertices[0]-vertices[-1])<1e-5:vertices=vertices[:-1]
        if not 3<=len(vertices)<=12:continue
        poly=Polygon(vertices)
        if not poly.is_valid or poly.area<=0:continue
        matches=[]
        for a in arrow_rows:
            pts=a['points'];axis=pts[-1]-pts[0];length=np.linalg.norm(axis)
            paint=a['style'].get('color') or a['style'].get('stroke')
            if not isinstance(paint,str) or min(np.linalg.norm(colour(c)-colour(paint)) for c in pigments)>100:continue
            if a.get('closed'):
                # A block arrow is a closed silhouette in both exporters.
                # Nominate the whole native object; never suppress its border
                # segments or turn an ordinary convex node/face into a route.
                other=Polygon(pts)
                if (not other.is_valid or other.area<=0 or
                    poly.area>=.98*poly.convex_hull.area):continue
                if not .4<=poly.area/other.area<=2.5:continue
                _,vectors=np.linalg.eigh(np.cov(vertices.T));axis=vectors[:,-1]
                _,other_vectors=np.linalg.eigh(np.cov(pts.T))
                if abs(float(axis@other_vectors[:,-1]))<.9:continue
                center=vertices.mean(0);along=(vertices-center)@axis
                across=(vertices-center)@np.array([-axis[1],axis[0]])
                span=float(np.ptp(along));width=float(np.ptp(across))
                if span<4 or width>.8*span:continue
                # As with open shafts, tolerate an offset within the painted
                # width when nominating. The source comparison resolves which
                # paths really exist; overlap alone never deletes a candidate.
                other_width=float(np.ptp(pts@np.array([-axis[1],axis[0]])))
                if poly.buffer(.5*min(width,other_width)).intersection(other).area/min(poly.area,other.area)<.3:continue
                ends=center+np.outer([along.min(),along.max()],axis)
                matches.append((float(np.linalg.norm(center-pts.mean(0))),a['id'],ends,width))
                continue
            if length<4:continue
            # A similar long-axis projection does not make a distant filled
            # native face a recovery of this shaft. Require contact with its
            # actual stroke footprint before opening whole-object selection.
            axis/=length;normal=np.array([-axis[1],axis[0]])
            along=(vertices-pts[0])@axis;across=(vertices-pts[0])@normal
            span=float(np.ptp(along));width=float(np.ptp(across));pad=max(3.,.3*length)
            if np.quantile(distances(pts,[pts[[0,-1]]]),.95)>1.5:continue
            if not .55*length<=span<=1.65*length or abs(along.min())>pad or abs(along.max()-length)>pad:continue
            if width>max(5.,.65*span):continue
            # Wide solid rectangles are nodes/faces, not filled line shafts.
            start=pts[0]+along.min()*axis;end=pts[0]+along.max()*axis
            terminals=filled_terminals(scene,styles,key,np.array([start,end]),width,pigments)
            stroke=a.get('style',{}).get('width_px')or 1.
            # A compact convex face has no shaft topology. A pointed filled
            # glyph can instead compete with a directional arrow when the
            # corresponding end tapers to a tip. Containment alone is never
            # evidence that a face represents the enclosing node's edges.
            taper=False
            for side,head in enumerate(a.get('heads',())):
                if head in (None,'none','oval','diamond'):continue
                terminal=along.min() if side==0 else along.max()
                tip=across[np.abs(along-terminal)<=max(1.,stroke)]
                if len(tip) and np.ptp(tip)<=max(2.,2*stroke) and width>2*stroke:taper=True
            if poly.area/poly.convex_hull.area>.85 and span/max(width,1)<6 and not (terminals or taper):continue
            if np.max(np.abs(across))>max(4.,.4*length,width+2*stroke if taper else 0.):continue
            paint_radius=max(2.,stroke/2,width/2 if taper else 0.)
            if poly.distance(LineString(pts))>paint_radius:continue
            matches.append((float(np.mean(np.abs(across))),a['id'],np.array([start,end]),width))
        if not matches:continue
        _,_,ends,width=min(matches,key=lambda v:v[0])
        terminals=filled_terminals(scene,styles,key,ends,width,pigments);heads=['none','none']
        for side,other,b in terminals:heads[side]='oval'
        refs=[p['object_id']+'/'+json_part_id(key)]+[p['object_id']+'/'+json_part_id(other) for side,other,b in terminals]
        result.append({'id':'shape:'+key,'layer':'shape','owner':p['object_id'],'key':key,'keys':[key,*[other for side,other,b in terminals]],
            'refs':refs,'attachment_refs':refs[1:],'points':resample([ends]),
            'heads':heads,'style':{**s,'stroke':pigments[0],'width_px':width},
            'length':float(np.linalg.norm(ends[1]-ends[0])),'filled_route':True,
            'competing_arrows':[m[1] for m in matches],'attachment_bounds':[vertices.tolist(),*[b.reshape(2,2).tolist() for side,other,b in terminals]]})
    return result


def attach_heads(rows, scene, styles):
    """Bind triangular heads by geometry, independently of their possibly wrong paint.

    A head is a complete native object, including hollow or miscoloured heads.
    It must never also enter the face-border segment suppression path.
    """
    attached=set()
    shafts=[r for r in rows if r['layer']=='shape' and not r.get('closed') and 'segment' not in r]
    for key,part in scene['parts'].items():
        commands=part.get('commands',[])
        if (part.get('status')!='ok' or part.get('native_parameters_original') or
                part.get('kind')!='path' or not commands or commands[-1][0]!='Z' or
                any(c[0] not in ('M','L','Z') for c in commands)):continue
        vertices=np.unique([c[1:3] for c in commands if c[0] in ('M','L')],axis=0)
        if len(vertices)!=3:continue
        paint=styles.get(key,{})
        if not paint.get('fill') and not paint.get('stroke'):continue
        center=vertices.mean(0);diameter=float(np.ptp(vertices,axis=0).max());matches=[];near_matches=[]
        for row in shafts:
            if row['owner']!=part['object_id'] or not 2<=diameter<=max(12.,.8*row['length']):continue
            points=row['points'];side=int(np.linalg.norm(center-points[0])>np.linalg.norm(center-points[-1]))
            end=points[-1 if side else 0];near=points[-2 if side else 1]
            direction=end-near;norm=np.linalg.norm(direction)
            if norm<1e-6 or np.linalg.norm(center-end)>.9*diameter:continue
            direction/=norm
            width=row['style'].get('width_px') or 1
            offset_limit=max(2.,2*width,.12*diameter)
            if abs(float(np.cross(direction,center-end)))>max(offset_limit,.75*diameter):continue
            axes=[]
            for i,tip in enumerate(vertices):
                base=np.delete(vertices,i,axis=0).mean(0);axis=tip-base
                alignment=abs(float(axis@direction))/max(np.linalg.norm(axis),1e-6)
                # The shaft crosses the base-to-tip axis, not a triangle corner.
                offset=abs(float(np.cross(direction,base-end)))
                if alignment>.92 and offset<=max(2.,2*width,.75*diameter):axes.append((alignment,tip,offset))
            if not axes and diameter<=max(12.,4*width):
                # Small fitted heads can be skewed. A shaft ending inside the
                # triangle (within one stroke) is physical attachment evidence;
                # its median need not align like an ideal symmetric arrowhead.
                edge=np.roll(vertices,-1,axis=0)-vertices
                signed=np.cross(edge,end-vertices)/np.maximum(np.linalg.norm(edge,axis=1),1e-8)
                tolerance=max(1.,width)
                if np.min(signed)>=-tolerance or np.max(signed)<=tolerance:
                    tip=vertices[np.argmax((vertices-end)@direction)]
                    near_matches.append((float(np.linalg.norm(center-end)),row,side,tip))
            if axes:
                _,tip,offset=max(axes,key=lambda v:v[0])
                match=(float(np.linalg.norm(center-end)),row,side,tip)
                if offset<=max(2.,2*width,.15*diameter) and abs(float(np.cross(direction,center-end)))<=offset_limit:matches.append(match)
                else:near_matches.append(match)
        if not matches and near_matches:
            near_matches.sort(key=lambda v:v[0])
            # A misplaced head still belongs to its sole nearby shaft. Do not
            # broadcast this relaxed nomination to several parallel routes.
            if len(near_matches)==1 or near_matches[1][0]-near_matches[0][0]>.25*diameter:
                matches=near_matches[:1]
        if not matches:continue
        # Nearby converging routes are not users of the same physical head.
        # Share only an identical terminal; otherwise bind to the closest
        # aligned shaft so deleting that connection also deletes its head.
        matches.sort(key=lambda v:v[0])
        best=matches[0];base=best[1]['points'];base_end=base[-1 if best[2] else 0]
        base_dir=base[-1]-base[-2] if best[2] else base[0]-base[1]
        base_dir/=max(np.linalg.norm(base_dir),1e-8)
        def same_terminal(match):
            _,row,side,_=match;points=row['points'];end=points[-1 if side else 0]
            direction=points[-1]-points[-2] if side else points[0]-points[1]
            direction/=max(np.linalg.norm(direction),1e-8)
            tolerance=max(1.5,min(row['style'].get('width_px') or 1,best[1]['style'].get('width_px') or 1))
            return np.linalg.norm(end-base_end)<=tolerance and direction@base_dir>.98
        matches=[best,*[m for m in matches[1:] if same_terminal(m)]]
        ref=part['object_id']+'/'+json_part_id(key)
        for _,row,side,tip in matches:
            row['keys'].append(key);row['refs'].append(ref)
            row.setdefault('attachment_refs',[]).append(ref);row['heads'][side]='triangle'
            row.setdefault('attachment_bounds',[]).append(vertices.tolist())
            endpoints=row.get('node_endpoints',row['points'][[0,-1]].copy())
            outward=row['points'][-1]-row['points'][-2] if side else row['points'][0]-row['points'][1]
            if (tip-endpoints[side])@outward>0:endpoints[side]=tip
            row['node_endpoints']=endpoints
        attached.add(key)
    return [r for r in rows if r['key'] not in attached and r.get('parent_key') not in attached]


def port_node(point, outward, nodes, boxes, uncertainty=2.):
    """Nominate a node across a short visible gap, without extending the line.

    Direct attachment remains preferred. Allow raster/fit uncertainty at the
    node boundary before requiring a ray intersection across a larger gap.
    A node behind the ray or an ambiguous pair does not establish identity.
    """
    from shape_integration.fitting.connectivity import endpoint_node
    direct=endpoint_node(point,nodes,boxes)
    if direct is not None:return direct
    outward=np.asarray(outward,float);length=np.linalg.norm(outward)
    if length<1e-8:return None
    outward/=length
    # Two independently fitted endpoints can miss a hard boundary by a pixel.
    # This only nominates a comparison; it never selects or deletes a route.
    expanded=[[(k,c,r+uncertainty) for k,c,r in group] for group in (nodes,boxes)]
    nearby=endpoint_node(point,*expanded)
    if nearby is not None:
        center=next(c for group in (nodes,boxes) for k,c,r in group if k==nearby)
        if (center-point)@outward>=0:return nearby
    for group in (nodes,boxes):
        scores=[]
        for key,center,radius in group:
            p=(point-center)/radius;v=outward/radius
            a=v@v;b=2*(p@v);c=p@p-1;disc=b*b-4*a*c
            if disc<0:continue
            entry=(-b-np.sqrt(disc))/(2*a)
            if 0<=entry<=2*min(radius):scores.append((entry/min(radius),key))
        scores.sort()
        if scores:
            return scores[0][1] if len(scores)==1 or scores[1][0]-scores[0][0]>=.2 else None
    return None


def candidates(scene, styles, arrows):
    rows=[]
    for key,p in scene['parts'].items():
        s=styles.get(key,{})
        if p.get('status')!='ok' or not s.get('stroke'):continue
        commands=p.get('commands',[])
        if not commands:continue
        if any(c[0]=='Z' for c in commands):
            contours=strokes(commands)
            native=p.get('native_parameters_original')
            curved_native=bool(native and (p.get('kind')!='rectangle' or native.get('preset')))
            if len(contours)==1 and (curved_native or
                    p.get('kind')=='path' and not native and not s.get('fill')):
                pts=resample(contours);length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
                if length>=12:
                    rows.append({'id':'shape:'+key,'layer':'shape','owner':p['object_id'],'key':key,'keys':[key],
                        'refs':[p['object_id']+'/'+json_part_id(key)],'points':pts,'heads':['none','none'],
                        'style':s,'length':length,'closed':True,
                        'immutable_border':bool(curved_native and s.get('fill'))})
            if native and (p.get('kind')!='rectangle' or native.get('preset')):continue
            if any(c[0] not in ('M','L','Z') for c in commands):continue
            vertices=np.asarray([c[1:3] for c in commands if c[0] in ('M','L')],float)
            if np.linalg.norm(vertices[0]-vertices[-1])<1e-5:vertices=vertices[:-1]
            if not 3<=len(vertices)<=32:continue
            delta=np.roll(vertices,-1,axis=0)-vertices
            for i in range(len(vertices)):
                route=vertices[[i,(i+1)%len(vertices)]];length=float(np.linalg.norm(route[1]-route[0]))
                if length<12:continue
                rows.append({'id':'shape:'+key+'::segment'+str(i),'layer':'shape','owner':p['object_id'],
                    'key':key+'::segment'+str(i),'parent_key':key,'keys':[],
                    'refs':[p['object_id']+'/'+json_part_id(key)],'segment':int(i),
                    'immutable_border':bool(native),
                    'filled_face':vertices.tolist() if s.get('fill') else None,
                    'points':resample([route]),'heads':['none','none'],'style':s,'length':length})
            continue
        if s.get('fill'):continue
        lines=strokes(commands)
        if len(lines)>1:
            from .compound_routes import recognize
            compound=recognize(lines,s.get('width_px') or 1.)
            if compound is not None:
                points=resample([compound.pop('points')])
                rows.append({'id':'shape:'+key,'layer':'shape','owner':p['object_id'],
                    'key':key,'keys':[key],'refs':[p['object_id']+'/'+json_part_id(key)],
                    'points':points,'paths':lines,'style':s,'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                    **compound})
                continue
            # A curved shaft and its short attached stroked head are one
            # native route. Keeping the head as an independent competitor
            # leaves an orphan mark when the shaft loses branch selection.
            if len(lines)==2 and any(c[0]=='C' for c in commands):
                lengths=[float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum()) for line in lines]
                body=int(np.argmax(lengths));head=1-body
                ends=np.asarray(lines[body])[[0,-1]];tips=np.asarray(lines[head])[[0,-1]]
                attached=float(np.linalg.norm(ends[:,None]-tips[None,:],axis=2).min())
                if lengths[head]<=.2*lengths[body] and attached<=max(1.,s.get('width_px') or 1.):
                    rows.append({'id':'shape:'+key,'layer':'shape','owner':p['object_id'],
                        'key':key,'keys':[key],'refs':[p['object_id']+'/'+json_part_id(key)],
                        'points':resample([lines[body]]),'heads':['none','none'],'style':s,
                        'length':lengths[body],'attached_stroked_head':True})
                    continue
            if any(c[0] not in ('M','L','C') for c in commands) or sum(c[0]=='M' for c in commands)!=len(lines):continue
            from .compound_routes import terminal_groups
            grouped=set()
            for a,b,compound in terminal_groups(lines,s.get('width_px') or 1.):
                grouped.update((a,b));points=resample([compound.pop('points')])
                rows.append({'id':'shape:'+key+'::subpath'+str(a),'layer':'shape','owner':p['object_id'],
                    'key':key+'::subpath'+str(a),'parent_key':key,'keys':[],
                    'refs':[p['object_id']+'/'+json_part_id(key)],'subpath':a,'subpaths':[a,b],'subpath_count':len(lines),
                    'points':points,'paths':[lines[a],lines[b]],'style':s,
                    'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),**compound})
            # A compound can store many independent connections in one native
            # custom path. Each M-started route gets its own evidence/ownership;
            # deleting one never deletes its unrelated siblings.
            for index,line in enumerate(lines):
                if index in grouped:continue
                points=resample([line]);length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
                if length<4:continue
                rows.append({'id':'shape:'+key+'::subpath'+str(index),'layer':'shape','owner':p['object_id'],
                    'key':key+'::subpath'+str(index),'parent_key':key,'keys':[],
                    'refs':[p['object_id']+'/'+json_part_id(key)],'subpath':index,'subpath_count':len(lines),
                    'points':points,'heads':['none','none'],'style':s,'length':length})
            continue
        if not lines:continue
        points=resample(lines);length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length<4:continue
        refs=[p['object_id']+'/'+json_part_id(key)];heads=['none','none'];attachments=[]
        node_endpoints=points[[0,-1]].copy()
        for side, endpoint in enumerate((points[0],points[-1])):
            dots=[]
            for k,d in scene['parts'].items():
                ds=styles.get(k,{})
                if d.get('object_id')!=p['object_id'] or not isinstance(ds.get('fill'),str):continue
                # Triangles are attached below for simple AND compound paths,
                # without requiring their fill to match an already-correct shaft.
                if d.get('kind')!='ellipse':continue
                b=np.asarray(d['bbox_2d']);diameter=max(b[2:]-b[:2]);center=(b[:2]+b[2:])/2
                if not 2<=diameter<=max(10.,3*(s.get('width_px') or 1)):continue
                if ds['fill']!=s['stroke'] and np.linalg.norm(colour(ds['fill'])-colour(s['stroke']))>55:continue
                if np.linalg.norm(center-endpoint)>max(4.,diameter*.75):continue
                peers=[]
                for other,op in scene['parts'].items():
                    if other==key or op.get('object_id')!=p['object_id']:continue
                    cs=op.get('commands',[])
                    if not cs or any(c[0]=='Z' for c in cs):continue
                    ls=strokes(cs)
                    if len(ls)==1 and min(np.linalg.norm(center-ls[0][i]) for i in (0,-1))<=max(4.,diameter*.75):peers.append(other)
                if not peers:dots.append(k)
            if len(dots)==1:
                k=dots[0];heads[side]='oval';attachments.append(k)
                refs.append(p['object_id']+'/'+json_part_id(k))
        rows.append({'id':'shape:'+key,'layer':'shape','owner':p['object_id'],'key':key,'keys':[key,*attachments],
                     'refs':refs,'points':points,'node_endpoints':node_endpoints,'heads':heads,'style':s,'length':length})
    for r in arrows:
        if r.get('kind')=='block_arrow' and not r.get('beziers') and not all(k in r for k in ('start','end')):
            continue  # Some native presets expose no comparable source contour.
        ls=arrow_strokes(r)
        if len(ls)!=1:continue
        if r.get('kind')=='block_arrow' and r.get('closed') and np.linalg.norm(ls[0][0]-ls[0][-1])>1e-6:
            ls=[np.vstack([ls[0],ls[0][0]])]
        pts=resample(ls);length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
        heads=[(r.get('style',{}).get(e) or {}).get('type') or (r.get('heads') or ['none','none'])[i]
               for i,e in enumerate(('start','end'))]
        # Head labels differ between exporters (arrow/open/unknown/stealth).
        # Ownership compares the actual native render, not this vocabulary.
        if length<4:continue
        closed=np.linalg.norm(pts[0]-pts[-1])<2 and len(pts)>=4
        rows.append({'id':'arrow:'+r['id'],'layer':'arrows','key':r['id'],'keys':[],
                     'refs':[r['id']],'points':pts,'heads':heads,'style':r.get('style',{}),'length':length,'closed':bool(closed)})
    rows=attach_heads(rows,scene,styles)
    filled=filled_routes(scene,styles,[r for r in rows if r['layer']=='arrows'],
                         {k for r in rows for k in r['keys']})
    # A filled route participates as one native object. Asking separately
    # about its border segments would produce incompatible ownership edits.
    filled_keys={r['key'] for r in filled}
    rows=[r for r in rows if r.get('parent_key') not in filled_keys]
    rows.extend(filled)
    from shape_integration.fitting.connectivity import endpoint_node,node_pair,node_catalog
    nodes,boxes=node_catalog(scene['parts'])
    # A terminal mark bound to a shaft is part of that implementation, not
    # an independent endpoint node. Otherwise a short arrow names the dot
    # while the complete shape shaft names the actual soma, escaping review.
    terminals={k for row in rows for k in row.get('keys',[])[1:]
               if scene['parts'].get(k,{}).get('kind')=='ellipse'}
    nodes=[row for row in nodes if row[0] not in terminals]
    boxes=[row for row in boxes if row[0] not in terminals]
    radii={key:float(max(radius)) for key,_,radius in nodes}
    radii.update({key:float(min(radius)) for key,_,radius in boxes})
    node_bounds={key:scene['parts'][key]['bbox_2d'] for key in radii}
    for row in rows:
        if row.get('closed'):continue
        points=row['points']
        # One reliable endpoint is enough to include a route in its network.
        # Pair identity is stricter and remains a separate nomination hint.
        # Face border segments are not node-link routes.
        if 'segment' not in row:
            row['network_nodes']=sorted({n for p in row.get('node_endpoints',points[[0,-1]])
                if (n:=endpoint_node(p,nodes,boxes)) is not None})
            # A native shaft may span several nodes while the other branch
            # stores only the visible gaps. Include that whole native object
            # in the same comparison; never let its hidden span leak back in.
            crossed=[key for key,center,radius in nodes+boxes
                     if distances(center[None],[points])[0]<=.7*min(radius)]
            if len(crossed)>=2:row['network_nodes']=sorted(set(row['network_nodes'])|set(crossed))
            if row['network_nodes']:row['node_radius_px']=max(radii[n] for n in row['network_nodes'])
        # Resolve each end independently, including mixed box/circle routes.
        # Prefer specific circular nodes over their enclosing group frames.
        pair=node_pair(row.get('node_endpoints',points[[0,-1]]),nodes,boxes)
        if not pair and 'segment' not in row:
            ends=row.get('node_endpoints',points[[0,-1]])
            uncertainty=max(2.,(row.get('style',{}).get('width_px') or 1)/2)
            ports=[port_node(ends[0],points[0]-points[1],nodes,boxes,uncertainty),
                   port_node(ends[1],points[-1]-points[-2],nodes,boxes,uncertainty)]
            if all(ports) and ports[0]!=ports[1]:pair=tuple(sorted(ports))
        if pair:row.update(endpoint_nodes=pair,node_radius_px=max(radii[n] for n in pair),
                           endpoint_regions=[node_bounds[n] for n in pair])
    return rows


def colour(value):return np.array([int(value[i:i+2],16) for i in (1,3,5)],float)


def shared_native_span(a, b, tolerance):
    """Length of a shared linear section of two complete native routes.

    A common corner or a transverse crossing is not a shared section. This
    only nominates whole native objects for source review; neither the points
    nor a route's unmatched external sections are cut at the overlap.
    """
    def longest(points, other):
        delta = np.diff(points, axis=0)
        segments = np.diff(other, axis=0)
        centers = (points[:-1] + points[1:]) / 2
        lengths = np.linalg.norm(delta, axis=1)
        squared = np.maximum(np.sum(segments * segments, axis=1), 1e-12)
        position = np.clip(np.sum((centers[:, None] - other[None, :-1]) *
                                  segments[None], axis=2) / squared, 0, 1)
        offset = centers[:, None] - (other[None, :-1] + position[:, :, None] * segments[None])
        distance = np.sum(offset * offset, axis=2)
        nearest = distance.argmin(axis=1)
        tangent = segments[nearest]
        alignment = abs(np.sum(delta * tangent, axis=1)) / np.maximum(
            lengths * np.linalg.norm(tangent, axis=1), 1e-12)
        supported = ((distance[np.arange(len(centers)), nearest] <= tolerance * tolerance)
                     & (alignment >= .9))
        directions = delta / np.maximum(lengths[:, None], 1e-12)
        other_directions = tangent / np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
        current = best = 0.
        previous = None
        for matches, length, direction, other_direction in zip(
                supported, lengths, directions, other_directions):
            # Numeric collinearity identifies the original straight legs of
            # the resampled native path. Nearby ends of two different curves
            # cannot accumulate into a shared leg merely by touching a node.
            continuous = (previous is not None
                and np.allclose(direction, previous[0], atol=1e-7, rtol=0)
                and np.allclose(other_direction, previous[1], atol=1e-7, rtol=0))
            current = (current + length if continuous else length) if matches else 0.
            best = max(best, current)
            previous = (direction, other_direction) if matches else None
        return best
    return min(longest(a, b), longest(b, a))


def nominate(rows,source_rgb=None):
    # Shape may store its head separately while arrows include the tip in the
    # route. Compare their complete visible extents without changing native
    # geometry; otherwise short arrows become two unrelated existence checks.
    normalized=[]
    for row in rows:
        heads=row.get('attachment_bounds',[])
        if not heads or row.get('filled_route'):
            normalized.append(row);continue
        points=row['points'].copy()
        points[[0,-1]]=row.get('node_endpoints',points[[0,-1]])
        width=row.get('style',{}).get('width_px') or 1
        for vertices in heads:
            vertices=np.asarray(vertices);center=vertices.mean(0)
            side=int(np.linalg.norm(center-points[0])>np.linalg.norm(center-points[-1]))
            axis=points[-1]-points[-2] if side else points[0]-points[1]
            normal=np.array([-axis[1],axis[0]])/max(np.linalg.norm(axis),1e-6)
            width=max(width,float(np.ptp(vertices@normal))/2)
        normalized.append({**row,'points':points,
            'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'style':{**row.get('style',{}),'width_px':width}})
    rows=normalized
    straight_ids={r['id'] for r in rows if not r.get('closed') and 'segment' not in r and
                  np.quantile(distances(r['points'],[r['points'][[0,-1]]]),.95)<1.}
    curved_ids={r['id'] for r in rows if 'segment' not in r and
                np.quantile(distances(r['points'],[r['points'][[0,-1]]]),.9)>
                max(3.,.1*r['length'])}
    from collections import Counter
    node_counts=Counter((tuple(r['endpoint_nodes']),r['layer']) for r in rows
                        if r.get('endpoint_nodes') and not r.get('closed') and 'segment' not in r)
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['layer']==b['layer'] and (a['layer']!='shape' or a.get('closed') or b.get('closed')
                    or any('segment' in r or r.get('filled_face') for r in (a,b))):continue
            filled=a if a.get('filled_route') else b if b.get('filled_route') else None
            if filled is not None:
                other=b if filled is a else a
                if other['id'] in filled['competing_arrows']:
                    pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],
                                  'p90_px':0.,'nomination':'matching_filled_route'})
                continue
            ap,bp=a['points'],b['points'];reverse=np.linalg.norm(ap[0]-bp[-1])+np.linalg.norm(ap[-1]-bp[0])<np.linalg.norm(ap[0]-bp[0])+np.linalg.norm(ap[-1]-bp[-1])
            # Appearance differences are precisely what branch selection must
            # resolve; colour/head mismatches must not hide overlapping routes.
            # Nominate using the visible stroke footprint as well as route
            # length. Two short, thick fits must not escape review merely
            # because their centre lines differ by a few source pixels.
            tolerance=max(2.,.045*min(a['length'],b['length']),
                          (a.get('style',{}).get('width_px') or 1)+(b.get('style',{}).get('width_px') or 1))
            ends=bp[::-1] if reverse else bp
            endpoint_error=max(np.linalg.norm(ap[0]-ends[0]),np.linalg.norm(ap[-1]-ends[-1]))
            if (a['layer']!=b['layer'] and a['id'] in straight_ids and b['id'] in straight_ids
                    and min(a['length'],b['length'])/max(a['length'],b['length'])>=.8
                    and endpoint_error<=min(max(10.,1.5*tolerance),.15*min(a['length'],b['length']))):
                # Use the same bounded localization uncertainty for both end
                # and shaft tests, including the coarse bounding-box test.
                # Only comparable complete straight spans qualify; short
                # fragments cannot join unrelated routes through this rule.
                tolerance=max(tolerance,endpoint_error)
            # Endpoint-node assignment is fallible. A distant divider and an
            # external connector cannot compete just because both touch the
            # same large boxes; apply the existing spatial gate first.
            if np.any(ap.max(0)<bp.min(0)-tolerance) or np.any(bp.max(0)<ap.min(0)-tolerance):continue
            # Curved routes can recover the same source loop with different
            # starts/attachments. Endpoint-node identity is fallible here.
            # Nominate by the whole contour; distinct visible source curves
            # are still protected before any ownership decision.
            curved=[r['id'] in curved_ids for r in (a,b)]
            if (a['layer']!=b['layer'] and all('segment' not in r for r in (a,b)) and
                    (all(curved) or all(not r.get('closed') for r in (a,b)) and any(curved))):
                lo=np.maximum(ap.min(0),bp.min(0));hi=np.minimum(ap.max(0),bp.max(0))
                intersection=float(np.maximum(hi-lo,0).prod())
                union=float(np.ptp(ap,axis=0).prod()+np.ptp(bp,axis=0).prod()-intersection)
                d=max(float(np.quantile(distances(ap,[bp]),.9)),float(np.quantile(distances(bp,[ap]),.9)))
                coverage=min(float(np.mean(distances(ap,[bp])<=tolerance)),
                             float(np.mean(distances(bp,[ap])<=tolerance)))
                if union>0 and intersection/union>=.5 and coverage>=.55:
                    pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],
                                  'p90_px':d,'shared_visible_span_fraction':coverage,
                                  'nomination':'overlapping_curved_routes'})
                    continue
                # One native polyline can contain sections from several
                # connections. Whole-route endpoint/coverage checks miss its
                # competition with separate shape routes. Couple actual
                # continuous overlap, not every object near a node or crop.
                shared = (shared_native_span(ap, bp, tolerance)
                          if all(not r.get('closed') for r in (a,b)) else 0.)
                if shared > 2 * tolerance:
                    pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],
                                  'p90_px':d,'shared_native_span':True,
                                  'shared_span_px':float(shared),
                                  'nomination':'overlapping_native_span'})
                    continue
            # An open visible arc may recover a closed outline behind a node.
            # Other closure mismatches still require a separate comparison.
            if a.get('closed',False)!=b.get('closed',False):continue
            same_nodes=bool(a.get('endpoint_nodes') and a.get('endpoint_nodes')==b.get('endpoint_nodes'))
            if same_nodes and all(node_counts[(tuple(a['endpoint_nodes']),layer)]==1 for layer in ('shape','arrows')):
                # Quality differences include wrong curvature and attachment
                # sides. A pixel-distance gate previously hid exactly these
                # competitors, leaving two versions of one connection alive.
                d=max(float(np.quantile(distances(ap,[bp]),.9)),float(np.quantile(distances(bp,[ap]),.9)))
                pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],
                              'p90_px':d,'nomination':'same_endpoint_nodes','unique_endpoint_pair':True})
                continue
            # A complete short connection can be part of a much longer native
            # route, including a return curve followed by a node chain. Its
            # terminal pair need not equal the long route's terminal pair.
            # Nominate a genuinely shorter span by shared node identities.
            # Comparable complete spans use the spatial checks below: two
            # crossing dividers can share the same large containing faces.
            # This only opens a SOURCE comparison; it authorizes no deletion.
            embedded=[(short,long) for short,long in ((a,b),(b,a))
                if short.get('endpoint_nodes') and
                set(short['endpoint_nodes'])<=set(long.get('network_nodes',()))]
            if (embedded and min(a['length'],b['length'])/max(a['length'],b['length'])<.8
                    and not a.get('closed') and
                    all('segment' not in r and not r.get('filled_face') for r in (a,b))):
                short,long=min(embedded,key=lambda pair:pair[0]['length'])
                d=float(np.quantile(distances(short['points'],[long['points']]),.9))
                pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],
                    'p90_px':d,'nomination':'embedded_endpoint_span',
                    'shared_native_span':True,
                    'source_endpoint_nodes':list(short['endpoint_nodes'])})
                continue
            span=not same_nodes and len(set(a.get('network_nodes',()))&set(b.get('network_nodes',())))>=2
            short,long=sorted((a,b),key=lambda r:r['length'])
            partial=(not a.get('closed') and not b.get('closed') and 'segment' not in a and 'segment' not in b
                and not any(h not in (None,'none') for h in short.get('heads',()))
                and not (a.get('endpoint_nodes') and b.get('endpoint_nodes') and not same_nodes)
                and short['length']/long['length']>=.2
                and np.quantile(distances(short['points'],[long['points']]),.95)<=tolerance)
            span=span or partial
            # A visible gap can duplicate a longer straight shaft. A loop or
            # decoration merely passing the same nodes is not that shaft.
            if span and not partial and any(np.quantile(distances(r['points'],[r['points'][[0,-1]]]),.95)>
                            max(2.,r.get('style',{}).get('width_px') or 1) for r in (a,b)):
                continue
            # Shared nodes do not identify a particular parallel port/route.
            # Keep the stroke-scale tolerance; a node-radius allowance can
            # collapse all parallel bus/grid lines between the same two boxes.
            # One branch may extend beneath node fills while the other stops
            # at their borders. Shared endpoint identity covers this difference.
            if not (same_nodes or span) and min(a['length'],b['length'])/max(a['length'],b['length'])<.8:continue
            if not span and not a.get('closed',False) and endpoint_error>max(10.,1.5*tolerance):continue
            ds=[float(np.quantile(distances(ap,[bp]),.9)),float(np.quantile(distances(bp,[ap]),.9))]
            d=ds[0 if a['length']<=b['length'] else 1] if span else max(ds)
            if d>tolerance:continue
            if a.get('filled_face') and b.get('filled_face'):
                pa,pb=Polygon(a['filled_face']),Polygon(b['filled_face'])
                # Adjacent opaque faces intentionally share a native boundary.
                # Suppressing one side could put the surviving half-stroke
                # behind its neighbour's fill. Independent line/arrow objects
                # are still eligible, as are overlapping duplicate faces.
                if pa.is_valid and pb.is_valid and pa.intersection(pb).area<=.01*min(pa.area,pb.area):continue
            pairs.append({'id':'pair_%03d'%len(pairs),'a':a['id'],'b':b['id'],'p90_px':d,
                          'shared_native_span':bool(partial),
                          'nomination':'same_endpoint_nodes' if same_nodes else 'shared_network_span' if span else 'mutual_route_proximity'})
    # A shifted terminal can be assigned to its neighbour even though both
    # branches recover the same node-owned route. Nominate mutually nearest
    # remaining complete paths within their existing endpoint-node uncertainty.
    # This adds comparison only; no geometry, native survivor or source count
    # is inferred from the association.
    involved={pair[side] for pair in pairs for side in ('a','b')}
    remaining=[row for row in rows if row['id'] not in involved and row.get('network_nodes')
               and not row.get('closed') and 'segment' not in row and not row.get('filled_route')]
    remaining_by_id={row['id']:row for row in remaining}
    associations=[]
    for index,a in enumerate(remaining):
        for b in remaining[index+1:]:
            if a['layer']==b['layer'] or not set(a['network_nodes']).intersection(b['network_nodes']):continue
            if min(a['length'],b['length'])/max(a['length'],b['length'])<.8:continue
            ap,bp=a['points'],b['points']
            if np.linalg.norm(ap[0]-bp[-1])+np.linalg.norm(ap[-1]-bp[0])<np.linalg.norm(ap[0]-bp[0])+np.linalg.norm(ap[-1]-bp[-1]):bp=bp[::-1]
            error=max(float(np.linalg.norm(ap[0]-bp[0])),float(np.linalg.norm(ap[-1]-bp[-1])))
            uncertainty=(a.get('node_radius_px')or 0)+(b.get('node_radius_px')or 0)
            if error>uncertainty:continue
            # Node uncertainty may explain translation, not an unrelated
            # direction or route. Compare complete paths after removing only
            # their centre displacement, with the SAME stroke-scale tolerance
            # used above. A shared large node must not nominate a diagonal
            # internal mark against a nearby vertical input connection.
            ac=ap-(ap[0]+ap[-1])/2;bc=bp-(bp[0]+bp[-1])/2
            tolerance=max(2.,.045*min(a['length'],b['length']),
                          (a.get('style',{}).get('width_px') or 1)+(b.get('style',{}).get('width_px') or 1))
            centered=max(float(np.quantile(distances(ac,[bc]),.9)),
                         float(np.quantile(distances(bc,[ac]),.9)))
            if centered>tolerance:continue
            middle=max(float(np.median(distances(ap,[bp]))),float(np.median(distances(bp,[ap]))))
            cost=error+middle
            associations.append((cost,a['id'],b['id']))
    emitted=set()
    while True:
        nearest={}
        for cost,a,b in associations:
            if a in emitted or b in emitted:continue
            for left,right in ((a,b),(b,a)):
                proposal=(cost,right)
                if left not in nearest or proposal<nearest[left]:nearest[left]=proposal
        mutual=[(ident,other,cost) for ident,(cost,other) in sorted(nearest.items())
                if ident<other and nearest.get(other,(None,None))[1]==ident]
        if not mutual:break
        for ident,other,cost in mutual:
            ap,bp=remaining_by_id[ident]['points'],remaining_by_id[other]['points']
            deviation=max(float(np.quantile(distances(ap,[bp]),.9)),float(np.quantile(distances(bp,[ap]),.9)))
            pairs.append({'id':'pair_%03d'%len(pairs),'a':ident,'b':other,
                          'nomination':'unpaired_node_route','p90_px':deviation,
                          'endpoint_pairing_cost_px':cost,
                          'native_endpoint_identity_uncertain':True})
            emitted.update((ident,other))
    if source_rgb is not None:
        from .source_route_nomination import nominate as source_nominate
        for pair in source_nominate(rows,source_rgb,pairs):
            pair['id']='pair_%03d'%len(pairs);pairs.append(pair)
    return pairs


def protect_parallel_pairs(rows,pairs,rgb):
    """Keep demonstrated distinct source strokes before any fallible review."""
    from .line_identity import separate_parallel_sources,distinct_source_routes
    table={r['id']:r for r in rows};kept=[];protected=[]
    for pair in pairs:
        if pair.get('native_endpoint_identity_uncertain'):
            # Nearby source strokes cannot settle this uncertain association;
            # keep the existing source/native review instead of certifying two
            # distinct occurrences from local stroke bands at a crossing.
            kept.append(pair);continue
        if pair.get('nomination')=='matching_filled_route':
            kept.append(pair);continue
        if distinct_source_routes(rgb,table[pair['a']],table[pair['b']]):
            protected.append(pair);continue
        a,b=sorted((table[pair['a']],table[pair['b']]),key=lambda r:r['length']);ap,bp=a['points'],b['points']
        straight=all(np.quantile(distances(r['points'],[r['points'][[0,-1]]]),.95)<1. for r in (a,b))
        color=a['style'].get('stroke') or a['style'].get('color','#000000')
        other_color=b['style'].get('stroke') or b['style'].get('color','#000000')
        if straight and separate_parallel_sources(rgb,ap[[0,-1]],bp[[0,-1]],color,max(4.,pair['p90_px']+2.),other_color):
            protected.append(pair);continue
        kept.append(pair)
    return kept,protected


def candidate_components(rows):
    """Couple only whole-contour/segment edits of the SAME native object.

    Shared nodes, endpoints, owners and heads do not make routes duplicates.
    Independent subpaths can be retained/deleted independently.
    """
    links={r['id']:{r['id']} for r in rows}
    for row in rows:
        parent='shape:'+row.get('parent_key','')
        if parent in links:
            links[row['id']].add(parent);links[parent].add(row['id'])
    components={}
    for ident in links:
        if ident in components:continue
        members=set();pending=[ident]
        while pending:
            current=pending.pop()
            if current in members:continue
            members.add(current);pending.extend(links[current]-members)
        for member in members:components[member]=members
    return components


def comparison_groups(pairs, rows=()):
    """Group direct competitors and actual native edit dependencies.

    Shared endpoints do not couple independently editable connections. Batching
    comparisons is separate from choosing a survivor for each source route.
    """
    remaining=list(pairs);groups=[]
    ids={p[k] for p in pairs for k in ('a','b')}
    dependencies={n:{n} for n in ids}
    dependencies.update(candidate_components(rows))
    while remaining:
        component=[remaining.pop(0)];nodes={component[0]['a'],component[0]['b']}
        while True:
            linked=[p for p in remaining if (dependencies[p['a']]|dependencies[p['b']])&nodes]
            if not linked:break
            component.extend(linked)
            nodes.update(n for p in linked for n in (p['a'],p['b']))
            remaining=[p for p in remaining if p not in linked]
        groups.append(component)
    return groups


def allowed_branches(batch, rows, scene, styles):
    """Only native editability limits choices; geometry never chooses quality."""
    if any(r.get('immutable_border') for r in rows):return ['shape']
    return [branch for branch in ('shape','arrows') if any(r['layer']==branch for r in rows)]


def connection_route(row):
    """Source-presence review is for connectors, never faces or decorations."""
    if row.get('closed') or 'segment' in row or row.get('filled_route'):return False
    if any(h not in (None,'none') for h in row.get('heads',())):return True
    if not row.get('network_nodes') or 'points' not in row:return False
    return np.quantile(distances(row['points'],[row['points'][[0,-1]]]),.95)<=max(2.,.03*row['length'])


def validate_choice(raw, allowed=('shape','arrows','none')):
    """One branch choice; echoed input metadata never becomes another decision."""
    if (not isinstance(raw,dict) or raw.get('keep') not in allowed
            or not isinstance(raw.get('reason'),str) or not raw['reason'].strip()):
        raise ValueError('Return one allowed keep value and a short source-based reason')
    return {'keep':raw['keep'],'reason':raw['reason']}


def validate_selection(raw, candidates, *, allow_empty=False, required=(), branch_by_id=None):
    """Select actual native recoveries, not every member of a winning branch."""
    if not isinstance(raw,dict) or not isinstance(raw.get('keep_ids'),list):
        raise ValueError('Return keep_ids and a source-based reason')
    ids=raw['keep_ids']
    if (not all(isinstance(k,str) for k in ids) or len(set(ids))!=len(ids)
            or not set(ids)<=set(candidates) or not set(required)<=set(ids)
            or not ids and not allow_empty or not isinstance(raw.get('reason'),str) or not raw['reason'].strip()):
        raise ValueError('Keep only listed candidates; preserve required native contours')
    if branch_by_id is not None and len({branch_by_id[ident] for ident in ids})>1:
        raise ValueError('Choose one branch for this local physical-route group; do not retain both versions')
    return {'keep_ids':ids,'reason':raw['reason']}


def validate_source_inventory(raw, source_box):
    """Validate source observations before any native candidates are shown."""
    if (isinstance(raw,dict) and 'source_objects' not in raw and isinstance(raw.get('output'),dict)
            and all(k in {'validation_error','correction'} and v in ('',None)
                    for k,v in raw.items() if k!='output')):
        raw=raw['output']
    if not isinstance(raw,dict) or not isinstance(raw.get('source_objects'),list) or not raw['source_objects']:
        raise ValueError('Return independently observed source_objects')
    result=[];seen=set();box=np.asarray(source_box,float)
    for obj in raw['source_objects']:
        if not isinstance(obj,dict):raise ValueError('Invalid source observation')
        ident=obj.get('id')
        try:path=np.asarray(obj.get('source_path',[]),float)
        except (TypeError,ValueError) as exc:raise ValueError('Source paths require pixel coordinate pairs') from exc
        if (not isinstance(ident,str) or not ident.strip() or ident in seen or
                not isinstance(obj.get('description'),str) or not obj['description'].strip()):
            raise ValueError('Give each observed source path a unique ID and visible description')
        if (path.ndim!=2 or path.shape[1:]!=(2,) or not 2<=len(path)<=64 or not np.isfinite(path).all()
                or np.any(path<box[:2]-1) or np.any(path>box[2:]+1)
                or np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<1):
            raise ValueError('Source path anchors must follow visible content inside the SOURCE image')
        seen.add(ident);result.append({'id':ident,'source_path':path.tolist(),'description':obj['description']})
    return {'source_objects':result}


def comparison_box(rows, size, padding=18):
    """Keep complete competing geometry; clip read-only peers to this view.

    A long node/rail displayed as context must not enlarge a local comparison
    into a whole-diagram source inventory. Candidate paths and attached heads
    still contribute their entire extent, including displaced alternatives.
    """
    points=np.concatenate([row['points'] for row in rows]+
                          [np.asarray(b) for row in rows for b in row.get('attachment_bounds',[])])
    bounds=np.r_[points.min(0)-padding,points.max(0)+padding]
    return tuple(int(v) for v in np.rint([max(0,bounds[0]),max(0,bounds[1]),
        min(size[0],bounds[2]+1),min(size[1],bounds[3]+1)]))


def enclosing_context_keys(rows, scene, styles):
    """Retain existing painted enclosures behind a native comparison.

    Endpoint nodes alone leave white recoveries invisible on a white slide.
    These shared native faces are read-only context, never extra competitors.
    Their full bounds do not enlarge the candidate-bounded SOURCE viewport.
    """
    from shapely.ops import unary_union
    from shape_integration.styling.geometry import footprint
    selected={k for row in rows for k in (*row['keys'],row.get('parent_key'))}
    paths=unary_union([LineString(row['points']) for row in rows])
    found=set()
    for key,part in scene['parts'].items():
        paint=styles.get(key,{})
        if (key in selected or part.get('status','ok')!='ok' or not part.get('commands')
                or paint.get('fill') is None or paint.get('fill_opacity',1)<=0):continue
        surface,error=footprint(part)
        if not error and surface.area>0 and surface.buffer(.5).covers(paths):found.add(key)
    return found


def source_span_evidence(nodes, table, definitions):
    """Read-only extent evidence; it never changes geometry or chooses a branch."""
    result={}
    for alias,n in nodes.items():
        points=table[n]['points'];items=[]
        for definition in definitions:
            observed=resample([np.asarray(definition['source_path'],float)])
            delta=distances(observed,[points])
            index=((observed[:,None]-points[None,:])**2).sum(2).argmin(1)
            items.append({'source_id':definition['id'],
                'median_distance_px':round(float(np.median(delta)),2),
                'p90_distance_px':round(float(np.quantile(delta,.9)),2),
                'native_path_interval':[round(float(index.min())/max(1,len(points)-1),3),
                                        round(float(index.max())/max(1,len(points)-1),3)]})
        result[alias]=items
    return result


def candidate_branch_labels(rows):
    """Opaque provenance labels avoid inferring heads from a branch name."""
    layers=list(dict.fromkeys(row['layer'] for row in rows))
    return {layer:f'B{i+1}' for i,layer in enumerate(layers)}


def observe_source(clean, box, folder, runtime):
    """A bounded source-only observation for ambiguous multi-candidate crops."""
    from shape_integration.common import save
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    crop=clean.crop(box);scale=min(8.,1024/max(crop.size))
    view=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    path=folder/'source.png';view.save(path)
    prompt={'task':'只根据这张原图局部，枚举可见的独立线形路径。本次没有提供任何复原候选，不能按候选数量补线。',
        'image_size_px':list(view.size),'rules':[
            '按像素可见的走向、端点、转折和颜色逐条描述。相邻、重复排布或共享节点不等于同一对象；不可按对称性、全连接或器件先验补足数量。',
            '可见线形包括无头线、有头线、闭合回路、虚线、分隔线、叠层轮廓、线框边界和窄的填充连接带。位于形状内部或外部、连接节点或描绘边界，都不是排除依据。填充面、背景色块和文字只作上下文；其上可见的独立线形仍需按像素记录。',
            '每条独立线形路径给一个ID，source_path沿其可见中心路径取端点及必要转折，使用当前图像像素坐标。路径上的头部是这条路径的部件，写入简短description，不拆成一个需要另选候选的新对象。',
            '同一连续路径被实体面遮挡时，只有遮挡两侧同位置、同走向的可见延续才可接续。不可把相邻平行线或相邻叠层轮廓合成一个对象；独立的闭合回路也作为完整路径，不拆成几条互相竞争的边。',
            '裁剪边缘的可见线段可以记录，但不要猜测框外内容；后续匹配允许裁剪边缘的上下文片段没有复原候选。每条description只用一句话说明走向、端点和头部。',
            '只记录原图证据，不给shape/arrows分类，不作保留或删除决定。'],
        'output':{'source_objects':[{'id':'S1','source_path':[[0,0],[10,10]],'description':'可见路径的走向、端点及颜色。'}]}}
    raw=runtime.call(folder,'inventory','只观察原图可见路径，不推测复原结果。只返回JSON。',prompt,
        [('SOURCE only; no recovered candidates are provided.',path)],
        lambda d:validate_source_inventory(d,[0,0,view.width-1,view.height-1]),8192)
    converted=[]
    for obj in raw['source_objects']:
        points=np.asarray(obj['source_path'],float)*np.array(crop.size)/np.array(view.size)+np.asarray(box[:2])
        converted.append({**obj,'source_path':points.tolist()})
    save(folder/'source_objects_original.json',{'crop_box_px':[int(v) for v in box],
        'view_size_px':list(view.size),'source_objects':converted,'source_only':True})
    return converted


def validate_source_selection(raw, candidates, *, branch_by_id, source_box, required=(), dependencies=(), source_definitions=None, required_source_ids=(), context_by_id=None):
    """The comparison window may contain several independent source routes.

    Keep binary ownership per observed source path, not per transitive proximity
    component. Anchors describe SOURCE pixels and never alter native geometry.
    """
    # Some clients echo the input envelope and put their actual response in
    # its explicit output field. Unwrap only when no top-level decision is
    # present, then apply exactly the same conservation/source checks below.
    wrapped=(isinstance(raw,dict) and 'keep_ids'not in raw
             and isinstance(raw.get('output'),dict) and 'keep_ids'in raw['output'])
    if wrapped:raw=raw['output']
    result=validate_selection(raw,candidates,required=required)
    if wrapped:result['response_envelope']='output'
    objects=raw.get('source_objects')
    if not isinstance(objects,list) or not objects:
        raise ValueError('Enumerate physical source_objects before matching native candidates')
    box=np.asarray(source_box,float);seen=set();source_ids=set();paths=[];assignments={};seen_context=set();unrepresented=set()
    for index,obj in enumerate(objects):
        if not isinstance(obj,dict):raise ValueError('Invalid source object')
        unit_sources=obj.get('source_ids',[obj.get('id')])
        if (not isinstance(unit_sources,list) or not unit_sources
                or not all(isinstance(n,str) and n.strip() for n in unit_sources)
                or len(set(unit_sources))!=len(unit_sources) or source_ids&set(unit_sources)
                or obj.get('id')!=unit_sources[0]):
            raise ValueError('Source decision units use distinct frozen source IDs; id is the first source_id')
        if source_definitions is not None:
            definitions={item['id']:item for item in source_definitions}
            if not set(unit_sources)<=set(definitions):raise ValueError('Match only the independently frozen source IDs; never invent another source object')
            frozen=definitions[obj['id']]['source_path']
            if 'source_path' in obj and obj['source_path']!=frozen:raise ValueError('The independently observed source path is locked')
            obj={**obj,'source_path':frozen}
        ident=obj.get('id');members=obj.get('candidate_ids')
        try:path=np.asarray(obj.get('source_path',[]),float)
        except (TypeError,ValueError) as exc:raise ValueError('source_path must contain original-pixel coordinate pairs') from exc
        if not isinstance(ident,str) or not ident.strip() or ident in source_ids:
            raise ValueError('Every physical source path needs a unique source ID')
        if (path.ndim!=2 or path.shape[1:]!=(2,) or not 2<=len(path)<=64 or not np.isfinite(path).all()
                or np.any(path<box[:2]-1) or np.any(path>box[2:]+1)
                or np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<1):
            raise ValueError('source_path must trace a visible SOURCE route in the given original-pixel crop')
        # Context aliases only account for SOURCE units without a C.
        # Represented units already have explicit native ownership.
        context_ids=obj.get('context_ids',[]) if not members else []
        context_only=(members==[] and obj.get('keep_ids')==[] and bool(context_ids))
        # Unrepresented SOURCE context has no authority over any native C.
        # Two-candidate calls observe SOURCE in this same request; their
        # passive context follows the same rule as an independent inventory.
        cropped_context=(members==[] and obj.get('keep_ids')==[] and not context_ids
            and not set(unit_sources)&set(required_source_ids))
        if (not isinstance(context_ids,list) or not all(isinstance(n,str) for n in context_ids)
                or len(set(context_ids))!=len(context_ids) or not set(context_ids)<=set(context_by_id or {})
                or context_only and seen_context&set(context_ids)):
            raise ValueError('Only listed read-only native context may account for a source object without competing candidates')
        if (not isinstance(members,list) or not members and not (context_only or cropped_context) or not all(isinstance(n,str) for n in members)
                or len(set(members))!=len(members) or not set(members)<=set(candidates) or seen&set(members)):
            assigned=[n for item in objects if isinstance(item,dict)
                      and isinstance(item.get('candidate_ids'),list)
                      for n in item['candidate_ids'] if isinstance(n,str)]
            repeated=sorted(n for n in set(assigned) if assigned.count(n)>1)
            unknown=sorted(set(assigned)-set(candidates))
            missing=sorted(set(candidates)-set(assigned))
            raise ValueError('Assign every native candidate exactly once. '
                f'Repeated candidate_ids={repeated}; unknown={unknown}; omitted={missing}. '
                'List each C in only one SOURCE unit, including rejected recoveries. '
                'Correct the allocation using the same locked SOURCE paths and native evidence; '
                'candidate_ids includes competitors, while keep_ids lists survivors only.')
        selected=validate_selection(obj,members,allow_empty=context_only or cropped_context,required=set(required)&set(members),branch_by_id=branch_by_id)
        # Relabelling the same traced SOURCE route cannot evade binary choice.
        sampled=resample([path])
        for other in paths:
            if max(float(distances(sampled,[other]).max()),float(distances(other,[sampled]).max()))<=1:
                raise ValueError('Repeated source_path: put all recoveries of that physical route in one source object')
        paths.append(sampled);source_ids.update(unit_sources);seen.update(members)
        if context_only:seen_context.update(context_ids)
        if cropped_context:unrepresented.update(unit_sources)
        assignments.update({n:index for n in members})
        obj={**selected,'id':ident,'source_ids':unit_sources,'source_path':path.tolist(),'candidate_ids':members,'context_ids':context_ids}
        if source_definitions is not None:
            obj['source_definitions']=[definitions[n] for n in unit_sources]
        result.setdefault('source_objects',[]).append(obj)
    # The top-level keep_ids already decides every C in this comparison.
    # A globally rejected recovery owns no surviving SOURCE geometry. Do not
    # invent an owner just to complete its metadata; every survivor is still
    # required in exactly one observed SOURCE unit.
    unassigned=set(candidates)-seen
    if (unassigned and source_definitions is not None and
            {item['id'] for item in source_definitions}-(source_ids-unrepresented)):
        raise ValueError('An unassigned rejected recovery requires complete independent SOURCE accounting; '
                         'retain a survivor or listed native context for every observed SOURCE object, '
                         'or assign the rejected C to its observed SOURCE unit.')
    if unassigned and source_definitions is None:
        raise ValueError('Unassigned rejected candidate IDs '+','.join(sorted(unassigned))+
                         ' require an independent SOURCE inventory')
    result['unassigned_rejected_candidate_ids']=sorted(unassigned-set(result['keep_ids']))
    if unassigned & set(result['keep_ids']):
        raise ValueError('Assign omitted native candidate IDs '+','.join(sorted(set(candidates)-seen))+
            ' to their SOURCE decision units, including rejected recoveries. Every listed C must occur exactly once in candidate_ids; keep_ids lists only survivors.')
    if not set(required_source_ids)<=source_ids:
        raise ValueError('Account for every fully observed source object with a survivor or listed read-only context; missing: '+','.join(sorted(set(required_source_ids)-source_ids)))
    kept={n for obj in result['source_objects'] for n in obj['keep_ids']}
    if kept!=set(result['keep_ids']):raise ValueError('keep_ids must equal the per-source survivor union')
    for component in dependencies:
        if len({assignments[n] for n in set(component)&seen})>1:
            raise ValueError('A native contour, its segment edits and directly shared native spans cannot belong to different source objects; combine their SOURCE IDs in one decision unit and choose one branch covering the complete unit')
    if source_definitions is not None:
        result['unassigned_source_ids']=sorted(({item['id'] for item in source_definitions}-source_ids)|unrepresented)
    return result


def apply_choices(rows, choices):
    """Keep the selected branch within each overlap; preserve everything outside."""
    table={r['id']:r for r in rows};removed=[];retained=set();guards=[]
    decisions={}
    for choice in choices:
        if {table[n]['layer'] for n in choice['candidate_ids']}=={'arrows'}:
            # Composition reconciles branch intersections. It has no second
            # deletion authority inside an already delivered arrow branch.
            retained.update(choice['candidate_ids'])
            continue
        keep=(choice['keep_ids'] if 'keep_ids' in choice else
              [n for n in choice['candidate_ids'] if table[n]['layer']==choice['keep']])
        keep=list(keep)
        # A local match is not permission to remove the rest of a frame.
        # Check every native contour point, including remote sides that a
        # close-up or an appealing callout notch can otherwise hide.
        for ident in choice['candidate_ids']:
            row=table[ident]
            if ident in keep or row['layer']!='shape' or not (row.get('closed') or 'segment' in row):continue
            replacements=[table[n] for n in keep if table[n]['layer']=='arrows']
            if not replacements:continue
            tolerance=max(3.,2*float(row.get('style',{}).get('width_px') or 1.))
            paths=[p for r in replacements for p in r.get('paths',[r['points']])]
            covered=float(np.mean(distances(row['points'],paths)<=tolerance))
            if covered<.95:
                keep.append(ident)
                guards.append({'retained':ident,'replacement_ids':[r['id'] for r in replacements],
                    'covered_fraction':round(covered,4),'tolerance_px':tolerance,
                    'reason':'Partial replacement cannot delete a complete native border'})
        for ident in choice['candidate_ids']:
            decision=ident in keep
            if ident in decisions and decisions[ident]!=decision:
                raise ValueError('Overlapping branch decisions disagree: '+ident)
            decisions[ident]=decision
        retained.update(keep)
        removed.extend({'remove':n,'keepers':keep,'keep_branch':choice.get('keep','explicit_candidates')}
                       for n in choice['candidate_ids'] if n not in keep)
    deleted={r['remove'] for r in removed}
    # A head may serve multiple shafts. Delete it only with its last user.
    live_keys={k for r in rows if r['id'] not in deleted and r['layer']=='shape' for k in r['keys']}
    live_heads={ref for r in rows if r['id'] not in deleted for ref in r.get('attachment_refs',[])}
    shape_keys=sorted({k for n in deleted if table[n]['layer']=='shape' for k in table[n]['keys']}-live_keys)
    suppression={};subpaths={}
    for n in deleted:
        row=table[n]
        if 'segment' in row:suppression.setdefault(row['refs'][0],[]).append(row['segment'])
        if 'subpath' in row:subpaths.setdefault(row['refs'][0],[]).extend(row.get('subpaths',[row['subpath']]))
    drop_names={ref for n in deleted if table[n]['layer']=='shape' and 'segment' not in table[n]
                and 'subpath' not in table[n] for ref in table[n]['refs']}
    drop_names.update(ref for n in deleted for ref in table[n].get('attachment_refs',[]))
    drop_names-=live_heads
    for ref,indices in list(subpaths.items()):
        part=next(r for r in rows if r.get('subpath_count') and r['refs'][0]==ref)
        if len(set(indices))==part['subpath_count']:
            shape_keys.append(part['parent_key']);drop_names.add(ref);del subpaths[ref]
    return {'drop_shape_keys':shape_keys,
            'drop_shape_names':sorted(drop_names),
            'shape_segment_suppression':{k:sorted(set(v)) for k,v in suppression.items() if k not in drop_names},
            'shape_subpath_suppression':{k:sorted(set(v)) for k,v in subpaths.items()},
            'drop_arrow_names':sorted(table[n]['key'] for n in deleted if table[n]['layer']=='arrows'),
            'retained_shape_keys':sorted({k for n in retained if table[n]['layer']=='shape'
                for k in (table[n]['keys'] or [table[n].get('parent_key')]) if k}),
            'removed':removed,'conflicting_components_retained':[],'native_frame_guards':guards}


def review(scene,styles,arrows,source,shape_pptx,arrow_pptx,folder,runtime,*,selected=None,assets=(),texts=()):
    # httpcore/AnyIO can first-load the async backend from concurrent native
    # requests. Initialize it here before starting the decision thread pool.
    import importlib
    importlib.import_module('anyio._backends._asyncio')
    from shape_integration.common import save
    from shape_integration.styling.export import render_native
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:clean=im.convert('RGB')
    from .shape_duplicates import duplicates,filled_duplicates,native_paint_order
    filled,filled_audit=filled_duplicates(scene,styles,np.asarray(clean),
        paint_order=native_paint_order(shape_pptx) if shape_pptx is not None else None)
    filled_names=[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in filled]
    from .handoff import surviving_scene
    scene=surviving_scene(scene,{'drop_shape_keys':filled})
    rows=candidates(scene,styles,arrows)
    if selected is not None:rows=[r for r in rows if r['key'] in selected]
    redundant,internal_audit=duplicates(rows,np.asarray(clean))
    internal=apply_choices(rows,[{'keep':'arrows','candidate_ids':sorted(redundant)}]) if redundant else {}
    internal['drop_shape_keys']=sorted(set(internal.get('drop_shape_keys',[]))|filled)
    internal['drop_shape_names']=sorted(set(internal.get('drop_shape_names',[]))|set(filled_names))
    internal_audit.extend(filled_audit)
    rows=[r for r in rows if r['id'] not in redundant]
    save(folder/'shape_duplicates.json',internal_audit)
    pairs,parallel=protect_parallel_pairs(rows,nominate(rows,np.asarray(clean)),np.asarray(clean))
    table={r['id']:r for r in rows}
    groups=comparison_groups(pairs,rows)
    # The union is the starting product. Only directly competing routes enter
    # the removal set; context and graph connectivity never enlarge it.
    members=[sorted({p[k] for p in group for k in ('a','b')}) for group in groups]
    # Geometry critic owns source completeness. Composition only resolves
    # actual competitors; a second existence vote could contradict that critic
    # and delete an otherwise useful unmatched route from the union.
    unique=[]
    save(folder/'source_only_candidates.json',unique)
    allowed_by_group={}
    for batch,nodes in zip(groups,members):
        allowed_by_group[batch[0]['id']]={
            'allow_empty':False,
            'required':[n for n in nodes if table[n].get('immutable_border')]}
    involved={n for group in members for n in group}
    # Source-protected parallel pairs are final too; legacy ownership must not
    # reinterpret them after this review.
    accounted=involved|{p[k] for p in parallel for k in ('a','b')}
    reviewed_arrows=sorted(table[n]['key'] for n in accounted if table[n]['layer']=='arrows')
    save(folder/'pairs.json',pairs)
    save(folder/'distinct_parallel_sources.json',parallel)
    save(folder/'comparison_groups.json',[[p['id'] for p in batch] for batch in groups])
    save(folder/'comparison_members.json',members)
    scope_audit={'source_ridge_nomination_pairs':sum(p.get('nomination')=='shared_original_straight_ridge'for p in pairs),
                 'ownership_completeness_scope':'nominated_competitor_groups',
                 'unreviewed_route_count':len({r['id']for r in rows}-accounted)}
    if not groups:return {**{'ordinary_line_owner':'local_competitor_groups','drop_shape_keys':[],'drop_shape_names':[],'drop_arrow_names':[], 'reviewed_arrow_names':reviewed_arrows, 'model_calls':0,'removed':[]},**internal,**scope_audit,'internal_duplicates':internal_audit,'line_ownership_complete':True}
    contexts={};context_by_group={}
    for batch,nodes in zip(groups,members):
        # Show the SAME native nodes above either candidate network. Otherwise
        # visible gaps between opaque nodes look like a dashed/missing shaft.
        node_ids={key for n in nodes for key in
                  (*table[n].get('network_nodes',()),*table[n].get('endpoint_nodes',()))}
        node_ids.update(enclosing_context_keys([table[n] for n in nodes],scene,styles))
        selected_keys={key for n in nodes for key in
                       (*table[n]['keys'],table[n].get('parent_key'))}
        context_ids=sorted(node_ids-selected_keys)
        context_by_group[batch[0]['id']]={f'N{i+1}':{'bbox_2d':scene['parts'][key]['bbox_2d'],'native_key':key}
            for i,key in enumerate(context_ids)}
        context=None;back_context=None
        if context_ids:
            back=[]
            for key in context_ids:
                b=scene['parts'][key]['bbox_2d']
                # Intrinsic glyphs inside a node must be visible ABOVE its
                # fill. Between-node routes still end underneath node faces.
                if any(np.all(table[n]['points']>=np.asarray(b[:2])-1) and
                       np.all(table[n]['points']<=np.asarray(b[2:])+1) for n in nodes):back.append(key)
            front=[k for k in context_ids if k not in back]
            context=folder/('context_'+batch[0]['id']+'.pptx')
            subset(shape_pptx,context,[scene['parts'][key]['object_id']+'/'+json_part_id(key) for key in front])
            if back:
                back_context=folder/('back_context_'+batch[0]['id']+'.pptx')
                subset(shape_pptx,back_context,[scene['parts'][key]['object_id']+'/'+json_part_id(key) for key in back])
        contexts[batch[0]['id']]=(context,back_context,[scene['parts'][key]['bbox_2d'] for key in sorted(node_ids)])
    from .line_group_choice import eligible,compare as compare_complete_groups
    whole_groups=[(batch,nodes) for batch,nodes in zip(groups,members)
                  if eligible([table[n] for n in nodes],batch)]
    whole_ids={batch[0]['id'] for batch,nodes in whole_groups}
    def render(job):
        # Native subsets retain each candidate's transforms, heads and paint.
        batch,nodes,selection,label=job;ident=(batch[0]['id'],label)
        branch=table[selection[0]]['layer']
        suffix=str(nodes.index(selection[0])) if len(selection)==1 else 'complete_'+branch
        local=folder/('candidate_render_'+ident[0]+'_'+suffix);local.mkdir(exist_ok=True)
        selected_rows=[table[n] for n in selection]
        source=shape_pptx if branch=='shape' else arrow_pptx
        ppt=local/'isolated.pptx';selection_subset(source,ppt,selected_rows)
        context,back_context,_=contexts[batch[0]['id']]
        if back_context is not None:
            shown=local/'with_back_context.pptx'
            merge(back_context,[(branch,ppt)],shown);ppt=shown
        if context is not None:
            shown=local/'with_context.pptx'
            merge(ppt,[('context',context)],shown)
            ppt=shown
        preview=render_native(ppt,local/'render',scene['source_size'],180)
        with Image.open(preview) as im:return ident,im.convert('RGB')
    jobs=[]
    for batch,nodes in zip(groups,members):
        if batch[0]['id'] in whole_ids:
            for layer in candidate_branch_labels([table[n] for n in nodes]):
                jobs.append((batch,nodes,[n for n in nodes if table[n]['layer']==layer],layer))
        else:
            jobs.extend((batch,nodes,[n],n) for n in nodes)
    with ThreadPoolExecutor(max_workers=4) as pool:native=dict(pool.map(render,jobs))
    def evidence(batch,nodes,*,complete=False):
        # Endpoint matches only locate context. An unresolved match is not a
        # missing connection, and must not become a branch-quality assertion.
        context,back_context,regions=contexts[batch[0]['id']]
        box=comparison_box([table[n] for n in nodes],clean.size)
        panels=[('SOURCE',clean)]
        # Candidates share source coordinates and identical node context.
        labels=candidate_branch_labels([table[n] for n in nodes])
        if complete:
            for layer,label in labels.items():
                aliases=[f'C{i+1}' for i,n in enumerate(nodes) if table[n]['layer']==layer]
                panels.append((label+' complete native '+','.join(aliases)+' + same context',native[(batch[0]['id'],layer)]))
        else:
            for i,n in enumerate(nodes):
                label=f'C{i+1} '+labels[table[n]['layer']]+' native candidate + same context'
                panels.append((label,native[(batch[0]['id'],n)]))
        # Endpoint details share the SAME render/crop/scale for both branches.
        # They are evidence in this call, not another selection or critic.
        boxes=[box]
        if max(box[2]-box[0],box[3]-box[1])>240:
            route=max((table[n] for n in nodes),key=lambda r:r['length'])
            for point in route['points'][[0,-1]]:
                b=tuple(np.rint([max(0,point[0]-64),max(0,point[1]-64),min(clean.width,point[0]+65),min(clean.height,point[1]+65)]).astype(int))
                if b not in boxes:boxes.append(b)
        images=[]
        # Bound panel width so a group with several overlapping candidates
        # does not shrink every thin route into an unreadable wide thumbnail.
        # The same request sees all pages with stable C IDs and source crops.
        for start in range(1,len(panels),2):
            page=[panels[0],*panels[start:start+2]]
            tiles=[[ImageOps.contain(im.crop(region),(704,676),Image.Resampling.LANCZOS)
                    for label,im in page] for region in boxes]
            heights=[max(tile.height for tile in row)+40 for row in tiles]
            sheet=Image.new('RGB',(720*len(page),sum(heights)),'#edf0f3');draw=ImageDraw.Draw(sheet)
            y=0
            for j,(region,row_tiles,height) in enumerate(zip(boxes,tiles,heights)):
                for i,(label,im) in enumerate(page):
                    x=i*720
                    draw.text((x+8,y+8),label+(' endpoint detail' if j else ''),fill='black')
                    tile=row_tiles[i]
                    sheet.paste(tile,(x+(720-tile.width)//2,y+32+(height-40-tile.height)//2))
                y+=height
            path=folder/('group_'+batch[0]['id']+f'_{start//2:02d}.png');sheet.save(path)
            images.append(('SOURCE and native candidates '+', '.join(n.split()[0] for n,_ in page[1:]),path))
        return images,boxes
    whole_choices,whole_calls=(compare_complete_groups(whole_groups,table,clean,native,
                              lambda batch,nodes:evidence(batch,nodes,complete=True),
                              context_by_group,folder/'complete_groups',runtime)
                              if whole_groups else ({},0))
    # Groups with independently editable alternatives use one SOURCE/native
    # subset comparison, including mixed branches and same-branch duplicates.
    def check(group):
        batch,nodes=group
        if batch[0]['id'] in whole_choices:
            return whole_choices[batch[0]['id']],None
        aliases={f'C{i+1}':n for i,n in enumerate(nodes)}
        required=[a for a,n in aliases.items() if n in allowed_by_group[batch[0]['id']]['required']]
        images,regions=evidence(batch,nodes)
        components=candidate_components([table[n] for n in nodes])
        reverse={n:a for a,n in aliases.items()}
        context_ids={v['native_key']:k for k,v in context_by_group[batch[0]['id']].items()}
        dependencies={frozenset(reverse[n] for n in component if n in reverse)
                      for component in components.values()}
        request={'task':'对当前原生候选做一次局部去重，保留每条实际源路径的完整实现。',
            'rules':[
                '在SOURCE沿端点和转折追踪每个C对应的源路径。endpoint_context_ids是接入节点提示；同一对节点间只有一条源路径时，有定位偏差的多个C仍是重复，邻近的其他节点连线不算第二条。真实平行路径按SOURCE分别保留。',
                '同源重复选择完整性、端点、线型最贴近SOURCE的实现；不同轮廓、叠层和互补区段保留。每个删除项必须由直接竞争的幸存C完整替代，不沿相交或共享节点的传递关系删除。',
                'context只读；required保留；coupled_candidate_ids每组整体取舍。分支来源和候选数量不决定源路径身份。',
                '只返回保留的C，几何、头部和绘制次序保持原样。证据不足以确认重复时保留对应候选。'],
            'candidates':list(aliases),'required':required,
            'direct_competitors':[[reverse[p['a']],reverse[p['b']]]for p in batch],
            'coupled_candidate_ids':[sorted(d) for d in dependencies if len(d)>1],
            'source_crop_box_px':[int(v) for v in regions[0]],
            'context_objects':{n:{'bbox_2d':v['bbox_2d']} for n,v in context_by_group[batch[0]['id']].items()},
            'candidate_geometry':{a:{'source_points':table[n]['points'][
                np.linspace(0,len(table[n]['points'])-1,5).astype(int)].round(2).tolist(),
                'endpoint_context_ids':[context_ids[k]for k in table[n].get('endpoint_nodes',())if k in context_ids]}
                for a,n in aliases.items()},
            'output':{'keep_ids':['C1'],'reason':'逐项说明同源重复或独立原图内容。'}}
        def validate_subset(raw):
            if isinstance(raw,dict) and 'keep_ids' not in raw and isinstance(raw.get('output'),dict):
                raw=raw['output']
            result=validate_selection(raw,aliases,required=required)
            kept=set(result['keep_ids'])
            if any(kept&d and not d<=kept for d in dependencies):
                raise ValueError('Keep or remove each complete native dependency together')
            for alias in set(aliases)-kept:
                ident=aliases[alias]
                neighbors={reverse[p['b'] if p['a']==ident else p['a']]
                           for p in batch if ident in (p['a'],p['b'])}
                if not neighbors&kept:
                    raise ValueError('A removed candidate requires a directly competing survivor')
            return result
        try:
            choice=runtime.for_stage(request_timeout_seconds=600,reasoning_effort='low',
                                     shape_schema_retries=0).call(
                folder/('branch_'+batch[0]['id']),'review',
                '按SOURCE保留实际原生候选；独立内容不能互删，只返回JSON。',
                request,images,validate_subset,8192)
            return {**choice,'keep_ids':[aliases[a] for a in choice['keep_ids']],
                    'candidate_ids':nodes,'pairs':[p['id'] for p in batch],
                    'selection_unit':'source_native_subset'},None
        except (ValueError,RuntimeError) as exc:
            return {'candidate_ids':nodes,'keep_ids':nodes,'pairs':[p['id'] for p in batch],
                    'selection_unit':'unresolved_competitor_group','source_relation':'unresolved',
                    'whole_group_branch':None,
                    'reason':'Comparison did not complete; retain every native candidate.'},\
                   {'pairs':[p['id'] for p in batch],'error_type':type(exc).__name__,'error':str(exc)}
    workers=max(1,min(32,int(getattr(runtime,'config',{}).get('api_concurrency',8))))
    with ThreadPoolExecutor(max_workers=workers) as pool:answers=list(pool.map(check,zip(groups,members)))
    choices=[choice for choice,error in answers if choice is not None]
    failures=[error for choice,error in answers if error]
    # Unfinished comparison never removes content or restarts the full case.
    # Its union survives with explicit unresolved ownership in the audit.
    result=apply_choices(rows,choices)
    for name in ('drop_shape_keys','drop_shape_names'):
        result[name]=sorted(set(result[name])|set(internal.get(name,[])))
    for name in ('shape_segment_suppression','shape_subpath_suppression'):
        for key,indices in internal.get(name,{}).items():
            result.setdefault(name,{})[key]=sorted(set(result.get(name,{}).get(key,[]))|set(indices))
    result['internal_duplicates']=internal_audit
    inventories=0  # No separate SOURCE-ID enumeration in the active ownership path.
    result.update(**scope_audit,ordinary_line_owner='local_competitor_groups',branch_choices=choices,
                  model_calls=len(groups)-len(whole_choices)+inventories+whole_calls,
                  complete_group_calls=whole_calls,complete_group_decisions=len(whole_choices),
                  source_only_observation_calls=inventories,reviewed_pairs=sum(len(b) for b in groups if b[0]['nomination']!='unmatched_source_check'),
                  comparison_scope='connected_native_candidates',selection_scope='complete_competitors_with_explicit_unresolved',
                  source_only_candidates=unique,
                  line_ownership_complete=not result['native_frame_guards'] and not any(c.get('source_relation')=='unresolved' for c in choices),
                  unresolved_competitor_groups=[c for c in choices if c.get('source_relation')=='unresolved'],
                  failed_components=[error for choice,error in answers if error])
    result['reviewed_arrow_names']=reviewed_arrows
    save(folder/'audit.json',result);return result


def review_legacy(scene,styles,rows,source,shape_pptx,arrow_pptx,folder,runtime):
    """Historical pairwise protocol retained only for read-only replays."""
    from shape_integration.common import save
    from shape_integration.styling.export import render_native
    folder=Path(folder)
    with Image.open(source) as im:clean=im.convert('RGB')
    internal = {}; internal_audit = []
    pairs,parallel=protect_parallel_pairs(rows,nominate(rows),np.asarray(clean))
    table={r['id']:r for r in rows}
    groups=comparison_groups(pairs,rows)
    # The union is the starting product. Only directly competing routes enter
    # the removal set; context and graph connectivity never enlarge it.
    members=[sorted({p[k] for p in group for k in ('a','b')}) for group in groups]
    # Geometry critic owns source completeness. Composition only resolves
    # actual competitors; a second existence vote could contradict that critic
    # and delete an otherwise useful unmatched route from the union.
    unique=[]
    save(folder/'source_only_candidates.json',unique)
    allowed_by_group={}
    for batch,nodes in zip(groups,members):
        allowed_by_group[batch[0]['id']]={
            'allow_empty':all(connection_route(table[n]) for n in nodes),
            'required':[n for n in nodes if table[n].get('immutable_border')]}
    involved={n for group in members for n in group}
    # Source-protected parallel pairs are final too; legacy ownership must not
    # reinterpret them after this review.
    accounted=involved|{p[k] for p in parallel for k in ('a','b')}
    reviewed_arrows=sorted(table[n]['key'] for n in accounted if table[n]['layer']=='arrows')
    save(folder/'pairs.json',pairs)
    save(folder/'distinct_parallel_sources.json',parallel)
    if not groups:return {**{'drop_shape_keys':[],'drop_shape_names':[],'drop_arrow_names':[], 'reviewed_arrow_names':reviewed_arrows, 'model_calls':0,'removed':[]},**internal,'internal_duplicates':internal_audit,'line_ownership_complete':True}
    contexts={}
    for batch,nodes in zip(groups,members):
        # Show the SAME native nodes above either candidate network. Otherwise
        # visible gaps between opaque nodes look like a dashed/missing shaft.
        node_ids={key for n in nodes for key in
                  (*table[n].get('network_nodes',()),*table[n].get('endpoint_nodes',()))}
        selected_keys={key for n in nodes for key in
                       (*table[n]['keys'],table[n].get('parent_key'))}
        context_ids=sorted(node_ids-selected_keys)
        context=None;back_context=None
        if context_ids:
            back=[]
            for key in context_ids:
                b=scene['parts'][key]['bbox_2d']
                # Intrinsic glyphs inside a node must be visible ABOVE its
                # fill. Between-node routes still end underneath node faces.
                if any(np.all(table[n]['points']>=np.asarray(b[:2])-1) and
                       np.all(table[n]['points']<=np.asarray(b[2:])+1) for n in nodes):back.append(key)
            front=[k for k in context_ids if k not in back]
            context=folder/('context_'+batch[0]['id']+'.pptx')
            subset(shape_pptx,context,[scene['parts'][key]['object_id']+'/'+json_part_id(key) for key in front])
            if back:
                back_context=folder/('back_context_'+batch[0]['id']+'.pptx')
                subset(shape_pptx,back_context,[scene['parts'][key]['object_id']+'/'+json_part_id(key) for key in back])
        contexts[batch[0]['id']]=(context,back_context,[scene['parts'][key]['bbox_2d'] for key in sorted(node_ids)])
    def render(job):
        # Native subsets retain each candidate's transforms, heads and paint.
        batch,nodes,candidate=job;ident=(batch[0]['id'],candidate)
        branch=table[candidate]['layer']
        local=folder/('candidate_render_'+ident[0]+'_'+str(nodes.index(candidate)));local.mkdir(exist_ok=True)
        selected_rows=[table[candidate]]
        source=shape_pptx if branch=='shape' else arrow_pptx
        ppt=local/'isolated.pptx';selection_subset(source,ppt,selected_rows)
        context,back_context,_=contexts[batch[0]['id']]
        if back_context is not None:
            shown=local/'with_back_context.pptx'
            merge(back_context,[(branch,ppt)],shown);ppt=shown
        if context is not None:
            shown=local/'with_context.pptx'
            merge(ppt,[('context',context)],shown)
            ppt=shown
        preview=render_native(ppt,local/'render',scene['source_size'],180)
        with Image.open(preview) as im:return ident,im.convert('RGB')
    jobs=[(batch,nodes,n) for batch,nodes in zip(groups,members) for n in nodes]
    with ThreadPoolExecutor(max_workers=4) as pool:native=dict(pool.map(render,jobs))
    def evidence(batch,nodes):
        points=np.concatenate([table[n]['points'] for n in nodes]+
                              [np.asarray(b) for n in nodes for b in table[n].get('attachment_bounds',[])])
        bounds=np.r_[points.min(0)-18,points.max(0)+18]
        # Endpoint matches only locate context. An unresolved match is not a
        # missing connection, and must not become a branch-quality assertion.
        context,back_context,regions=contexts[batch[0]['id']]
        if regions:
            endpoints=np.asarray(regions)
            bounds[:2]=np.minimum(bounds[:2],endpoints[:,:2].min(0)-8)
            bounds[2:]=np.maximum(bounds[2:],endpoints[:,2:].max(0)+8)
        box=tuple(np.rint([max(0,bounds[0]),max(0,bounds[1]),min(clean.width,bounds[2]+1),min(clean.height,bounds[3]+1)]).astype(int))
        panels=[('SOURCE',clean)]
        # Candidates share source coordinates and identical node context.
        for i,n in enumerate(nodes):
            label=f'C{i+1} '+table[n]['layer']+' native candidate + same context'
            panels.append((label,native[(batch[0]['id'],n)]))
        # Endpoint details share the SAME render/crop/scale for both branches.
        # They are evidence in this call, not another selection or critic.
        boxes=[box]
        if max(box[2]-box[0],box[3]-box[1])>240:
            route=max((table[n] for n in nodes),key=lambda r:r['length'])
            for point in route['points'][[0,-1]]:
                b=tuple(np.rint([max(0,point[0]-64),max(0,point[1]-64),min(clean.width,point[0]+65),min(clean.height,point[1]+65)]).astype(int))
                if b not in boxes:boxes.append(b)
        images=[]
        # Bound panel width so a group with several overlapping candidates
        # does not shrink every thin route into an unreadable wide thumbnail.
        # The same request sees all pages with stable C IDs and source crops.
        for start in range(1,len(panels),2):
            page=[panels[0],*panels[start:start+2]]
            sheet=Image.new('RGB',(720*len(page),720*len(boxes)),'#edf0f3');draw=ImageDraw.Draw(sheet)
            for j,region in enumerate(boxes):
                for i,(label,im) in enumerate(page):
                    x,y=i*720,j*720
                    draw.text((x+8,y+8),label+(' endpoint detail' if j else ''),fill='black')
                    tile=ImageOps.contain(im.crop(region),(704,676),Image.Resampling.LANCZOS)
                    sheet.paste(tile,(x+(720-tile.width)//2,y+32+(676-tile.height)//2))
            path=folder/('group_'+batch[0]['id']+f'_{start//2:02d}.png');sheet.save(path)
            images.append(('SOURCE and native candidates '+', '.join(n.split()[0] for n,_ in page[1:]),path))
        return images
    prompt={
        'task':'只为当前候选连接选择保留的C编号。同一条原图连接只保留一个实现，不按分支整批保留。',
        'rules':[
            '面板的坐标、裁剪和缩放相同。候选展示原生线形和头部，并叠加相同节点上下文；只比较当前连接，不评价共同节点、文字或背景。',
            '先看 SOURCE 中实际有哪些连接，再逐一对照两路。漏掉原图连接与新增、重复连接同样是错误，不偏好线多或线少。连接正确后，再比较端点、路径、箭头方向、颜色和粗细。',
            '只检查候选位置；独立的原图连接不属于本次选择。只有原图确有不同路径或互补线段时才保留多个C，不能保留同一路径的重复版本。',
            '遮挡处只认干净SOURCE中同一位置、同一走向的可见延续；旁边的竖直箭头不能证明当前对角线，单个头部不能证明未知起点。不要按全连接、对称性或无法排除来补线。',
            '多线汇聚的尖角不等于箭头头部；放大端点检查是否有独立头部。只比较当前线，不能把邻线或纹理当作它的证据。',
            '原图支持该连接时，即使候选都不完美，也保留最准确的一份。仅当allow_empty=true且所有候选均无原图依据时才返回空列表。required必须保留。'],
        'output':{'keep_ids':['C1'],'reason':'对应的原图路径；多个保留项各自恢复什么独立或互补内容。'}}
    save(folder/'comparison_groups.json',[[p['id'] for p in batch] for batch in groups])
    save(folder/'comparison_members.json',members)
    def check(group):
        batch,nodes=group
        aliases={f'C{i+1}':n for i,n in enumerate(nodes)}
        limits=allowed_by_group[batch[0]['id']]
        required=[a for a,n in aliases.items() if n in limits['required']]
        request={**prompt,'candidates':list(aliases),'allow_empty':limits['allow_empty'],'required':required}
        try:
            choice=runtime.call(folder/('branch_'+batch[0]['id']),'review',
                '只判断当前候选连接。按原图可见路径择优，不按网络先验补线。只返回JSON。',
                request,evidence(batch,nodes),
                lambda d:validate_selection(d,aliases,allow_empty=limits['allow_empty'],required=required),8192)
            return {**choice,'keep_ids':[aliases[a] for a in choice['keep_ids']],
                    'candidate_ids':nodes,'pairs':[p['id'] for p in batch]},None
        except (ValueError,RuntimeError) as exc:
            return None,{'pairs':[p['id'] for p in batch],'error':str(exc)}
    workers=max(1,min(32,int(getattr(runtime,'config',{}).get('api_concurrency',8))))
    with ThreadPoolExecutor(max_workers=workers) as pool:answers=list(pool.map(check,zip(groups,members)))
    choices=[choice for choice,error in answers if choice is not None]
    failures=[error for choice,error in answers if error]
    if failures:
        save(folder/'audit.json',{'state':'failed','branch_choices':choices,'failed_components':failures})
        raise RuntimeError('Branch selection incomplete; keep checkpoints and retry instead of exporting duplicates')
    result=apply_choices(rows,choices)
    for name in ('drop_shape_keys','drop_shape_names'):
        result[name]=sorted(set(result[name])|set(internal.get(name,[])))
    for name in ('shape_segment_suppression','shape_subpath_suppression'):
        for key,indices in internal.get(name,{}).items():
            result.setdefault(name,{})[key]=sorted(set(result.get(name,{}).get(key,[]))|set(indices))
    result['internal_duplicates']=internal_audit
    result.update(branch_choices=choices,model_calls=len(groups),reviewed_pairs=sum(len(b) for b in groups if b[0]['nomination']!='unmatched_source_check'),
                  source_only_candidates=unique,
                  line_ownership_complete=not result['native_frame_guards'],
                  failed_components=[error for choice,error in answers if error])
    result['reviewed_arrow_names']=reviewed_arrows
    save(folder/'audit.json',result);return result
