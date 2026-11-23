"""Reconcile already recovered objects at composition; no model requests.

Only coincident intrinsic strokes and fully represented operator glyphs are
deduplicated. Containers, labels and neighbouring occurrences stay intact.
"""
import math
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def strokes(commands, samples=24):
    out=[];current=[];position=None;start=None
    for c in commands:
        if c[0]=='M':
            if len(current)>1:out.append(np.asarray(current,float))
            position=np.asarray(c[1:3],float);start=position.copy();current=[position]
        elif c[0]=='L':
            position=np.asarray(c[1:3],float);current.append(position)
        elif c[0]=='C' and position is not None:
            p=np.array([position,c[1:3],c[3:5],c[5:7]],float)
            for t in np.linspace(0,1,samples)[1:]:
                current.append((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3])
            position=p[-1]
        elif c[0]=='Z' and start is not None:current.append(start)
    if len(current)>1:out.append(np.asarray(current,float))
    return out


def surviving_scene(scene,plan):
    """Read the same surviving disconnected strokes as the native merge."""
    dropped=set(plan.get('drop_shape_keys',()));suppression=plan.get('shape_subpath_suppression',{})
    parts={}
    for key,part in scene['parts'].items():
        if key in dropped:continue
        ref=part['object_id']+'/'+part.get('id',json_part_id(key));indices=set(suppression.get(ref,()))
        if indices:
            commands=part.get('commands',[])
            if any(c[0]=='Z' for c in commands):raise ValueError('Subpath ownership requires open strokes')
            groups=[]
            for c in commands:
                if c[0]=='M':groups.append([])
                if not groups:raise ValueError('Invalid disconnected contour')
                groups[-1].append(c)
            if not indices<=set(range(len(groups))):raise ValueError('Invalid scene subpath index')
            commands=[c for i,g in enumerate(groups) if i not in indices for c in g]
            if not commands:continue
            part={**part,'commands':commands}
        parts[key]=part
    return {**scene,'parts':parts}


def resample(lines,count=80):
    out=[]
    for p in lines:
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        keep=np.r_[True,np.diff(length)>1e-8];p,length=p[keep],length[keep]
        if len(p)<2:continue
        t=np.linspace(0,length[-1],count)
        out.append(np.c_[np.interp(t,length,p[:,0]),np.interp(t,length,p[:,1])])
    return np.concatenate(out) if out else np.empty((0,2))


def distances(points,lines):
    segments=[(p[:-1],p[1:]) for p in lines if len(p)>1]
    if not segments:return np.full(len(points),np.inf)
    a=np.concatenate([v[0] for v in segments]);b=np.concatenate([v[1] for v in segments]);v=b-a
    delta=points[:,None]-a;den=np.maximum((v*v).sum(1),1e-12)
    t=np.clip((delta*v).sum(2)/den,0,1)
    return np.linalg.norm(delta-t[...,None]*v,axis=2).min(1)


def arrow_strokes(row):
    if row.get('beziers'):
        commands=[['M',*row['beziers'][0][0]]]
        commands.extend(['C',*np.asarray(c[1:]).reshape(-1)] for c in row['beziers'])
        return strokes(commands)
    return [np.array([row['start'],*row.get('via',[]),row['end']],float)]


def duplicate_arrow_routes(arrows, source):
    """Remove only coincident straight routes when CLEAN shows one source stroke."""
    if source is None:return [],[]
    from .line_identity import compatible, separate_parallel_sources, single_source_stroke
    with Image.open(source) as opened:rgb=np.asarray(opened.convert('RGB'))
    candidates=[]
    for row in arrows:
        if row.get('kind')=='block_arrow':continue
        lines=arrow_strokes(row)
        if len(lines)!=1 or len(lines[0])<2:continue
        points=lines[0]
        if np.quantile(distances(points,[points[[0,-1]]]),.95)>1.:continue
        length=float(np.linalg.norm(points[-1]-points[0]))
        style=row.get('style',{});color=style.get('color') or style.get('stroke')
        if length<5 or not isinstance(color,str) or len(color)!=7 or not color.startswith('#'):continue
        try:int(color[1:],16)
        except ValueError:continue
        candidates.append((row,points[[0,-1]],length,color,float(style.get('width_px') or 1.)))

    drop=[];audit=[]
    dropped=set()
    for index,(a,pa,la,ca,wa) in enumerate(candidates):
        if a['id'] in dropped:continue
        sa=a.get('style',{})
        for b,pb,lb,cb,wb in candidates[index+1:]:
            if b['id'] in dropped or a.get('heads',['none','none'])!=b.get('heads',['none','none']):continue
            sb=b.get('style',{})
            if any(sa.get(end)!=sb.get(end) for end in ('start','end')):continue
            if not compatible({'stroke':ca,'width_px':wa,'dash':sa.get('dash','solid')},
                              {'color':cb,'width_px':wb,'dash':sb.get('dash','solid')}):continue
            if not .95<=min(la,lb)/max(la,lb):continue
            reverse=np.linalg.norm(pa[0]-pb[1])+np.linalg.norm(pa[1]-pb[0])<np.linalg.norm(pa[0]-pb[0])+np.linalg.norm(pa[1]-pb[1])
            other=pb[::-1] if reverse else pb
            endpoint=max(np.linalg.norm(pa[0]-other[0]),np.linalg.norm(pa[1]-other[1]))
            tolerance=max(1.5,.5*min(wa,wb))
            if endpoint>tolerance:continue
            if max(float(np.quantile(distances(resample([pa]),[other]),.9)),
                   float(np.quantile(distances(resample([other]),[pa]),.9)))>tolerance:continue
            if separate_parallel_sources(rgb,pa,other,ca,tolerance,cb):continue
            score=max(single_source_stroke(rgb,pa,other,color,tolerance,max(wa,wb))
                      for color in {ca,cb})
            if score<.95:continue
            drop.append(b['id']);dropped.add(b['id'])
            audit.append({'arrow':b['id'],'duplicate_of':a['id'],
                'single_source_stroke_fraction':float(score),
                'reason':'same_complete_straight_route_confirmed_by_single_source_stroke'})
    return sorted(set(drop)),audit


def has_matching_heads(row,part,scene,styles):
    ends=[]
    for end,head in zip(('start','end'),row.get('heads',['none','none'])):
        actual=row.get('style',{}).get(end)
        if actual:
            head=actual.get('type',head) if isinstance(actual,dict) else actual
        if head not in ('none',None):ends.append((np.asarray(row[end],float),head))
    if not ends:return True
    owner=part['object_id'];commands=part['commands']
    # Open V heads can be additional pen strokes in the same shape path.
    branches=strokes(commands) if sum(c[0]=='M' for c in commands)>1 else []
    triangles=[]
    for key,p in scene['parts'].items():
        if p.get('object_id')!=owner or p.get('status')!='ok' or not styles.get(key,{}).get('fill'):continue
        cs=p.get('commands',[])
        if p.get('kind')!='path' or not cs or cs[-1][0]!='Z':continue
        vertices=[c[1:3] for c in cs if c[0] in ('M','L')]
        unique=np.unique(vertices,axis=0)
        if len(unique)==3:triangles.append(unique)
    width=max(styles.get(next((k for k,p in scene['parts'].items() if p is part),''),{}).get('width_px',1),
              row.get('style',{}).get('width_px') or 1)
    reach=max(3.,4*width)
    for end,head in ends:
        # Preserve endpoint dots/diamonds and other heads until their same
        # native geometry is demonstrated, rather than mistaking a nearby
        # triangle or node for an arbitrary endpoint marker.
        options=triangles if head=='triangle' else branches if head in ('open','open_arrow','open_triangle') else []
        if not any(np.linalg.norm(p-end,axis=1).min()<=reach and np.ptp(p,axis=0).max()<=3*reach
                   for p in options):return False
    return True


def duplicate_arrows(scene,styles,arrows,source=None,*,represented_image_parts=None):
    from .line_identity import compatible,single_source_stroke,node_pair,separate_parallel_sources
    rgb=None
    if source is not None:
        with Image.open(source) as opened:rgb=np.asarray(opened.convert('RGB'))
    candidates=[]
    for key,p in scene['parts'].items():
        cs=p.get('commands',[]);s=styles.get(key,{})
        if p.get('status')!='ok' or not cs or cs[-1][0]=='Z' or not s.get('stroke'):continue
        lines=strokes(cs)
        if lines:candidates.append((key,p,s,lines,np.concatenate(lines)))
    drop=[];audit=[]
    for row in arrows:
        if row.get('kind')=='block_arrow':
            # A second opaque contour paints over a shape's already ordered
            # children. Require mutual area coverage, never mere containment.
            contours=row.get('contours',[])
            polygons=[Polygon(c) for c in contours if len(c)>=3]
            if not polygons or any(not p.is_valid for p in polygons):continue
            # SAM may fragment one filled owner around labels/overlays. Match
            # its registered surface, not an arbitrary contour-list length.
            polygon=unary_union(polygons)
            if not polygon.is_valid or polygon.area<=0:continue
            matches=[]
            for key,p in {**scene['parts'],**(represented_image_parts or {})}.items():
                cs=p.get('commands',[]);fill=styles.get(key,{}).get('fill')
                color=row.get('style',{}).get('color')
                if p.get('status')!='ok' or not isinstance(fill,str) or not isinstance(color,str):continue
                if fill.upper()!=color.upper() and np.linalg.norm(np.array([int(fill[i:i+2],16) for i in (1,3,5)])-
                        np.array([int(color[i:i+2],16) for i in (1,3,5)]))>24:continue
                if not cs or cs[-1][0]!='Z':continue
                paths=strokes(cs)
                if len(paths)!=1:continue
                target=Polygon(paths[0])
                if not target.is_valid or target.area<=0:continue
                overlap=polygon.intersection(target).area
                coverage=min(overlap/polygon.area,overlap/target.area)
                if coverage>=.88:matches.append((coverage,key,p))
            if len(matches)==1:
                coverage,key,p=matches[0];drop.append(row['id'])
                audit.append({'arrow':row['id'],'shape_part':key,'shape_owner':p['object_id'],
                    'mutual_area_coverage':coverage,
                    'reason':('same_filled_contour_represented_in_image' if key in (represented_image_parts or {})
                              else 'same_filled_contour_already_ordered_in_shape')})
            continue
        lines=arrow_strokes(row);q=resample(lines)
        if len(q)<2:continue
        matches=[]
        for key,p,s,target,points in candidates:
            if not compatible(s,row.get('style',{})):continue
            tol=max(1.5,min(3.,1.25*(s.get('width_px',1)+(row.get('style',{}).get('width_px') or 1))))
            if np.any(q.max(0)<points.min(0)-tol) or np.any(q.min(0)>points.max(0)+tol):continue
            distance=distances(q,target);fraction=float(np.mean(distance<=tol))
            if fraction<.9 or not has_matching_heads(row,p,scene,styles):continue
            if rgb is not None and len(lines)==len(target)==1 and len(lines[0])==len(target[0])==2 and separate_parallel_sources(
                    rgb,target[0],lines[0],s['stroke'],tol):continue
            matches.append((float(np.quantile(distance,.9)),key,p,fraction))
        if matches and len({m[2]['object_id'] for m in matches})==1:
            distance,key,p,fraction=min(matches,key=lambda m:m[0])
            drop.append(row['id']);audit.append({'arrow':row['id'],'shape_part':key,
                'shape_owner':p['object_id'],'supported_fraction':fraction,
                'distance_p90_px':distance,
                'reason':'same_intrinsic_stroke_already_represented_by_shape'})
            continue
        # Independent fits often differ by a few pixels. Extend only for a
        # unique complete straight route with matching type/weight, matched
        # heads, endpoints and one observed source stroke between predictions.
        if rgb is None or len(lines)!=1 or len(lines[0])!=2:continue
        route=lines[0];length=np.linalg.norm(route[1]-route[0]);wide=[]
        for key,p,s,target,points in candidates:
            if len(target)!=1 or len(target[0])!=2 or not compatible(s,row.get('style',{})):continue
            fitted=target[0];span=np.linalg.norm(fitted[1]-fitted[0])
            if not .8<=length/max(span,1e-9)<=1.25:continue
            tol=max(4.,min(12.,.065*min(length,span)))
            endpoint=min(max(np.linalg.norm(route[i]-fitted[i]) for i in (0,1)),
                         max(np.linalg.norm(route[i]-fitted[1-i]) for i in (0,1)))
            if endpoint>1.5*tol:continue
            if np.quantile(distances(q,target),.9)>tol:continue
            if not has_matching_heads(row,p,scene,styles):continue
            score=single_source_stroke(rgb,fitted,route,s['stroke'],tol,
                max(s.get('width_px') or 1,row.get('style',{}).get('width_px') or 1))
            if score>=.8:wide.append((endpoint,key,p,score))
        wide.sort(key=lambda m:m[0])
        if wide and (len(wide)==1 or wide[1][0]-wide[0][0]>2):
            endpoint,key,p,single=wide[0];drop.append(row['id'])
            audit.append({'arrow':row['id'],'shape_part':key,'shape_owner':p['object_id'],
                'endpoint_error_px':float(endpoint),'single_source_stroke_fraction':single,
                'reason':'same_complete_straight_route_confirmed_by_source'})
    # Intrinsic networks supply a stronger identity than pixel proximity:
    # the same two independently recovered nodes joined by one unique edge.
    # Require one-to-one correspondence in BOTH branches. Parallel multi-edges,
    # unmatched heads and different source styles remain separate.
    if source is not None:
        nodes=[]
        for key,p in scene['parts'].items():
            if p.get('status')!='ok' or p.get('kind')!='ellipse':continue
            b=np.asarray(p.get('bbox_2d',[]),float)
            if b.shape!=(4,):continue
            radius=(b[2:]-b[:2])/2
            if min(radius)<=1 or max(radius)>1.6*min(radius):continue
            nodes.append((key,(b[:2]+b[2:])/2,radius))
        graph={};routes={}
        for candidate in candidates:
            key,p,s,target,points=candidate
            if len(target)!=1 or len(target[0])!=2:continue
            owned=[node for node in nodes if scene['parts'][node[0]].get('object_id')==p['object_id']]
            pair=node_pair(target[0],owned)
            if pair:graph.setdefault(pair,[]).append(candidate)
        for row in arrows:
            if row.get('kind')=='block_arrow':continue
            target=arrow_strokes(row)
            if len(target)!=1 or len(target[0])!=2:continue
            pair=node_pair(target[0],nodes)
            if pair:routes.setdefault(pair,[]).append(row)
        for pair,registered in graph.items():
            linked=routes.get(pair,[])
            if len(registered)!=1 or len(linked)!=1:continue
            row=linked[0]
            if row['id'] in drop:continue
            key,p,s,target,points=registered[0];route=arrow_strokes(row)[0];fitted=target[0]
            a=fitted[1]-fitted[0];b=route[1]-route[0]
            cosine=abs(float(a@b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9)
            if cosine<.97 or not compatible(s,row.get('style',{})) or not has_matching_heads(row,p,scene,styles):continue
            if separate_parallel_sources(rgb,fitted,route,s['stroke'],max(4.,min(12.,.065*np.linalg.norm(a)))):continue
            if np.quantile(distances(resample([route]),target),.9)>max(12.,.06*np.linalg.norm(a)):continue
            drop.append(row['id']);audit.append({'arrow':row['id'],'shape_part':key,'shape_owner':p['object_id'],
                'endpoint_node_keys':list(pair),'shape_line_style':s,'arrow_line_style':row.get('style',{}),
                'reason':'same_unique_node_pair_and_compatible_intrinsic_edge'})
    return drop,audit


def duplicate_ellipsis_text(scene,styles,rows):
    """Three coincident filled dots own a standalone ellipsis, not nearby labels.

    Match the recovered geometry, with no pixel threshold or extra model pass.
    Missing dots, hollow nodes and punctuation inside an expression survive.
    """
    drop=[];audit=[]
    directions={'...':np.array([1.,0.]),'…':np.array([1.,0.]),'⋯':np.array([1.,0.]),
                '⋮':np.array([0.,1.]),'⋱':np.array([1.,1.])/math.sqrt(2),
                '⋰':np.array([1.,-1.])/math.sqrt(2)}
    for row in rows:
        value=''.join(row.get('text','').split())
        if value not in directions:continue
        b=np.asarray(row['bbox_2d'],float);size=b[2:]-b[:2]
        if min(size)<=0:continue
        direction=directions[value];normal=np.array([-direction[1],direction[0]])
        extent=float(abs(direction)@size);dots=[]
        for key,p in scene['parts'].items():
            style=styles.get(key,{})
            if p.get('status')!='ok' or p.get('kind')!='ellipse' or not isinstance(style.get('fill'),str):continue
            bb=np.asarray(p['bbox_2d'],float);diam=bb[2:]-bb[:2];center=(bb[:2]+bb[2:])/2
            if min(diam)<=0 or max(diam)>1.5*min(diam) or max(diam)>.3*extent:continue
            # Text frames contain ascender space; dot centers, not full shape
            # boxes, must lie in the same occurrence with a small margin.
            if np.any(center<b[:2]-.15*size) or np.any(center>b[2:]+.15*size):continue
            # Independent paint recovery may disagree on colour for the same dots.
            dots.append((key,center,float(np.mean(diam))))
        if len(dots)!=3:continue
        dots.sort(key=lambda d:float(d[1]@direction));centers=np.array([d[1] for d in dots])
        diameter=np.array([d[2] for d in dots]);spacing=np.diff(centers@direction)
        if max(diameter)>1.8*min(diameter) or min(spacing)<max(diameter):continue
        if max(spacing)>1.7*min(spacing) or np.ptp(centers@normal)>.6*max(diameter):continue
        span=float(np.ptp(centers@direction)+np.mean(diameter))
        if not .5*extent<=span<=1.3*extent:continue
        drop.append(row['id']);audit.append({'text_id':row['id'],'text':row['text'],
            'shape_parts':[d[0] for d in dots],'reason':'complete_three_dot_ellipsis_already_present'})
    return drop,audit


def ellipsis_shape_ownership(scene,styles,rows,source,arrows=()):
    """One source dot-group matcher shared with native text replacement."""
    from .source_dot_ownership import resolve
    return resolve(scene,styles,rows,source,arrows)


def split_brace_duplicates(scene,styles,arrows,source,excluded=()):
    """One registered brace may also have been recovered as two headless arms.

    Require their shared physical owner, complete mutual contour coverage and
    one visible source stroke across normals. Parallel/nested braces survive.
    This only reconciles existing native objects; it never changes inference.
    """
    if source is None:return [],[]
    owners={}
    for row in arrows:
        owner=row.get('physical_target_id') or row.get('geometry_owner_id')
        if (row['id'] in excluded or not owner or row.get('kind')!='curve'
                or any(h not in ('none',None) for h in row.get('heads',[]))):continue
        owners.setdefault(owner,[]).append(row)
    braces=[]
    for key,p in scene['parts'].items():
        candidate=scene.get('objects',{}).get(p.get('object_id'),{}).get('candidate',{})
        cs=p.get('commands',[]);s=styles.get(key,{})
        if ('brace' not in candidate.get('kind','').lower() or p.get('status')!='ok'
                or not s.get('stroke') or not cs or cs[-1][0]=='Z'):continue
        lines=strokes(cs)
        if len(lines)==1:braces.append((key,p,s,lines))
    if not braces or not any(len(rows)>=2 for rows in owners.values()):return [],[]
    with Image.open(source) as opened:rgb=np.asarray(opened.convert('RGB')).astype(float)
    drop=[];audit=[]
    def length(lines):return sum(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in lines)
    for owner,rows in owners.items():
        if len(rows)<2:continue
        routes=[p for row in rows for p in arrow_strokes(row)]
        matches=[]
        for key,p,s,lines in braces:
            tolerance=max(3.,min(10.,2.5*(s.get('width_px') or 1.)))
            target=resample(lines,140);ratio=length(routes)/max(length(lines),1e-9)
            if not .8<=ratio<=1.25:continue
            if any(np.mean(distances(resample(arrow_strokes(r)),lines)<=tolerance)<.9 for r in rows):continue
            if np.mean(distances(target,routes)<=tolerance)<.9:continue
            # A broad tolerance is safe only when CLEAN contains one stroke,
            # not two nearby occurrences that the geometry happened to merge.
            tangent=np.gradient(target,axis=0);norm=np.linalg.norm(tangent,axis=1)
            normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(norm[:,None],1e-9)
            offsets=np.arange(-int(np.ceil(tolerance))-2,int(np.ceil(tolerance))+3)
            xy=np.rint(target[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
            valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
            xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
            pigment=np.array([int(s['stroke'][i:i+2],16) for i in (1,3,5)])
            ink=np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]]-pigment,axis=2)<90
            runs=np.sum(np.diff(np.pad(ink.astype(int),((0,0),(1,1))),axis=1)==1,axis=1)
            single=float(np.mean(runs[valid]==1)) if valid.sum()>=30 else 0.
            if single>=.85:matches.append((key,p,single))
        if len(matches)!=1:continue
        key,p,single=matches[0]
        drop.extend(r['id'] for r in rows)
        audit.append({'arrows':[r['id'] for r in rows],'physical_target_id':owner,
            'shape_part':key,'shape_owner':p['object_id'],'single_source_stroke_fraction':single,
            'reason':'split_headless_brace_already_represented_by_registered_shape'})
    return drop,audit


def duplicate_operator_text(scene,styles,rows):
    """Require the operator's actual strokes, not just an overlapping box."""
    drop=[];audit=[]
    for row in rows:
        value=row.get('text','').strip()
        if value not in ('+','−','×','⊕','⊗'):continue
        box=row['bbox_2d'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        w=max(box[2]-box[0],1);h=max(box[3]-box[1],1);c=np.array([cx,cy])
        horizontal=[];vertical=[];rising=[];falling=[];rings=[]
        for key,p in scene['parts'].items():
            if p.get('status')!='ok':continue
            s=styles.get(key,{})
            if not (s.get('stroke') or s.get('fill')):continue
            b=p['bbox_2d'];center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
            if np.any(abs(center-c)>np.array([w,h])*.35):continue
            if b[2]-b[0]>1.6*w or b[3]-b[1]>1.6*h:continue
            if p.get('kind')=='ellipse' and min(b[2]-b[0],b[3]-b[1])>.55*min(w,h):rings.append(key)
            # A geometric plus/minus may use filled rectangular bars instead
            # of stroked paths. Treat both representations as the same glyph.
            if p.get('kind')=='rectangle' and s.get('fill'):
                bw,bh=b[2]-b[0],b[3]-b[1]
                if bw>4*max(bh,.01) and bw>.4*w:horizontal.append(key)
                if bh>4*max(bw,.01) and bh>.4*h:vertical.append(key)
            for line in strokes(p['commands']):
                if len(line)!=2:continue
                delta=line[1]-line[0];length=np.linalg.norm(delta)
                if length<.4*min(w,h):continue
                middle=(line[0]+line[1])/2
                if np.any(abs(middle-c)>np.array([w,h])*.3):continue
                angle=math.degrees(math.atan2(delta[1],delta[0]))%180
                if min(angle,180-angle)<12:horizontal.append(key)
                elif abs(angle-90)<12:vertical.append(key)
                elif 25<angle<65:falling.append(key)
                elif 115<angle<155:rising.append(key)
        cross=horizontal and vertical if value in ('+','⊕') else rising and falling
        represented=bool(horizontal) if value=='−' else bool(cross)
        if value in ('⊕','⊗'):represented=represented and bool(rings)
        if represented:
            drop.append(row['id']);audit.append({'text_id':row['id'],'text':value,
                'shape_parts':sorted(set(horizontal+vertical+rising+falling+rings)),
                'reason':'complete_geometric_operator_already_present'})
    return drop,audit


def bracket_kind(commands):
    """Recognise a complete three-stroke square delimiter, never a box side."""
    if [c[0] for c in commands]!=['M','L','L','L']:return None
    q=np.asarray([c[1:3] for c in commands],float);d=np.diff(q,axis=0)
    h=abs(d[1,1]);w=max(abs(d[0,0]),abs(d[2,0]))
    if h<=0 or not .03*h<=w<=.5*h:return None
    if abs(d[1,0])>.08*h or max(abs(d[0,1]),abs(d[2,1]))>.15*w:return None
    if d[0,0]*d[2,0]>=0 or min(abs(d[0,0]),abs(d[2,0]))<.5*w:return None
    return '[' if q[[0,3],0].mean()>q[[1,2],0].mean() else ']'


def duplicate_bracket_text(scene,styles,rows,source):
    """Resolve standalone delimiters only with one source-supported occurrence.

    Nearby boxes alone cannot prove duplication: nested formula delimiters must
    survive. Embedded formula text is never removed by this reconciliation.
    """
    if source is None:return [],[]
    import cv2
    with Image.open(source) as opened:rgb=np.asarray(opened.convert('RGB')).astype(float)
    candidates=[]
    for key,p in scene['parts'].items():
        s=styles.get(key,{})
        if p.get('status')!='ok' or not s.get('stroke') or s.get('fill'):continue
        kind=bracket_kind(p.get('commands',[]))
        if kind:candidates.append((key,p,s,kind))
    drop=[];audit=[]
    for row in rows:
        kind=row.get('text','').strip()
        if kind not in ('[',']'):continue
        b=np.asarray(row['bbox_2d'],float);w,h=b[2:]-b[:2]
        if min(w,h)<=0:continue
        matches=[]
        for key,p,s,value in candidates:
            if value!=kind:continue
            a=np.asarray(p['bbox_2d'],float);pw,ph=a[2:]-a[:2]
            overlap=max(0,min(a[3],b[3])-max(a[1],b[1]))
            if not .5<=ph/h<=2 or overlap<.65*min(ph,h):continue
            if abs((a[0]+a[2]-b[0]-b[2])/2)>max(w,pw,.5*h):continue
            pigment=np.array([int(s['stroke'][i:i+2],16) for i in (1,3,5)])
            colour=row.get('color','#000000')
            if not isinstance(colour,str) or np.linalg.norm(pigment-np.array([int(colour[i:i+2],16) for i in (1,3,5)]))>40:continue
            lo=np.maximum(0,np.floor(np.minimum(a[:2],b[:2])-3)).astype(int)
            hi=np.minimum(rgb.shape[1::-1],np.ceil(np.maximum(a[2:],b[2:])+4)).astype(int)
            if np.any(hi<=lo):continue
            ink=(np.linalg.norm(rgb[lo[1]:hi[1],lo[0]:hi[0]]-pigment,axis=-1)<60).astype('uint8')
            count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
            occurrences=[]
            for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                if not .45*h<=ch<=1.6*h or not .03*ch<=cw<=.5*ch or area<8:continue
                component=labels[y:y+ch,x:x+cw]==index
                # Both short arms and the middle spine are necessary. A
                # neighbouring cell border clipped by this ROI is incomplete.
                spine=component[:,:max(1,int(.3*cw))] if kind=='[' else component[:,int(.7*cw):]
                arm=max(1,int(.2*ch))
                if spine.any(1).mean()<.65 or component[:arm].any(0).mean()<.65 or component[-arm:].any(0).mean()<.65:continue
                other=component[arm:-arm,int(.7*cw):] if kind=='[' else component[arm:-arm,:max(1,int(.3*cw))]
                if other.size and other.any(1).mean()>.3:continue
                occurrences.append(np.r_[lo+[x,y],lo+[x+cw-1,y+ch-1]].tolist())
            if len(occurrences)!=1:continue
            observed=np.asarray(occurrences[0],float)
            # Geometry must actually follow that source delimiter. A poor fit
            # cannot suppress a potentially better text reconstruction.
            tolerance=max(2.,s.get('width_px') or 1.)
            if np.max(abs(a-observed))>tolerance:continue
            matches.append((key,p,occurrences[0]))
        if len(matches)==1:
            key,p,observed=matches[0];drop.append(row['id'])
            audit.append({'text_id':row['id'],'text':kind,'shape_parts':[key],
                          'source_delimiter_bbox':observed,'source_occurrences':1,
                          'reason':'standalone_delimiter_already_represented_by_source_aligned_geometry'})
    return drop,audit


def source_occlusion(scene,styles,arrows,source,excluded,*,part_placements=False,texts=()):
    """Place a line behind a filled object only where source pixels hide it."""
    if source is None:return {},[]
    with Image.open(source) as image:rgb=np.asarray(image.convert('RGB'))
    from shapely import affinity
    from shapely.geometry import box as text_box
    text_regions=[]
    for text in texts:
        if 'frame_px' not in text:continue
        x,y,w,h=text['frame_px'];geometry=text_box(x,y,x+w,y+h)
        text_regions.append(affinity.rotate(geometry,text.get('rotation_deg',0),origin=(x+w/2,y+h/2)))
    text_union=unary_union(text_regions)
    fills=[]
    for key,p in scene['parts'].items():
        s=styles.get(key,{})
        if p.get('status')!='ok' or not isinstance(s.get('fill'),str):continue
        opacity=s.get('fill_opacity',1)
        if opacity<=0:continue
        cs=p.get('commands',[])
        if not cs or cs[-1][0]!='Z':continue
        contours=strokes(cs)
        if p.get('kind')=='cylinder' and len(contours)==2:
            contours=contours[-1:]  # Native rim seam is not a second filled face.
        if len(contours)!=1:continue
        geom=Polygon(contours[0])
        if not geom.is_valid or geom.area<=0:continue
        inner=geom.buffer(-max(.5,(s.get('width_px') or 0)/2))
        if not inner.is_empty:
            color=np.array([int(s['fill'][i:i+2],16) for i in (1,3,5)])
            fills.append((key,p['object_id'],inner,color,opacity))
    placements={};audit=[]
    for row in arrows:
        if row['id'] in excluded:continue
        if row.get('kind')=='block_arrow':
            # Source-visible native details stay in front of an opaque block
            # contour. This reuses the same stroke-in-face evidence as shape
            # internal paint order; it never removes or changes either object.
            from shapely.geometry import LineString
            from shape_integration.styling.occlusion import visible_stroke_in_face
            paints=list(row.get('style',{}).get('shape_parts',{}).values())
            pigment=[p.get('fill') for p in paints]
            if not all(isinstance(p,str) for p in pigment):continue
            colors=set(pigment)
            contours=[Polygon(c) for c in row.get('contours',()) if len(c)>=3]
            if (not part_placements or not paints or len(colors)!=1 or
                    not all(isinstance(c,str) for c in colors) or
                    any(p.get('fill_opacity',1)!=1 for p in paints) or
                    not contours or any(not c.is_valid for c in contours)):continue
            face=unary_union(contours);fill={'fill':next(iter(colors))};before=[]
            # Filled labels can have the same border pigment as the block.
            # Observe their interior paint as well as the existing edge test.
            backdrop=np.array([int(fill['fill'][i:i+2],16) for i in (1,3,5)])
            for key,owner,geom,pigment,opacity in fills:
                target=face.intersection(geom).buffer(-2)
                if target.is_empty or target.area<16:continue
                expected=opacity*pigment+(1-opacity)*backdrop
                if np.linalg.norm(expected-backdrop)<30:continue
                x0,y0,x1,y1=target.bounds
                xx,yy=np.meshgrid(np.linspace(x0,x1,24),np.linspace(y0,y1,24))
                points=np.c_[xx.ravel(),yy.ravel()]
                valid=np.array([target.contains(Point(p)) and not text_union.covers(Point(p)) for p in points])
                xy=np.rint(points[valid]).astype(int)
                if len(xy)<12:continue
                xy[:,0]=np.clip(xy[:,0],0,rgb.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,rgb.shape[0]-1)
                pixels=rgb[xy[:,1],xy[:,0]].astype(float)
                front=np.linalg.norm(pixels-expected,axis=1);back=np.linalg.norm(pixels-backdrop,axis=1)
                support=float(np.mean((front<45)&(front+12<back)))
                if support<.8:continue
                name=owner+'/'+scene['parts'][key].get('id',json_part_id(key));before.append(name)
                audit.append({'arrow':row['id'],'shape_part':key,'shape_owner':owner,
                              'placement_requested':True,'applied':False,
                              'reason':'source_visible_native_fill_in_filled_contour',
                              'supported_fraction':support,'source_samples':len(xy)})
            for key,part in scene['parts'].items():
                style=styles.get(key,{})
                if part.get('status')!='ok' or not style.get('stroke'):continue
                lines=[LineString(p) for p in strokes(part.get('commands',())) if len(p)>1]
                if not lines:continue
                support=visible_stroke_in_face(unary_union(lines),face,style,fill,rgb)
                if support is None:continue
                name=part['object_id']+'/'+part.get('id',json_part_id(key));before.append(name)
                audit.append({'arrow':row['id'],'shape_part':key,'shape_owner':part['object_id'],
                    'placement_requested':True,'applied':False,
                    'reason':'source_visible_native_stroke_in_filled_contour','source_support':support})
            if before:placements[row['id']]={'before':sorted(set(before)),'after':[]}
            continue
        routes=arrow_strokes(row)
        length=sum(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in routes)
        q=resample(routes,min(2048,max(80,math.ceil(length))))
        if not len(q):continue
        xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,rgb.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,rgb.shape[0]-1)
        pixels=rgb[xy[:,1],xy[:,0]].astype(float)
        color=row.get('style',{}).get('color')
        if not isinstance(color,str):continue
        ink=np.array([int(color[i:i+2],16) for i in (1,3,5)])
        ink_distance=np.linalg.norm(pixels-ink,axis=1)
        points=[Point(p) for p in q]
        # A fitted centerline can miss a one-pixel source stroke. Inspect a
        # bounded normal band, rather than treating background at that one
        # coordinate as proof that a shape hides the complete native route.
        tangent=np.gradient(q,axis=0);norm=np.linalg.norm(tangent,axis=1)
        normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(norm[:,None],1e-8)
        radius=max(2.,min(6.,(row.get('style',{}).get('width_px') or 1.)*.9+1.))
        offsets=np.arange(-radius,radius+.25,.5)
        band=np.rint(q[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
        band[:,:,0]=np.clip(band[:,:,0],0,rgb.shape[1]-1);band[:,:,1]=np.clip(band[:,:,1],0,rgb.shape[0]-1)
        band_pixels=rgb[band[:,:,1],band[:,:,0]].astype(float)
        band_ink=np.linalg.norm(band_pixels-ink,axis=2)
        labelled=np.array([text_union.contains(p) for p in points]) if not text_union.is_empty else np.zeros(len(q),bool)
        before=[];after=[]
        for key,owner,geom,fill,opacity in fills:
            part=scene['parts'][key]
            measured=part.get('diagnostics_local',{}).get('source_primitive_registration',{})
            if measured.get('perimeter_support',0)>=.8 and opacity==1:
                # Source-measured opaque nodes retain their visible perimeter
                # above an attached native head. The centerline alone can end
                # outside the node while its oval/triangle paint overlaps it.
                width=row.get('style',{}).get('width_px')or 1.
                ends=[Point(path[index])for path in routes for index in (0,-1)]
                attached=any(geom.buffer(3*width).covers(point)for point in ends)
                probe=geom.buffer(-max(1.,width/2))
                if attached and not probe.is_empty:
                    b=np.asarray(probe.bounds);lo=np.maximum(0,np.floor(b[:2])).astype(int)
                    hi=np.minimum(rgb.shape[1::-1],np.ceil(b[2:])).astype(int)
                    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
                    inside_face=np.array([probe.contains(Point(x,y))for x,y in zip(xx.ravel(),yy.ravel())]).reshape(xx.shape)
                    values=rgb[lo[1]:hi[1],lo[0]:hi[0]][inside_face].astype(float)
                    contrast=np.linalg.norm(ink-fill)
                    visible_ink=(np.linalg.norm(values-ink,axis=1)+12<np.linalg.norm(values-fill,axis=1)) if contrast>35 else np.zeros(len(values),bool)
                    if len(values)>=12 and np.mean(np.linalg.norm(values-fill,axis=1)<20)>=.9 and visible_ink.mean()<.05:
                        name=owner+'/'+part.get('id',json_part_id(key));before.append(name)
                        audit.append({'arrow':row['id'],'shape_owner':owner,'shape_part':key,'placement_requested':True,
                            'applied':False,'reason':'complete_source_node_perimeter_above_attached_native_head'})
                        continue
            # The normal search band must stay inside the opaque face. Its
            # own black border is not evidence of a shaft crossing its fill.
            core=geom.buffer(-radius)
            inside=np.array([core.contains(p) for p in points])&~labelled
            if inside.sum()<3:continue
            part=scene['parts'][key]
            name=owner+'/'+part.get('id',json_part_id(key))
            band_fill=np.linalg.norm(band_pixels-fill,axis=2)
            if opacity<1:
                # A translucent face attenuates a rear connector rather than
                # hiding it. Use the known paint mixture, not opaque-fill proof.
                mixture=opacity*fill+(1-opacity)*ink
                band_mix=np.linalg.norm(band_pixels-mixture,axis=2)
                tinted=np.any((band_mix<35)&(band_mix+12<band_ink)&(band_mix+12<band_fill),axis=1)
                raw_visible=np.any((band_ink<35)&(band_ink+12<band_mix),axis=1)
                if np.mean(tinted[inside])>=.65 and np.mean(raw_visible[inside])<.1:
                    if part_placements:before.append(name)
                    else:placements.setdefault(row['id'],[]).append(owner)
                    audit.append({'arrow':row['id'],'shape_owner':owner,'shape_part':key,
                        'applied':not part_placements,'placement_requested':part_placements,
                        'reason':'source_translucent_face_tints_stroke','fill_opacity':opacity,
                        'source_tinted_samples':int(np.sum(inside&tinted))})
                elif np.any(inside&raw_visible):after.append(name)
                continue
            visible=np.any((band_ink<70)&(band_ink+12<band_fill),axis=1)
            hidden=(np.linalg.norm(pixels-fill,axis=1)+12<ink_distance)&~visible
            # An isolated crossing glyph can cover several resampled points.
            # Require source-visible ink to persist along the route beyond its
            # normal search corridor, or cover a substantial part of this face.
            # Express the run in source pixels, independent of sample density.
            evidence=inside&visible
            edge=np.diff(np.r_[False,evidence,False].astype(int))
            starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
            step=float(np.median(np.linalg.norm(np.diff(q,axis=0),axis=1)))
            span=max(3.,2*radius+(row.get('style',{}).get('width_px') or 1.))
            persistent=any(b-a>=3 and float(np.linalg.norm(np.diff(q[a:b],axis=0),axis=1).sum())+step>=span
                           for a,b in zip(starts,ends))
            source_visible=persistent or float(np.mean(visible[inside]))>=.1
            if np.mean(hidden[inside])<.9 or source_visible:
                if source_visible:after.append(name)
                continue
            # Absence near a fitted route cannot prove that an opaque face
            # hides it when the entire route lies inside that face. There is
            # no observed entry/exit boundary to establish occlusion; a small
            # localization error otherwise hides long lines under panels.
            # Leave such existence/geometry questions to source ownership.
            if all(geom.covers(p) for p in points):
                audit.append({'arrow':row['id'],'shape_owner':owner,'shape_part':key,
                    'applied':False,'placement_requested':False,
                    'reason':'contained_route_without_occlusion_boundary_evidence',
                    'source_band_radius_px':radius,'source_visible_samples':int(np.sum(inside&visible))})
                continue
            if part_placements:
                before.append(name)
                audit.append({'arrow':row['id'],'shape_owner':owner,'shape_part':key,
                              'occluded_samples':int(inside.sum()),'applied':False,
                              'placement_requested':True,'reason':'source_fill_hides_stroke_part',
                              'source_band_radius_px':radius,'source_visible_samples':int(np.sum(inside&visible)),
                              'source_visible_minimum_span_px':span,'source_visible_contiguous':persistent,
                              'text_covered_samples_excluded':int(labelled.sum()),
                              'application_check':'composition.source_occlusion_placements'})
                continue
            # A whole compound cannot be moved in front if its opaque panel
            # would also hide source-visible portions of this same route.
            blocked=False
            for other,oid,body,pigment,other_opacity in fills:
                if oid!=owner or other==key:continue
                covered=np.array([body.contains(p) for p in points])&~inside
                if np.sum(covered&visible)>=3:blocked=True;break
            audit.append({'arrow':row['id'],'shape_owner':owner,'shape_part':key,
                          'occluded_samples':int(inside.sum()),'applied':not blocked,
                          'reason':'source_fill_hides_stroke' if not blocked else 'interleaving_compound_kept_unresolved'})
            if not blocked:placements.setdefault(row['id'],[]).append(owner)
        if part_placements and before:
            placements[row['id']]={'before':sorted(set(before)),'after':sorted(set(after))}
    return placements,audit


def json_part_id(key):
    # Canonical scene keys contain provenance, owner and local part ID.
    import json
    try:return str(json.loads(key)[-1])
    except (ValueError,TypeError,IndexError):return str(key)


def enclosed_glyph_shapes(scene,styles,rows):
    """A circled Unicode glyph owns its ring at the same source occurrence."""
    import unicodedata
    dropped=[];audit=[]
    for row in rows:
        text=row.get('text','').strip()
        if len(text)!=1 or not unicodedata.name(text,'').startswith('CIRCLED '):continue
        bounds=np.asarray(row.get('bbox_2d',[]),float)
        if bounds.shape!=(4,):continue
        span=bounds[2:]-bounds[:2]
        if min(span)<=0:continue
        matches=[]
        for key,part in scene['parts'].items():
            if part.get('kind')!='ellipse' or part.get('status')!='ok':continue
            style=styles.get(key,{})
            if style.get('fill') is not None or not style.get('stroke'):continue
            b=np.asarray(part.get('bbox_2d',[]),float)
            if b.shape!=(4,):continue
            size=b[2:]-b[:2]
            if (np.any(size<.65*span) or np.any(size>1.5*span)
                    or np.any(np.abs((b[:2]+b[2:]-bounds[:2]-bounds[2:])/2)>.3*span)):continue
            matches.append(key)
        if len(matches)==1:
            dropped.extend(matches)
            audit.append({'text_id':row['id'],'shape_parts':matches,
                          'reason':'circled_glyph_owns_same_source_ring'})
    return dropped,audit


def reconcile(scene,style_data,arrow_scene,text_scene,source=None,*,quality=None,texture=None):
    styles={k:v['style'] for k,v in style_data['parts'].items()}
    quality=quality or {};texture=texture or {}
    dropped=set(quality.get('drop_shape_keys',[]))|set(texture.get('drop_shape_keys',[]))
    image_parts={k:scene['parts'][k] for asset in texture.get('assets',())
                 for k in asset.get('replace_parts',()) if k in scene['parts']}
    scene=surviving_scene(scene,{**quality,'drop_shape_keys':list(dropped)})
    # Source/native ownership and source protections are final.
    # Only uncovered legacy cases enter the fallback.
    selected_drop=set(quality.get('drop_arrow_names',[]))|set(texture.get('drop_arrow_names',[]))
    same_branch,same_branch_audit=duplicate_arrow_routes(
        [r for r in arrow_scene['arrows'] if r['id'] not in selected_drop],source)
    reviewed=set(quality.get('reviewed_arrow_names',[]))
    unreviewed=[r for r in arrow_scene['arrows'] if r['id'] not in reviewed
                and r['id'] not in selected_drop and r['id'] not in set(same_branch)]
    # Ordinary routes have one owner: quality_choice. Running the old line
    # selector afterwards silently changes the branch shown to the reviewer.
    # Keep compatibility for old callers and opaque block-contour ownership.
    source_owned = quality.get('ordinary_line_owner') in ('source_objects','local_competitor_groups')
    fallback=([r for r in unreviewed if r.get('kind')=='block_arrow']
              if source_owned or quality.get('line_ownership_complete') else unreviewed)
    arrows,arrow_audit=duplicate_arrows(scene,styles,fallback,source,represented_image_parts=image_parts)
    arrows.extend(same_branch);arrow_audit.extend(same_branch_audit)
    braces,brace_audit=split_brace_duplicates(scene,styles,
        [] if source_owned or quality.get('line_ownership_complete') else unreviewed,source,arrows)
    arrows.extend(braces);arrow_audit.extend(brace_audit)
    arrows=sorted(set(arrows)|set(quality.get('drop_arrow_names',[]))|set(texture.get('drop_arrow_names',[])))
    texts,text_audit=duplicate_operator_text(scene,styles,text_scene['texts'])
    from .text_product import exact_audit_duplicates
    repeated,repeat_audit=exact_audit_duplicates(text_scene['texts'])
    texts=sorted(set(texts)|set(repeated));text_audit.extend(repeat_audit)
    ellipsis_shapes=[];ellipsis_names=[]
    if source is not None:
        ellipsis_shapes,ellipse_audit=ellipsis_shape_ownership(scene,styles,text_scene['texts'],source,
            [r for r in arrow_scene['arrows'] if r['id'] not in set(arrows)])
        ellipsis_names=[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in ellipsis_shapes]
        dropped.update(ellipsis_shapes)
        scene=surviving_scene(scene,{'drop_shape_keys':ellipsis_shapes})
    else:
        ellipses,ellipse_audit=duplicate_ellipsis_text(scene,styles,text_scene['texts'])
        texts.extend(ellipses)
    text_audit.extend(ellipse_audit)
    glyph_shapes,glyph_audit=enclosed_glyph_shapes(scene,styles,text_scene['texts'])
    dropped.update(glyph_shapes)
    ellipsis_names.extend(scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in glyph_shapes)
    scene=surviving_scene(scene,{'drop_shape_keys':glyph_shapes})
    text_audit.extend(glyph_audit)
    brackets,bracket_audit=duplicate_bracket_text(scene,styles,text_scene['texts'],source)
    texts.extend(brackets);text_audit.extend(bracket_audit)
    parts,part_audit=source_occlusion(scene,styles,arrow_scene['arrows'],source,set(arrows),part_placements=True,texts=text_scene['texts'])
    from .node_terminal_ownership import observe as round_terminal_ownership
    caps,cap_audit=round_terminal_ownership(scene,styles,arrow_scene['arrows'],source,set(arrows))if source is not None else ({},[])
    # Compatibility view for older readers; production consumes part placements.
    placements={}
    for arrow,spec in parts.items():
        owners={name.rsplit('/',1)[0] for name in spec['before']}
        visible={name.rsplit('/',1)[0] for name in spec['after']}
        if owners-visible:placements[arrow]=sorted(owners-visible)
    return {'protocol':'native_composition_ownership.11','drop_arrow_names':arrows,
            'drop_shape_names':sorted(set(quality.get('drop_shape_names',[]))|set(texture.get('drop_shape_names',[]))|
                set(ellipsis_names)),
            'drop_shape_keys':sorted(dropped),'quality_choice':quality,'texture_fallback':texture,
            'shape_segment_suppression':quality.get('shape_segment_suppression',{}),
            'shape_subpath_suppression':quality.get('shape_subpath_suppression',{}),
            'fallback_assets':[],  # Assets have one commit before arrow/text merge.
            'drop_text_ids':texts,'arrow_audit':arrow_audit,'text_audit':text_audit,
            'arrows_before_shape_owners':placements,'source_occlusion_audit':part_audit,
            'arrows_between_shape_parts':parts,'source_part_occlusion_audit':part_audit,
            'arrow_round_terminal_suppression':caps,'source_node_terminal_audit':cap_audit,
            'model_calls':quality.get('model_calls',0)+texture.get('model_calls',0),'shape_image_dedup_unchanged':False,'image_ownership_protocol':texture.get('protocol')}
