"""GLM-authored line/curve parts. Shared by the pilot and main pipeline.

Validation checks references and coordinates only; masks never alter these points.
"""
import copy,math
import protocol

def drawable_paths(route):
    """A target owns independent drawable paths; never bridge a reference break.

    The preferred wire form names paths explicitly. Historical flat segments
    are read unchanged, starting a new path at each discontinuity. This is a
    representation conversion: no point, primitive or direction is changed.
    """
    if 'paths' in route:
        paths=route['paths']
        if not isinstance(paths,list) or not paths:raise ValueError('A target needs one or more paths')
        if route.get('segments'):raise ValueError('Use paths or legacy segments, not both')
        seen=set()
        for path in paths:
            if not isinstance(path,dict) or not isinstance(path.get('id'),str) or not path['id'] or '/' in path['id'] or path['id'] in seen:
                raise ValueError('Path IDs must be unique nonempty strings without /')
            seen.add(path['id'])
            if not isinstance(path.get('segments'),list) or not path['segments']:raise ValueError('Every path needs segments')
        return paths
    segments=route.get('segments')
    if not isinstance(segments,list) or not segments:raise ValueError('Every target needs paths or legacy segments')
    paths=[];previous=None
    for segment in segments:
        if not isinstance(segment,dict) or not isinstance(segment.get('nodes'),list) or not segment['nodes']:
            raise ValueError('Segment needs nonempty node references')
        if not paths or previous!=segment['nodes'][0]:paths.append(dict(id=f'P{len(paths)+1}',segments=[]))
        paths[-1]['segments'].append(segment);previous=segment['nodes'][-1]
    return paths


def validate(answer,targets,size,reference_points=(),require_heads=False):
    if not isinstance(answer,dict) or not isinstance(answer.get('nodes'),list) or not isinstance(answer.get('routes'),list):
        raise ValueError('Return nodes and routes arrays')
    answer=copy.deepcopy(answer)
    if (len(answer['routes'])==1 and answer['nodes'] and
            all(isinstance(n,list) and len(n)==2 for n in answer['nodes'])):
        # An equivalent single-route wire form uses literal coordinate pairs.
        # Only exact references to its explicit point table are normalized.
        # Never infer cross-route junctions from equal coordinates.
        lookup={tuple(xy):'p'+str(i+1) for i,xy in enumerate(answer['nodes'])}
        if len(lookup)!=len(answer['nodes']):raise ValueError('Ambiguous duplicated inline point table')
        route=answer['routes'][0]
        segments=route.get('segments',[]) if 'paths' not in route else [s for p in route['paths'] for s in p.get('segments',[])]
        for segment in segments:
            refs=segment.get('nodes',[])
            if not all(isinstance(xy,list) and len(xy)==2 and tuple(xy) in lookup for xy in refs):
                raise ValueError('Inline segment references must match the supplied coordinate table')
            segment['nodes']=[lookup[tuple(xy)] for xy in refs]
        answer['nodes']=[[lookup[tuple(xy)],*xy] for xy in answer['nodes']]
    table={};known={tuple(p) for p in reference_points}
    for node in answer['nodes']:
        if not isinstance(node,list) or len(node)!=3:raise ValueError('Each node is [id,x,y]')
        name,x,y=node
        if type(name) not in (str,int) or name in table:raise ValueError('Node IDs must be unique strings or integers')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in [x,y]):raise ValueError('Finite numerical coordinates required')
        if (not 0<=x<size[0] or not 0<=y<size[1]) and (x,y) not in known:
            raise ValueError('New coordinates must be inside CLEAN; only exact supplied existing points may be reused outside. Request expanded context for new geometry outside CLEAN.')
        table[name]=[x,y]
    seen=set()
    for route in answer['routes']:
        rid=route.get('id')
        if rid not in targets or rid in seen:raise ValueError('Each supplied route ID must appear once')
        if 'heads' in route:
            heads=route.get('heads')
            if not isinstance(heads,list) or len(heads)!=2 or any(h not in protocol.HEADS for h in heads):
                raise ValueError('Each route needs two head types attached to its own first/last endpoint')
        seen.add(rid);paths=drawable_paths(route)
        for path in paths:
            heads=path.get('heads',route.get('heads') if len(paths)==1 else None)
            if require_heads or 'heads' in path:
                if not isinstance(heads,list) or len(heads)!=2 or any(h not in protocol.HEADS for h in heads):
                    raise ValueError('Each drawable path needs two ordered endpoint heads')
            previous=None
            for segment in path['segments']:
                kind=segment.get('kind');ids=segment.get('nodes')
                if kind not in ('line','curve') or not isinstance(ids,list):raise ValueError('Segment needs kind=line|curve and nodes')
                if (kind=='line' and len(ids)!=2) or (kind=='curve' and len(ids)<3):raise ValueError('line needs 2 nodes; curve needs >=3 on-curve nodes')
                if any(x not in table for x in ids):raise ValueError('Undefined point reference')
                if previous is not None and previous!=ids[0] and table[previous]!=table[ids[0]]:
                    raise ValueError('Parts within a path must meet; use separate paths for branches')
                if any(table[a]==table[b] for a,b in zip(ids,ids[1:])):raise ValueError('Consecutive points must differ')
                previous=ids[-1]
    if seen!=set(targets):raise ValueError('Return all supplied route IDs, no additions or omissions')
    return answer


def geometry(answer,baseline):
    """Compile independent paths; preserve the existing interpolation per part."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    table={n:np.array([x,y],float) for n,x,y in answer['nodes']};result=[]
    node_use={};flat=[]
    for route in answer['routes']:
        paths=drawable_paths(route)
        for path in paths:
            flat.append((route,path,len(paths)))
            for n in {n for s in path['segments'] for n in s['nodes']}:node_use[n]=node_use.get(n,0)+1
    row_ids=set()
    for route,path,npaths in flat:
        row=copy.deepcopy(baseline[route['id']]);beziers=[];samples=[];references=[]
        owner=row.get('id',str(route['id']))
        row.update(geometry_target_id=route['id'],geometry_path_id=path['id'],geometry_owner_id=owner,physical_target_id=owner)
        if npaths>1:row['id']=owner+'/'+path['id']
        elif 'id' not in row:row['id']=owner
        if row['id'] in row_ids:raise ValueError('Compiled path ID collision')
        row_ids.add(row['id'])
        heads=path.get('heads',route.get('heads') if npaths==1 else None)
        if heads is not None:
            row['heads']=list(heads);row['head_source']='glm_with_final_geometry'
        elif npaths>1:
            # A head on the identity-level route cannot be copied to each arm.
            # The existing endpoint model will classify these final path ends.
            row['heads']=['unknown','unknown'];row['head_source']='deferred_per_path'
            if 'style' in row:row['style'].update(start=None,end=None)
        for segment in path['segments']:
            p=np.array([table[n] for n in segment['nodes']]);references+=segment['nodes']
            if segment['kind']=='line':
                a,b=p;beziers.append([a,a+(b-a)/3,a+2*(b-a)/3,b]);sample=p
            else:
                t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
                spline=CubicSpline(t,p,axis=0,bc_type='natural')
                for i,h in enumerate(np.diff(t)):
                    beziers.append([p[i],p[i]+spline(t[i],1)*h/3,p[i+1]-spline(t[i+1],1)*h/3,p[i+1]])
                sample=spline(np.unique(np.r_[t,np.linspace(0,t[-1],128)]))
                sample[0]=p[0];sample[-1]=p[-1]
            samples.extend(sample if not samples else sample[1:])
        p=np.array(samples);curved=any(s['kind']=='curve' for s in path['segments'])
        row.update(kind='curve' if curved else ('straight' if len(path['segments'])==1 else 'polyline'),
                   start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),
                   beziers=np.asarray(beziers).tolist() if curved else [],
                   coordinate_source='glm_segmented_points',geometry_method='glm_parts_shape_natural_cubic',
                   model_segments=copy.deepcopy(path['segments']),
                   model_nodes=copy.deepcopy(answer['nodes']),
                   shared_nodes=[n for n in set(references) if node_use[n]>1])
        row.pop('points',None);result.append(row)
    return result


def prompt(targets,size,locator_mode='masks'):
    if locator_mode not in ('masks','boxes'):raise ValueError('Unknown geometry locator mode')
    locator=('CLEAN is the reference. CONTEXT shows the whole figure. TARGET BOXES locate ONLY the already confirmed R targets. '
             'Each bbox is approximate and may be partial: follow that target to its visible ends, but do not recover unrelated arrows elsewhere in CLEAN. '
             'A box does not establish connectivity or prescribe endpoints. No mask matching is required.'
             if locator_mode=='boxes' else
             'CLEAN is the reference. CONTEXT shows the whole figure. MASKS marks the target evidence with R IDs; masks may be incomplete or include a neighbor. Do not treat mask boundaries as the centerline.')
    return dict(task='Fit only the centerline geometry of these confirmed arrows/lines from CLEAN.',
        size=list(size),targets=targets,
        instructions=[protocol.OCCLUSION,
            locator,
            'You choose the complete geometry, including endpoints and turns, from CLEAN. description only locates the intended ink; it does not prescribe coordinates or route order. Return one target record per supplied R ID. Each target may contain multiple independently drawable paths; the combined paths should reproduce its visible geometry. Do not discover unrelated objects or estimate width, color or dash dimensions.',
            'Preserve the turns actually visible in CLEAN. A sharp corner is two line segments sharing the exact corner node; do not round or smooth across it. Use curve only for a visibly curved portion; a real rounded elbow may combine line/curve/line and a broad arc may be one curve. Antialiasing, stroke thickness and assumptions about typical drawing styles are not evidence of a curved centerline. If a corner is visually sharp or roundness is unsupported, keep the sharp line junction.',
            'A line uses two point IDs. A curve uses at least three ordered points ON its centerline, not Bezier controls. Parts within one path are consecutive; different paths need not meet or form a single traversal. Use separate paths for branches. No artificial connecting segment or backtracking is needed to satisfy the format. Align tangents only where the source is visibly smooth.',
            'A visibly shared location can use the same point ID; network naming and trunk/branch ownership proofs are not required. Mere crossings do not imply a junction. Coordinates are local CLEAN pixels, one decimal. Include true endpoint/tip positions. Head types are classified separately on the final path endpoints.',
            'Return compact JSON only.'],
        output_schema=dict(nodes='[[point_id,x,y],...]',routes='[{id:R_ID,paths:[{id:path_ID,segments:[{kind:line|curve,nodes:[point_ids]}]}]}]'))


def draw_points(answer,clean,path):
    from PIL import Image,ImageDraw
    import visual
    image=Image.open(clean).convert('RGB');d=ImageDraw.Draw(image)
    for name,x,y in answer['nodes']:
        d.ellipse((x-3,y-3,x+3,y+3),fill='#ff0080',outline='white')
        visual.label(d,(x,y),str(name),'#ff0080',image)
    image.save(path)

