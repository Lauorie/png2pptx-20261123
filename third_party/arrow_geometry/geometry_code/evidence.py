"""Global hints create discovery tasks. Native local observations never create targets."""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
import graph as gr
import visual
from transport.common import read_json as read,write_json as write,sha256 as sha


def mask(path):
    with Image.open(path) as im:return np.asarray(im.convert('L'))>0


def compact(rows):return [{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]


def support(m,origin=(0,0)):
    y,x=np.nonzero(m)
    if not len(x):return np.zeros((0,0),bool),[origin[0],origin[1],origin[0],origin[1]]
    x0,y0,x1,y1=int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
    return m[y0:y1,x0:x1],[x0+origin[0],y0+origin[1],x1+origin[0],y1+origin[1]]


def intersect(a,b):
    aa,bb=a['box'],b['box'];x0=max(aa[0],bb[0]);y0=max(aa[1],bb[1]);x1=min(aa[2],bb[2]);y1=min(aa[3],bb[3])
    if x0>=x1 or y0>=y1:return 0
    return int((a['_support'][y0-aa[1]:y1-aa[1],x0-aa[0]:x1-aa[0]]&b['_support'][y0-bb[1]:y1-bb[1],x0-bb[0]:x1-bb[0]]).sum())


def gap(a,b):return float(np.hypot(max(0,a[0]-b[2],b[0]-a[2]),max(0,a[1]-b[3],b[1]-a[3])))


def bounds(boxes):
    p=np.asarray(boxes);return [p[:,0].min(),p[:,1].min(),p[:,2].max(),p[:,3].max()]


def identity_view(size):
    return dict(bbox_global=[0,0,*size],size=list(size),global_size=list(size),scale=[1.,1.],global_to_view=np.eye(3).tolist(),view_to_global=np.eye(3).tolist())


def load_case(case,config,out):
    cv2.setNumThreads(1);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if config.get('input_mode')=='global_sam_only':
        from global_input import load_case as global_case
        return global_case(case,config,out)
    source=Path(config['local_runs'][str(float(config['crop_scale']))])
    base=source/'cases'/case;anchor_file=base/'input_targets.json';anchors=read(anchor_file)
    clean_path=Path(config['clean_run'])/'cases'/case/'clean.png';clean=Image.open(clean_path).convert('RGB');clean.save(out/'clean.png')
    hashes={str(anchor_file):sha(anchor_file),str(clean_path):sha(clean_path)};globals=[];local=[]
    for gid in anchors:
        p=Path(config['global_run'])/'cases'/case/'filtered/global'/f'{gid}.png'
        m=mask(p)
        if m.shape!=tuple(clean.size[::-1]):raise ValueError('Global hint frame mismatch')
        s,b=support(m);label='G'+str(int(gid[1:]))
        globals.append(dict(id=label,source_id=gid,path=str(p),sha256=sha(p),view=identity_view(clean.size),
            box=b,area=int(s.sum()),_support=s));hashes[str(p)]=sha(p)
        folder=base/'targets'/gid;index=folder/'candidates.json';hashes[str(index)]=sha(index)
        for row in read(index):
            p=Path(row['filtered_path'])
            if sha(p)!=row['filtered_sha256']:raise ValueError('Changed native evidence')
            m=mask(p);view=row['view']
            if m.shape!=tuple(view['size'][::-1]):raise ValueError('Native mask frame mismatch')
            x0,y0,x1,y1=map(int,view['bbox_global'])
            projected=cv2.resize(m.astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>0
            s,b=support(projected,(x0,y0))
            local.append(dict(id=f'L{len(local)+1}',origin_hint=label,source_id=row['id'],path=str(p),sha256=row['filtered_sha256'],
                raw_path=row['raw_path'],raw_sha256=row['raw_sha256'],view=view,filter=row['filter'],
                box=b,area=int(s.sum()),_support=s))
    write(out/'global_hints.json',compact(globals));write(out/'local_observations.json',compact(local))
    write(out/'source_manifest.json',dict(case=case,global_hints=len(globals),local_observations=len(local),source_run=str(source),
        source_sha256=hashes,global_filter='inherited reversible mask_refine filtering',local_filter='inherited instance_sam filtering; records retained',
        projection_for='retrieval and preview ONLY; selected evidence retains native paths and transforms',
        local_observations_create_targets=False,new_sam_calls=0))
    return clean,globals,local


def batches(hints,limit):
    todo=list(hints);result=[]
    while todo:
        group=[todo.pop(0)]
        while todo and len(group)<limit:
            candidates=[]
            for i,r in enumerate(todo):
                overlap=max(intersect(r,g)/max(1,min(r['area'],g['area'])) for g in group)
                candidates.append((-overlap,min(gap(r['box'],g['box']) for g in group),i))
            _,distance,index=min(candidates)
            if distance>64:break
            group.append(todo.pop(index))
        result.append(group)
    return result


def frame(clean,boxes,folder,config,box=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    x0,y0,x1,y1=map(float,box if box is not None else bounds(boxes));pad=config['context_padding']
    cx,cy=(x0+x1)/2,(y0+y1)/2;w=x1-x0+2*pad;h=y1-y0+2*pad
    minimum=max(config['min_crop_side'],.18*max(w,h));w=max(w,minimum);h=max(h,minimum)
    x0=max(0,int(np.floor(cx-w/2)));y0=max(0,int(np.floor(cy-h/2)))
    x1=min(clean.width,int(np.ceil(cx+w/2)));y1=min(clean.height,int(np.ceil(cy+h/2)))
    if x1<=x0 or y1<=y0:raise ValueError('Empty crop')
    from display_resolution import long_edge
    side=long_edge((x1-x0,y1-y0),config.get('display_long_side_range'),config['display_side'])
    scale=side/max(x1-x0,y1-y0);size=[max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))]
    sx,sy=size[0]/(x1-x0),size[1]/(y1-y0)
    matrix=np.array([[sx,0,(-x0+.5)*sx-.5],[0,sy,(-y0+.5)*sy-.5],[0,0,1]])
    view=dict(bbox_global=[x0,y0,x1,y1],size=size,global_size=list(clean.size),scale=[sx,sy],global_to_view=matrix.tolist(),view_to_global=np.linalg.inv(matrix).tolist())
    crop=clean.crop((x0,y0,x1,y1)).resize(size,Image.Resampling.LANCZOS);crop.save(folder/'clean.png');write(folder/'transform.json',view)
    return crop,view


def project(row,view):
    old=row['view'];matrix=np.asarray(old['global_to_view'])@np.asarray(view['view_to_global'])
    # Pillow maps pixel-edge coordinates; convert our pixel-center affine accordingly.
    linear=matrix[:2,:2];offset=matrix[:2,2]+.5-linear@np.array([.5,.5])
    affine=(linear[0,0],linear[0,1],offset[0],linear[1,0],linear[1,1],offset[1])
    with Image.open(row['path']) as im:
        return np.asarray(im.convert('L').transform(tuple(view['size']),Image.Transform.AFFINE,affine,resample=Image.Resampling.NEAREST))>0


def crop_in_context(view,context_size):
    # This is a coordinate transform, not a semantic or geometry prediction.
    b=np.asarray(view['bbox_global'],float)*np.tile(np.asarray(context_size)/view['global_size'],2)
    return [round(float(x),1) for x in b]


def overview(clean,folder,config,hints=(),targets=()):
    image=clean.copy();image.thumbnail((config['context_side'],config['context_side']),Image.Resampling.LANCZOS)
    v=identity_view(clean.size);sx=image.width/clean.width;sy=image.height/clean.height
    matrix=np.array([[sx,0,.5*sx-.5],[0,sy,.5*sy-.5],[0,0,1]])
    v.update(size=list(image.size),scale=[sx,sy],global_to_view=matrix.tolist(),view_to_global=np.linalg.inv(matrix).tolist())
    path=Path(folder)/'context.png'
    if hints:visual.masks(image,[(r['id'],project(r,v)) for r in hints],path)
    elif targets:visual.targets(image,[{**t,'graph':gr.local(t['graph'],v)} for t in targets],path)
    else:image.save(path)
    return image.size


def discovery_images(clean,hints,folder,config,box=None):
    crop,view=frame(clean,[r['box'] for r in hints],folder,config,box)
    context_size=overview(clean,folder,config,hints=hints)
    visual.masks(crop,[(r['id'],project(r,view)) for r in hints],Path(folder)/'masks.png')
    write(Path(folder)/'hints.json',compact(hints))
    return view,context_size,[('CONTEXT: global hint highlights are location annotations',Path(folder)/'context.png'),
        ('CLEAN: unpainted original crop',Path(folder)/'clean.png'),('MASKS: global hints over CLEAN, not final targets',Path(folder)/'masks.png')]


def raster(graph,size,width=3):
    im=Image.new('L',size);d=ImageDraw.Draw(im)
    for r in graph['routes']:
        p=list(map(tuple,gr.points(graph,r)))
        if r['kind']=='block_arrow':d.polygon(p,fill=255)
        else:d.line(p,fill=255,width=width)
    return np.asarray(im)>0


def graph_support(graph,size):
    m,b=support(raster(graph,size));return dict(box=b,_support=m,area=int(m.sum()))


def samples(graph):
    points=[]
    for r in graph['routes']:
        p=gr.points(graph,r)
        for a,b in zip(p,p[1:]):points.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/3)+1)))
    return np.asarray(points)


def rank_evidence(target,rows,size):
    """Rank all observations, including other crops. Never reject a model selection."""
    route=raster(target['graph'],size,1);distance=distance_transform_edt(~route)
    radius=max(4.,max(size)*.006);q=np.rint(samples(target['graph'])).astype(int)
    q[:,0]=q[:,0].clip(0,size[0]-1);q[:,1]=q[:,1].clip(0,size[1]-1)
    ranked=[]
    for row in rows:
        x0,y0,x1,y1=row['box'];m=row['_support']
        purity=float((distance[y0:y1,x0:x1][m]<=radius).mean()) if row['area'] else 0.
        within=(q[:,0]>=x0)&(q[:,0]<x1)&(q[:,1]>=y0)&(q[:,1]<y1)
        hits=np.zeros(len(q),bool)
        if within.any() and row['area']:
            local_distance=distance_transform_edt(~np.pad(m,1,constant_values=False))
            hits[within]=local_distance[q[within,1]-y0+1,q[within,0]-x0+1]<=radius
        coverage=float(hits.mean());score=2*purity*coverage/max(1e-9,purity+coverage)
        ranked.append(dict(id=row['id'],score=score,purity=purity,coverage=coverage,
            distance=gap(gr.box(target['graph']),row['box'])))
    return sorted(ranked,key=lambda r:(-r['score'],r['distance'],r['id']))


def evidence_page(ranked,index,seen,limit):
    """Prefer diverse observations for display; skipped near-copies remain eligible later."""
    remaining=[r for r in ranked if r['id'] not in seen];page=[]
    while remaining and len(page)<limit:
        def priority(r):
            row=index[r['id']]
            overlap=max((intersect(row,index[x['id']])/max(1,min(row['area'],index[x['id']]['area'])) for x in page),default=0)
            return r['score']*(1-.8*overlap),-r['distance']
        chosen=max(remaining,key=priority);page.append(chosen);remaining.remove(chosen)
    return [index[r['id']] for r in page]
