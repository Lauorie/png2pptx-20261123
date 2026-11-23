"""Editable mask-derived geometry proposals. No GLM coordinates on the normal path."""
import copy
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import approximate_polygon
from PIL import Image,ImageDraw,ImageFont
import inkgraph as ig
import arrows
import straight_mask
import visual
from transport.common import write_json as write


def compress(g):
    """Join degree-two chains, including proposed gaps; keep branch nodes explicit."""
    changed=True
    while changed:
        changed=False
        for n in list(g.nodes):
            if n not in g.nodes or g.degree(n)!=2 or len(g.nodes[n]['edges'])!=2:continue
            e1,e2=list(g.nodes[n]['edges']);a,b=g.other(e1,n),g.other(e2,n)
            if a==n or b==n or a==b:continue
            path=np.vstack([g.oriented(e1,a),g.oriented(e2,n)[1:]])
            gaps=[]
            for e in (e1,e2):
                d=g.edges[e];gaps+=d.get('gaps',[])
                if d.get('gap'):gaps.append(d['path'].tolist())
            g.remove_edge(e1);g.remove_edge(e2);del g.nodes[n];g.add_edge(a,b,path,gaps=gaps);changed=True


def safe_row(path,ident,dt):
    row=arrows.path_row(path,['none','none'],ident,dt);p=np.asarray([row['start'],*row['via'],row['end']])
    margin=max(3.,row['width_estimate']*2)
    if not np.isfinite(p).all() or (p<np.min(path,axis=0)-margin).any() or (p>np.max(path,axis=0)+margin).any():
        p=approximate_polygon(path,tolerance=.75)
        row.update(kind='polyline',start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),beziers=[],
            numerical_fallback='Fit overshoot; retained observed pixel chain')
    # All incident rows use the exact same observed junction coordinate.
    if row.get('beziers'):
        curves=np.asarray(row['beziers']);da=path[0]-curves[0,0];db=path[-1]-curves[-1,3]
        curves[0,:2]+=da;curves[-1,2:]+=db;row['beziers']=curves.tolist()
    row['start']=path[0].tolist();row['end']=path[-1].tolist();return row


def geometry_support(mask):
    """Fill enclosed pinholes smaller than surrounding ink thickness.

    This is a derived measurement image only. Native masks remain unchanged for
    style and audit. A thin ring's large interior is deliberately not filled.
    """
    filled=ndimage.binary_fill_holes(mask);holes=filled&~mask
    labels,count=ndimage.label(holes);dt=ndimage.distance_transform_edt(mask)
    result=mask.copy();changes=[]
    for i,box in enumerate(ndimage.find_objects(labels),1):
        if box is None:continue
        y,x=box;h=labels[box]==i
        radius=float(ndimage.distance_transform_edt(np.pad(h,1)).max())
        pad=max(2,int(np.ceil(radius*2)))
        window=(slice(max(0,y.start-pad),min(mask.shape[0],y.stop+pad)),
                slice(max(0,x.start-pad),min(mask.shape[1],x.stop+pad)))
        ink_radius=float(dt[window].max())
        if radius<=ink_radius and h.sum()<=np.pi*ink_radius**2:
            result[box]|=h;changes.append(dict(area=int(h.sum()),radius=radius,surrounding_ink_radius=ink_radius))
    return result,changes


def draft(candidates,size,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);mask=np.zeros(size[::-1],bool)
    for r in candidates:mask|=r['mask']
    Image.fromarray(mask.astype('uint8')*255).save(out/'support.png')
    direct=straight_mask.estimate(mask) if len(candidates)==1 else None
    if direct is not None:
        Image.fromarray(mask.astype('uint8')*255).save(out/'geometry_support.png')
        direct.update(nodes=['N1','N2'],gaps=[])
        data=dict(nodes={n:dict(xy=direct[e],degree=1) for n,e in [('N1','start'),('N2','end')]},
            paths=[direct],diagnostics=dict(method='direct_straight_mask',proposal_gap_links=0,
                mask_pixels=int(mask.sum()),geometry_only_filled_holes=[],source_masks=[r['id'] for r in candidates],
                semantic_connectivity_confirmed=False))
        write(out/'draft.json',data);return data,mask
    measured,holes=geometry_support(mask)
    Image.fromarray(measured.astype('uint8')*255).save(out/'geometry_support.png')
    dt=ndimage.distance_transform_edt(measured);g=ig.build(skeletonize(measured))
    before=dict(nodes=len(g.nodes),edges=len(g.edges));ig.prune(g,dt);ig.analyze_ends(g,measured,dt)
    def instance_of(n):
        p=g.nodes[n].get('tip',g.nodes[n]['xy']);x,y=np.rint(p).astype(int)
        return {r['id'] for r in candidates if r['mask'][max(0,y-2):y+3,max(0,x-2):x+3].any()}
    links=ig.link_gaps(g,dt,instance_of,max_collinear=0,max_same=500)
    compress(g);nodes={};node_map={}
    for i,(n,d) in enumerate(g.nodes.items(),1):
        if not d['edges']:continue
        label=f'N{i}';node_map[n]=label
        nodes[label]=dict(xy=(d.get('tip',d['xy']) if g.degree(n)==1 else d['xy']).tolist(),degree=g.degree(n))
    paths=[]
    for i,(e,d) in enumerate(g.edges.items(),1):
        a,b=node_map[d['a']],node_map[d['b']];path=d['path'].copy()
        path[0]=nodes[a]['xy'];path[-1]=nodes[b]['xy']
        if len(path)<2 or np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<1:continue
        try:row=safe_row(path,f'E{i}',dt)
        except (ValueError,np.linalg.LinAlgError):continue
        row.update(nodes=[a,b],gaps=d.get('gaps',[])+([d['path'].tolist()] if d.get('gap') else []),coordinate_source='local_sam_mask')
        paths.append(row)
    data=dict(nodes=nodes,paths=paths,diagnostics=dict(raw_graph=before,proposal_gap_links=len(links),
        mask_pixels=int(mask.sum()),geometry_only_filled_holes=holes,source_masks=[r['id'] for r in candidates],semantic_connectivity_confirmed=False))
    write(out/'draft.json',data);return data,mask


def compact(data):
    used={n for p in data['paths'] for n in p['nodes']}
    return dict(nodes={n:[round(v,1) for v in d['xy']] for n,d in data['nodes'].items() if n in used},
        paths=[dict(id=r['id'],nodes=r['nodes'],kind=r['kind'],gaps=len(r['gaps'])) for r in data['paths']])


def draw(clean,data,path):
    im=clean.copy();d=ImageDraw.Draw(im);f=ImageFont.truetype(visual.FONT,17)
    occupied=[]
    def text_at(q,text,c):
        w=d.textbbox((0,0),text,font=f)[2]+4;h=22
        positions=[(q[0]+dx,q[1]+dy) for dy in (-24,4,-48,28,-72,52) for dx in (5,-w-5,30,-w-30)]
        for x,y in positions:
            x=max(1,min(im.width-w-1,x));y=max(1,min(im.height-h-1,y));box=(x,y,x+w,y+h)
            if not any(x<b[2] and x+w>b[0] and y<b[3] and y+h>b[1] for b in occupied):break
        occupied.append(box)
        d.line([tuple(q),(x+w/2,y+h/2)],fill=c,width=1)
        d.text((x,y),text,font=f,fill=c,stroke_width=2,stroke_fill='white')
    for i,r in enumerate(data['paths']):
        points=[r['start'],*r['via'],r['end']];c=visual.color(i)
        if r.get('beziers'):
            t=np.linspace(0,1,25)[:,None];points=[]
            for c0,c1,c2,c3 in np.asarray(r['beziers']):
                points.extend(((1-t)**3*c0+3*(1-t)**2*t*c1+3*(1-t)*t**2*c2+t**3*c3).tolist())
        d.line([tuple(p) for p in points],fill=c,width=2)
        p=np.asarray(r.get('route_path',points));q=p[len(p)//2]
        text_at(q,r['id'],c)
    used={n for p in data['paths'] for n in p['nodes']}
    for n in sorted(used,key=lambda n:int(n[1:])):
        x,y=data['nodes'][n]['xy'];d.ellipse([x-3,y-3,x+3,y+3],fill='#333333')
        text_at((x,y),n,'#222222')
    im.save(path)


def materialize(data,decision):
    """Emit each approved pixel path once; headings bind to visible node IDs."""
    index={r['id']:r for r in data['paths']};result=[];emitted={}
    for i,target in enumerate(decision['targets'],1):
        if target.get('repair'):continue
        for key in target['paths']:
            if key in emitted:
                emitted[key].setdefault('also_owned_by',[]).append(i);continue
            row=copy.deepcopy(index[key]);heads=[decision['heads'].get(n,'unknown') for n in row['nodes']]
            row.update(heads=heads,identity_index=i,globals=target['sources'],shared_nodes=[n for n in row['nodes'] if n in decision['junctions']],
                node_ids=[n if n in decision['junctions'] else f'T{i}:{n}' for n in row['nodes']])
            row['line_style']=decision['line_styles'][key]
            for end,h in zip(['start','end'],heads):row.update({end+'_head':h not in ('none','unknown'),end+'_head_type':h if h not in ('none','unknown') else None})
            result.append(row);emitted[key]=row
    return result
