"""Local, model-assisted reconciliation of overlapping representations."""
import argparse,copy,fcntl,json,threading,time,traceback,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'lib')]
import evidence as ev
import segmented_geometry as sg
import geometry_critic as gc
import geometry_heads as heads
import native
from lib import fit
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model

SYSTEM='Use the unmarked source image as evidence. Reconcile only the supplied local linework. Return the requested JSON.'
PALETTE=['#0055d4','#d40000','#008a00','#8a00b0','#a05a00','#008a9a']


def global_document(rows):
    """Stored model_nodes are in each request's CLEAN frame, not the final scene frame."""
    nodes=[];paths=[];shared={};mapping={}
    for i,row in enumerate(rows,1):
        if not row.get('model_nodes') or not row.get('model_segments'):
            row=copy.deepcopy(row);document=gc.as_points([row],with_heads=True)
            row['model_nodes']=document['nodes'];row['model_segments']=document['routes'][0]['paths'][0]['segments']
            row['geometry_frame']={'view_to_global':np.eye(3).tolist()}
        m=np.asarray(row['geometry_frame']['view_to_global'],float)
        table={n[0]:(m@np.r_[n[1:],1])[:2] for n in row['model_nodes']}
        segments=[];scope=(row.get('geometry_owner_id',row['id']),row.get('geometry_refinement',row.get('geometry_request','')))
        for part in row['model_segments']:
            refs=[]
            for ref in part['nodes']:
                xy=table[ref];key=(*scope,str(ref),*np.round(xy,6))
                if key not in shared:
                    shared[key]='n'+str(len(nodes)+1);nodes.append([shared[key],*xy.tolist()])
                refs.append(shared[key])
            segments.append(dict(kind=part['kind'],nodes=refs))
        first=table[row['model_segments'][0]['nodes'][0]];last=table[row['model_segments'][-1]['nodes'][-1]]
        if not np.allclose(first,row['start'],atol=1e-4) or not np.allclose(last,row['end'],atol=1e-4):raise ValueError('Stored point frame does not match retained geometry: '+row['id'])
        name='P'+str(i);mapping[name]=row['id'];paths.append(dict(id=name,heads=list(row['heads']),segments=segments))
    return dict(nodes=nodes,routes=[dict(id='R1',paths=paths)]),mapping


def transform_document(document,matrix):
    result=copy.deepcopy(document);m=np.asarray(matrix)
    for node in result['nodes']:node[1:]=(m@np.r_[node[1:],1])[:2].tolist()
    return result


def prompt(current,size):
    return dict(task='Reconcile these local paths into a faithful, non-redundant drawing of the same visible arrows/lines.',
        size=list(size),current=current,instructions=[
        'CLEAN is the source. CURRENT shows the supplied editable paths in annotation colors with P labels; gray paths are surrounding read-only context. Colors and labels in CURRENT are not source ink. Coordinates refer only to CLEAN.',
        'Check all supplied paths together. Different requests may describe the same stroke using different endpoints or segmentation. Remove repeated drawing of a full path or only part of it. You may split, combine or reorganize paths; keep every genuine branch, visible terminal, intermediate arrowhead and connection. A shared trunk needs drawing once, with all real branches still connected.',
        'Semantic correctness comes first: actual tails/tips, direction and which visible lines connect. Small pixel offsets or modest curve-shape deviations are acceptable. Do not refine only to chase pixel precision. Do not infer hidden links; crossings and nearby parallel lines are not automatically one connection.',
        'Path count is not arrow count. A shaft with an interior arrowhead and a visible unheaded continuation may need separate paths. Head wings are endpoint style, not extra shafts. Do not delete a unique part merely because another part overlaps.',
        'Only reorganize the supplied paths. Do not copy gray context paths, discover unrelated arrows, trace closed container borders, or estimate styles. Some inputs are already correct: return accept if no reconciliation is needed. Return uncertain if the evidence cannot support a safe change.',
        'For refine return the COMPLETE replacement of all supplied editable paths, using route R1. Keep only essential line vertices; for a simple curve prefer 3-5 ordered points ON the curve, with additional points only when needed. line uses exactly two point IDs. Parts of one path meet; multiple paths may remain disconnected.',
        'Each path has heads=[start_type,end_type]: none, triangle (generic directional head), oval (round terminal), or unknown. Bind heads to the ordered endpoints. Preserve real interior heads and both ends of double-headed arrows. Sizes are med/med; ignore width, color and dashes.'
        ],output_schema=dict(decision='accept|refine|uncertain',reason='one short sentence for refine/uncertain; omit for accept',
            geometry='only for refine: {nodes:[[id,x,y],...],routes:[{id:R1,paths:[{id,heads:[start_type,end_type],segments:[{kind:line|curve,nodes:[point_ids]}]}]}]}'))


def validate(value,size):
    return gc.validate(value,size,require_heads=True)


def bounds(rows):
    xy=np.asarray([p for r in rows for p in [r['start'],*r.get('via',[]),r['end']]])
    return np.r_[xy.min(axis=0),xy.max(axis=0)]




def native_views(crop,rows,folder,width,color='#666666'):
    folder.mkdir(parents=True,exist_ok=True)
    heads.render(crop,rows,folder,width_px=width,color=color)
    write(folder/'scene.json',dict(size=list(crop.size),arrows=rows,geometry_only=True))


def labeled_current(crop,rows,context,folder,width):
    allrows=copy.deepcopy(context+rows);styles=[]
    for i,row in enumerate(allrows):
        color='#aaaaaa' if i<len(context) else PALETTE[(i-len(context))%len(PALETTE)]
        styles.append(dict(color=color,width_px=width,dash=dict(type='solid',preset='solid'),start=heads.head_style(row['heads'][0]),end=heads.head_style(row['heads'][1]),state='geometry_preview'))
    native.write(crop.size,allrows,styles,folder/'current.pptx');native.render(folder/'current.pptx',folder/'native_current',crop.size)
    rgb=np.asarray(Image.open(folder/'native_current/styled.png').convert('RGB'),float)
    # Annotation palette has a zero channel; unmatte the white page to keep source visible.
    alpha=np.clip(1-rgb.min(axis=2)/255,0,1);foreground=np.clip((rgb-255*(1-alpha[:,:,None]))/np.maximum(alpha[:,:,None],1e-8),0,255)
    rgba=np.dstack([foreground,alpha*255]).round().astype('uint8');overlay=crop.convert('RGBA');overlay.alpha_composite(Image.fromarray(rgba,'RGBA'));overlay=overlay.convert('RGB')
    draw=ImageDraw.Draw(overlay);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for i,row in enumerate(rows):
        points=[row['start'],*row.get('via',[]),row['end']];p=points[len(points)//2];x=min(crop.width-40,max(2,p[0]+7));y=min(crop.height-20,max(2,p[1]+8+(i%3)*14))
        draw.text((x,y),'P'+str(i+1),font=font,fill=PALETTE[i%len(PALETTE)],stroke_width=2,stroke_fill='white')
    overlay.save(folder/'current.png')


def apply_patch(rows,edit_ids,replacements):
    ids=set(edit_ids)
    if len(ids)!=len(edit_ids) or not ids<=set(r['id'] for r in rows):raise ValueError('Invalid edit scope')
    others=[r for r in rows if r['id'] not in ids]
    out=others+copy.deepcopy(replacements)
    if len({r['id'] for r in out})!=len(out):raise ValueError('Replacement ID collision')
    return out

