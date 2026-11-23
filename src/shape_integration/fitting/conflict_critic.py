"""Bounded source observation when a revision retypes an existing primitive."""
import math
from pathlib import Path
from PIL import Image,ImageDraw
from ..common import read,save
from .paths import command_strokes


def changed_primitives(before,after):
    # A context request explicitly rejected the completeness of the old crop.
    # Its clipped primitive cannot veto the newly observed complete contour.
    if before.get('frame',{}).get('crop_bounds_global_half_open') != after.get('frame',{}).get('crop_bounds_global_half_open'):
        return []
    old={p['id']:p for p in before['parts'] if p['status']=='ok'}
    new={p['id']:p for p in after['parts'] if p['status']=='ok'}
    if set(old)!=set(new):return []  # A rollback must not erase newly added geometry.
    return [key for key,p in old.items() if key in new and
            p['kind'] in ('rectangle','rounded_rectangle','ellipse','cylinder') and new[key]['kind']!=p['kind']]


def inspect_revision(source,before,after,folder,runtime,options):
    """Choose between existing geometries; never infer replacement coordinates."""
    ids=changed_primitives(before,after)
    if not ids:return None
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    old={p['id']:p for p in before['parts']};new={p['id']:p for p in after['parts']}
    images=[];targets=[]
    with Image.open(source) as original:
        original=original.convert('RGB')
        for i,key in enumerate(ids[:3]):
            a,b=old[key],new[key];boxes=[a['bbox_2d'],b['bbox_2d']]
            pad=max(4.,.08*min(a['bbox_2d'][2]-a['bbox_2d'][0],a['bbox_2d'][3]-a['bbox_2d'][1]))
            crop=[max(0,math.floor(min(v[0] for v in boxes)-pad)),max(0,math.floor(min(v[1] for v in boxes)-pad)),
                  min(original.width,math.ceil(max(v[2] for v in boxes)+pad+1)),min(original.height,math.ceil(max(v[3] for v in boxes)+pad+1))]
            clean=original.crop(crop);scale=min(4.,1200/max(clean.size));size=tuple(max(2,round(v*scale)) for v in clean.size)
            clean=clean.resize(size,Image.Resampling.LANCZOS)
            path=folder/f'part_{i}_source.png';clean.save(path);images.append((f'Part P{i} CLEAN source',path))
            for name,part in [('before',a),('after',b)]:
                view=clean.copy();draw=ImageDraw.Draw(view)
                for stroke in command_strokes(part['commands']):
                    points=[((x-crop[0]+.5)*size[0]/(crop[2]-crop[0])-.5,
                             (y-crop[1]+.5)*size[1]/(crop[3]-crop[1])-.5) for x,y in stroke]
                    if len(points)>1:draw.line(points,fill='#d10070',width=max(1,round(scale)))
                path=folder/f'part_{i}_{name}.png';view.save(path);images.append((f'Part P{i} {name} fitted contour overlay',path))
            targets.append({'id':f'P{i}','before_kind':a['kind'],'after_kind':b['kind']})
    prompt={'protocol':'shape_integration.primitive_change_observation.1','targets':targets,
        'task':'Compare the BEFORE and AFTER contours with CLEAN at the same scale. '
               'The magenta paths are alternative predictions, not source edges. '
               'Check only whether retyping the primitive improves or degrades its visible contour.',
        'rules':['Observe the actual top/bottom/side edges of THIS face. Do not borrow a neighbor\'s '
                 'slanted extrusion or gray shadow as its boundary.',
                 'before: the revision visibly displaced a better source-aligned contour. '
                 'after: the changed contour clearly follows the source better. '
                 'unknown: neither is clearly correct or visible evidence is insufficient.',
                 'Judge pixels, not a preferred 3D template. No coordinates, new geometry or style changes. '
                 'If any reviewed primitive clearly regressed choose before; otherwise require evidence for after.',
                 'A dashed/dotted or antialiased stroke has changing raster edges around a smooth '
                 'centreline. Do not accept a scalloped freeform contour merely to trace those '
                 'gaps or raster bumps; those belong to stroke style, not shape geometry.'],
        'output':{'choice':'before|after|unknown','source_evidence':'one brief concrete source observation'}}
    def validate(raw):
        if not isinstance(raw,dict) or raw.get('choice') not in ('before','after','unknown'):
            raise ValueError('Expected before/after/unknown')
        if not isinstance(raw.get('source_evidence'),str) or not raw['source_evidence'].strip():
            raise ValueError('Source evidence is required')
        return {key:raw[key] for key in ('choice','source_evidence')}
    result=runtime.for_stage(shape_schema_retries=0).call(folder/'calls','observe',
        'Inspect a potentially regressed geometric correction. Return concise JSON only.',prompt,images,validate,
        min(options['check_max_tokens'],8192))
    save(folder/'result.json',{'changed_part_ids':ids,'observed_part_ids':ids[:3],'review':result})
    return result
