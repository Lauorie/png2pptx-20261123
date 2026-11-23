"""Source-grounded text-frame paint inside the existing composition stage.

Geometry and typography are inputs, never corrections. Only labels crossing a
frame or shaft are reviewed; all other text remains transparent. No raster mask,
shape trimming, broad image-error optimisation or extra pipeline is involved.
"""
import copy
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union
from .handoff import strokes, arrow_strokes, json_part_id, surviving_scene

PROTOCOL = 'source_text_stroke_fill.8'


def write_backings(path, rows, size):
    """Native label backings precede every glyph, including Office equations."""
    from zipfile import ZipFile, ZIP_DEFLATED
    from lxml import etree
    from .compose import P,A
    NS={'p':P,'a':A}
    with ZipFile(path) as archive:
        package={name:archive.read(name) for name in archive.namelist()}
    slide=etree.fromstring(package['ppt/slides/slide1.xml'])
    presentation=etree.fromstring(package['ppt/presentation.xml'])
    canvas=presentation.find('p:sldSz',NS)
    sx,sy=int(canvas.get('cx'))/size[0],int(canvas.get('cy'))/size[1]
    tree=slide.find('p:cSld/p:spTree',NS)
    # These generated backings are owned by this step. Reapplication starts
    # from glyphs, rather than accumulating stale opaque rectangles.
    for node in list(tree)[2:]:
        names={n.get('name','') for n in node.findall('.//p:cNvPr',NS)}
        if names and all(name.startswith('label backing ') for name in names):
            tree.remove(node)
    ident=max(int(e.get('id')) for e in tree.findall('.//p:cNvPr',NS))+1
    insertion=2
    for row in rows:
        if not row.get('background_fill') or not row.get('background_box_px'):continue
        x0,y0,x1,y1=row['background_box_px'];color=row['background_fill'].lstrip('#')
        shape=etree.fromstring(f'''<p:sp xmlns:p="{NS['p']}" xmlns:a="{NS['a']}">
          <p:nvSpPr><p:cNvPr id="{ident}" name="label backing {ident}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
          <p:spPr><a:xfrm><a:off x="{round(x0*sx)}" y="{round(y0*sy)}"/><a:ext cx="{round((x1-x0)*sx)}" cy="{round((y1-y0)*sy)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>
        </p:sp>''')
        tree.insert(insertion,shape);insertion+=1;ident+=1
    package['ppt/slides/slide1.xml']=etree.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        for name,data in package.items():archive.writestr(name,data)


def region(row):
    if row.get('background_box_px') is not None:
        return box(*row['background_box_px'])
    x, y, w, h = row['frame_px']
    return affinity.rotate(box(x, y, x+w, y+h), row.get('rotation_deg', 0), origin=(x+w/2, y+h/2))


def part_strokes(part):
    """Use the same surviving border segments as the native composition."""
    paths=strokes(part.get('commands', []),64)
    suppressed=set(part.get('_composition_suppressed_segments',()))
    if not suppressed:return paths
    commands=part.get('commands',[])
    if len(paths)!=1 or not commands or commands[-1]!=['Z'] or any(c[0] not in ('M','L','Z') for c in commands):
        raise ValueError('Segment ownership requires one linear closed contour')
    vertices=np.asarray([c[1:3] for c in commands if c[0] in ('M','L')],float)
    if len(vertices)>1 and np.linalg.norm(vertices[0]-vertices[-1])<1e-5:vertices=vertices[:-1]
    return [vertices[[i,(i+1)%len(vertices)]] for i in range(len(vertices)) if i not in suppressed]


def part_geometry(part):
    paths = strokes(part.get('commands', []), 64)
    lines = unary_union([LineString(p) for p in part_strokes(part) if len(p)>1])
    closed = part.get('commands', [])[-1:]==[['Z']]
    poly = Polygon(paths[-1]) if closed and paths and len(paths[-1])>=4 else None
    return lines, poly if poly is not None and poly.is_valid else None


def frame_candidates(row, scene, styles):
    mask = region(row); result = []
    for key, part in scene['parts'].items():
        if part.get('status')!='ok' or part.get('commands',[])[-1:]!=[['Z']]:
            continue
        style = styles.get(key, {})
        if not style.get('stroke'):continue
        if len(strokes(part['commands']))!=1:continue
        line, poly = part_geometry(part)
        if poly is None or mask.covers(poly):continue
        # Real boundary crossing, not text merely inside or adjacent to a panel.
        if line.intersection(mask).length>1 and mask.intersection(poly).area>1 and mask.difference(poly).area>1:
            result.append(key)
    return result


def arrow_region(row):
    if row.get('kind')=='block_arrow':
        polygons=[Polygon(p) for p in row.get('contours', []) if len(p)>=3]
        # An older recovered filled contour can self-intersect. Only this
        # protective mask is conservatively enlarged; native arrow geometry
        # and its source artifact are never repaired or rewritten here.
        return unary_union([p if p.is_valid else p.convex_hull for p in polygons])
    lines = arrow_strokes(row)
    width = max(1., row.get('style', {}).get('width_px') or 1.)
    protected = [LineString(p).buffer(width/2+1) for p in lines if len(p)>1]
    # Include the full native head envelope, not just the shaft endpoint.
    for i, end in enumerate(('start', 'end')):
        head = row.get('heads', ['none', 'none'])[i]
        style_head = row.get('style', {}).get(end)
        if head not in (None, 'none') or style_head:
            h = style_head if isinstance(style_head, dict) else {}
            factor = 6 if 'lg' in (h.get('width'), h.get('length')) else 4
            from shapely.geometry import Point
            protected.append(Point(row[end]).buffer(factor*width+2))
    return unary_union(protected)


def stroke_candidates(row,scene,styles,arrows):
    """Nominate actual intersecting shafts as well as closed frame borders."""
    mask=region(row);shape_keys=[];arrow_ids=[]
    for key,part in scene['parts'].items():
        cs=part.get('commands',[]);style=styles.get(key,{})
        if part.get('status')!='ok' or not style.get('stroke') or not cs or cs[-1][0]=='Z':continue
        line,_=part_geometry(part)
        if line.intersection(mask).length>1 and line.difference(mask).length>2:shape_keys.append(key)
    for arrow in arrows:
        if arrow.get('kind')=='block_arrow':continue
        line=unary_union([LineString(p) for p in arrow_strokes(arrow) if len(p)>1])
        if line.intersection(mask).length>1 and line.difference(mask).length>2:arrow_ids.append(arrow['id'])
    return shape_keys,arrow_ids


def head_region(row):
    from shapely.geometry import Point
    width=max(1.,row.get('style',{}).get('width_px') or 1.);heads=[]
    for i,end in enumerate(('start','end')):
        native=row.get('style',{}).get(end) or {};kind=native.get('type') if isinstance(native,dict) else native
        kind=kind or (row.get('heads') or ['none','none'])[i]
        if kind not in (None,'none'):
            large=isinstance(native,dict) and 'lg' in (native.get('width'),native.get('length'))
            heads.append(Point(row[end]).buffer((6 if large else 4)*width+2))
    return unary_union(heads)


def stroke_backing(row, approved, approved_arrows, scene, styles, arrows):
    """Cover only approved ink inside the observed label, keeping nearby borders."""
    if not row.get('background_box_px'):return row
    supports=[]
    for key in approved:
        width=styles.get(key,{}).get('width_px') or 1.
        supports.extend(LineString(p).buffer(width/2+1.) for p in part_strokes(scene['parts'][key]) if len(p)>1)
    for arrow in arrows:
        if arrow['id'] not in approved_arrows:continue
        width=arrow.get('style',{}).get('width_px') or 1.
        supports.extend(LineString(p).buffer(width/2+1.) for p in arrow_strokes(arrow) if len(p)>1)
    covered=region(row).intersection(unary_union(supports))
    return {**row,'background_box_px':list(covered.bounds)} if not covered.is_empty else row


def visible_padding_stroke(row,source,color,paths,stroke,width):
    """Never erase source-visible continuation outside the observed glyph box."""
    if not stroke or 'bbox_2d' not in row:return False
    protected=region(row).difference(box(*row['bbox_2d']).buffer(1.))
    if protected.is_empty:return False
    from .handoff import resample
    from shapely.geometry import Point
    q=resample(paths,512)
    if len(q)<2:return False
    use=np.array([protected.contains(Point(p)) for p in q])
    if use.sum()<3:return False
    tangent=np.gradient(q,axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]]
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    offsets=np.arange(-max(1.,min(3.,width)),max(1.,min(3.,width))+.25,.5)
    xy=np.rint(q[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
    xy[:,:,0]=np.clip(xy[:,:,0],0,source.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,source.shape[0]-1)
    pixels=source[xy[:,:,1],xy[:,:,0]].astype(float)
    pigment=np.array([int(stroke[i:i+2],16) for i in (1,3,5)])
    fill=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    ink=np.linalg.norm(pixels-pigment,axis=2);back=np.linalg.norm(pixels-fill,axis=2)
    visible=np.any((ink<60)&(ink+12<back),axis=1)&use
    # Three adjacent source-supported samples, not one crossing pixel.
    return bool(np.any(visible[:-2]&visible[1:-1]&visible[2:]))


def visible_source_stroke(row, source, color, paths, stroke, width, glyph_pixels=None):
    """An approved backing cannot erase an independently visible source stroke.

    A native glyph mask explains text ink only inside this occurrence's source
    box. Evidence needs a continuous run in physical source pixels, so repeated
    resampling of one crossing pixel cannot become an occlusion constraint.
    """
    if not stroke or not paths:
        return False
    import cv2
    from shapely.geometry import Point
    pigment=np.array([int(stroke[i:i+2],16) for i in (1,3,5)])
    fill=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    glyph_mask=None
    if glyph_pixels is not None and row.get('bbox_2d'):
        if glyph_pixels.shape != source.shape:
            raise ValueError('Glyph and source evidence grids differ')
        glyph_mask=np.zeros(source.shape[:2],dtype='uint8')
        x0,y0,x1,y1=map(int,row['bbox_2d'])
        x0,y0=max(0,x0),max(0,y0);x1,y1=min(source.shape[1],x1+1),min(source.shape[0],y1+1)
        if x1>x0 and y1>y0:
            glyph_mask[y0:y1,x0:x1]=(np.max(255-glyph_pixels[y0:y1,x0:x1].astype(float),axis=2)>30)
        glyph_mask=cv2.dilate(glyph_mask,np.ones((3,3),'uint8'))
    mask=region(row)
    for path in paths:
        points=np.asarray(path,float)
        if len(points)<2:continue
        pieces=[]
        for a,b in zip(points[:-1],points[1:]):
            count=max(2,int(np.ceil(np.linalg.norm(b-a)))+1)
            pieces.append(a+np.linspace(0,1,count,endpoint=False)[:,None]*(b-a))
        q=np.vstack(pieces)
        tangent=np.gradient(q,axis=0)
        normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
        radius=max(1.,min(3.,width/2+1.))
        xy=np.rint(q[:,None]+normal[:,None]*np.arange(-radius,radius+.25,.5)[None,:,None]).astype(int)
        valid=(xy[:,:,0]>=0)&(xy[:,:,0]<source.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<source.shape[0])
        xy[:,:,0]=np.clip(xy[:,:,0],0,source.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,source.shape[0]-1)
        pixels=source[xy[:,:,1],xy[:,:,0]].astype(float)
        ink=np.linalg.norm(pixels-pigment,axis=2);back=np.linalg.norm(pixels-fill,axis=2)
        supported=valid&(ink<60)&(ink+12<back)
        if glyph_mask is not None:supported &= glyph_mask[xy[:,:,1],xy[:,:,0]]==0
        inside=np.array([mask.contains(Point(p)) for p in q])
        visible=inside&supported.any(1)
        changes=np.diff(np.r_[False,visible,False].astype(int))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if b-a>=3 and np.linalg.norm(np.diff(q[a:b],axis=0),axis=1).sum()>=max(3.,2*radius+width):
                return True
    return False


def validate(raw, candidates, palette):
    if not isinstance(raw, dict) or set(raw)!={'decision', 'fill', 'occluded_parts', 'evidence'}:
        raise ValueError('Return decision, fill, occluded_parts and evidence only')
    if raw['decision'] not in ('transparent', 'fill', 'unresolved'):
        raise ValueError('Unsupported decision')
    # Harmless wire-format repair only; never change a paint decision or its
    # evidence to make the response fit the schema.
    if raw['decision']!='fill' and raw['fill']=='null':raw={**raw,'fill':None}
    if not isinstance(raw['evidence'], str) or not raw['evidence'].strip():
        raise ValueError('Visible source evidence is required')
    ids = raw['occluded_parts']
    if not isinstance(ids, list) or any(k not in candidates for k in ids) or len(set(ids))!=len(ids):
        raise ValueError('Reference only the supplied frame parts')
    if raw['decision']=='fill':
        if not ids or not isinstance(raw['fill'], str) or raw['fill'].upper() not in palette:
            raise ValueError('Fill needs source-palette pigment and explicit occluded frame parts')
    elif raw['fill'] is not None or ids:
        raise ValueError('Transparent/unresolved must have null fill and no occluded parts')
    return raw


def safe_fill(row, approved, color, scene, styles, arrows, rows, images=(),*,approved_arrows=(),source_pixels=None,glyph_pixels=None):
    mask = region(row)
    for arrow in arrows:
        if mask.intersects(arrow_region(arrow)):
            if arrow['id'] not in approved_arrows or arrow.get('kind')=='block_arrow':
                return 'protected_arrow:'+arrow['id']
            if mask.intersects(head_region(arrow)):
                return 'protected_arrow_head:'+arrow['id']
            if arrow.get('style',{}).get('state')=='geometry_preview':
                # Default gray preview paint cannot prove source invisibility.
                # Keep glyphs transparent until import establishes real paint.
                return 'unresolved_arrow_source_paint:'+arrow['id']
            if source_pixels is not None and visible_padding_stroke(row,source_pixels,color,arrow_strokes(arrow),
                    arrow.get('style',{}).get('color'),arrow.get('style',{}).get('width_px') or 1.):
                return 'source_visible_arrow_in_text_padding:'+arrow['id']
            if source_pixels is not None and visible_source_stroke(row,source_pixels,color,arrow_strokes(arrow),
                    arrow.get('style',{}).get('color'),arrow.get('style',{}).get('width_px') or 1.,glyph_pixels):
                return 'source_visible_arrow_under_text_backing:'+arrow['id']
    for other in rows:
        if not row.get('background_box_px') and other['id']!=row['id'] and mask.intersection(region(other)).area>.5:
            return 'protected_other_text:'+other['id']
    for image in images:
        if mask.intersection(box(*image['bbox_2d'])).area>.5:
            return 'protected_image:'+image.get('id', image.get('name', 'image'))
    for key, part in scene['parts'].items():
        if part.get('status')!='ok':continue
        line, poly = part_geometry(part); style = styles.get(key, {})
        if key not in approved and style.get('stroke') and mask.intersects(line.buffer((style.get('width_px') or 1)/2+.5)):
            return 'protected_shape_stroke:'+key
        if key in approved and source_pixels is not None and visible_padding_stroke(row,source_pixels,color,
                part_strokes(part),style.get('stroke'),style.get('width_px') or 1.):
            return 'source_visible_border_in_text_padding:'+key
        if key in approved and source_pixels is not None and visible_source_stroke(row,source_pixels,color,
                part_strokes(part),style.get('stroke'),style.get('width_px') or 1.,glyph_pixels):
            return 'source_visible_shape_under_text_backing:'+key
        # Filling a differently coloured/transparent face would change its
        # appearance even when no boundary is crossed.
        fill = style.get('fill')
        if poly is not None and fill and mask.intersection(poly).area>.5:
            if not isinstance(fill, str) or fill.upper()!=color.upper() or style.get('fill_opacity', 1)<1:
                return 'protected_shape_fill:'+key
    return None


def select(rows, scene, style_data, arrow_scene, plan, source, folder, runtime, save, images=(),*,review_ids=None,glyph_native=None):
    folder.mkdir(parents=True, exist_ok=True)
    scene=surviving_scene(scene,plan)
    suppressed=plan.get('shape_segment_suppression',{})
    scene={**scene,'parts':{k:{**p,'_composition_suppressed_segments':
        suppressed.get(p['object_id']+'/'+p.get('id',json_part_id(k)),[])}
        for k,p in scene['parts'].items()}}
    styles = {k:v['style'] for k,v in style_data['parts'].items()}
    arrows = [r for r in arrow_scene['arrows'] if r['id'] not in set(plan.get('drop_arrow_names', []))]
    result = copy.deepcopy(rows); audit = []
    with Image.open(source) as opened: original = opened.convert('RGB')
    source_pixels=np.asarray(original)
    glyph_pixels=np.asarray(glyph_native.convert('RGB')) if glyph_native is not None else None
    for row in result:
        # Nominate and paint the observed label, not the typography frame's
        # ascent/descent padding. Backings are exported below ALL glyphs.
        row.pop('background_box_px',None)
        if row.get('bbox_2d') and not row.get('rotation_deg',0):
            row['background_box_px'] = list(box(*row['bbox_2d']).buffer(1.).bounds)
        row['background_fill'] = None
        row['background_evidence'] = 'Composition default: transparent; no approved source-grounded knockout.'

    def review(row):
        if review_ids is not None and row['id'] not in review_ids:
            return {'text_id':row['id'],'applied':False,'model_calls':0,'selection':'outside_requested_review'}
        keys = frame_candidates(row, scene, styles)
        lines,arrow_ids=stroke_candidates(row,scene,styles,arrows)
        record = {'text_id':row['id'], 'candidate_frame_parts':keys,'candidate_shape_lines':lines,
                  'candidate_arrows':arrow_ids, 'applied':False, 'model_calls':0}
        if not keys and not lines and not arrow_ids:
            return record
        root = folder/row['id']; root.mkdir(exist_ok=True)
        x0, y0, x1, y1 = region(row).bounds
        pad = max(24, (y1-y0)*2)
        crop = [max(0, math.floor(x0-pad)), max(0, math.floor(y0-pad)),
                min(original.width, math.ceil(x1+pad)), min(original.height, math.ceil(y1+pad))]
        clean = original.crop(crop)
        overlay = clean.copy(); draw = ImageDraw.Draw(overlay)
        draw.polygon([(x-crop[0], y-crop[1]) for x,y in region(row).exterior.coords], outline='#E00090', width=1)
        aliases = {f'F{i+1}':k for i,k in enumerate(keys)}
        aliases.update({f'L{i+1}':k for i,k in enumerate(lines)})
        arrow_aliases={f'A{i+1}':ident for i,ident in enumerate(arrow_ids)}
        arrow_lookup={a['id']:a for a in arrows}
        for ident, key in aliases.items():
            paths = part_strokes(scene['parts'][key])
            for points in paths:
                draw.line([(float(x-crop[0]), float(y-crop[1])) for x,y in points], fill='#0080DB', width=1)
            points = np.concatenate(paths)
            near = points[np.argmin(np.linalg.norm(points-[(x0+x1)/2, (y0+y1)/2], axis=1))]
            draw.text((float(np.clip(near[0]-crop[0],0,clean.width-20)), float(np.clip(near[1]-crop[1],0,clean.height-15))), ident, fill='#0080DB')
        for ident,key in arrow_aliases.items():
            for points in arrow_strokes(arrow_lookup[key]):
                draw.line([(float(x-crop[0]),float(y-crop[1])) for x,y in points],fill='#C45B00',width=1)
                near=points[len(points)//2]
                draw.text((float(np.clip(near[0]-crop[0],0,clean.width-20)),float(np.clip(near[1]-crop[1],0,clean.height-15))),ident,fill='#C45B00')
        model_scale=min(4.,max(1.,1080/max(clean.size)))
        model_size=tuple(max(2,round(v*model_scale)) for v in clean.size)
        clean.resize(model_size,Image.Resampling.LANCZOS).save(root/'clean.png')
        overlay.resize(model_size,Image.Resampling.LANCZOS).save(root/'identity.png')
        record.update(source_crop_half_open=crop,model_image_size=list(model_size))
        pixels = np.asarray(original.crop((max(0,int(x0)),max(0,int(y0)),min(original.width,math.ceil(x1)),min(original.height,math.ceil(y1))))).reshape(-1,3)
        quant = pixels//8; colors, counts = np.unique(quant, axis=0, return_counts=True)
        palette = []
        for index in np.argsort(-counts)[:8]:
            rgb = np.median(pixels[np.all(quant==colors[index], axis=1)], axis=0).round().astype(int)
            palette.append('#'+''.join(f'{v:02X}' for v in rgb))
        prompt = {'protocol':PROTOCOL, 'text':row.get('text', ''),
                  'frame_parts':[{'id':ident,'kind':scene['parts'][key]['kind'],
                      'recovered_stroke_color':styles[key].get('stroke'),
                      'recovered_line_pattern':styles[key].get('dash'),
                      'role':'Fitted identity hint; verify corresponding border in CLEAN'} for ident,key in aliases.items()],
                  'arrow_shafts':[{'id':ident,'role':'Possible shaft hidden by this label; endpoint heads remain protected',
                                   'style':arrow_lookup[key].get('style',{})} for ident,key in arrow_aliases.items()],
                  'source_palette':palette, 'images':['CLEAN original crop', 'Identification only: magenta text frame and blue fitted frame parts'],
                  'task':'Does this label hide a border or line in CLEAN? Choose transparent or a uniform native backing.',
                  'rules':['Read CLEAN first. Use fill only when a uniform label backing interrupts a named stroke; otherwise transparent. If unclear use unresolved.',
                           'A border stopping before the label and resuming after it along the same contour is evidence of a label interruption. No stroke under the glyphs is EXPECTED for an opaque backing; do not reject fill for that reason. For dashed borders distinguish one normal dash gap from a label-width interruption spanning multiple expected dashes. Check both continuations in CLEAN.',
                           'Confirm the same border/shaft in CLEAN, with a visible interruption caused by a uniform label backing. Blank background between characters or a fitted stroke crossing a glyph does not establish backing. Continuous source borders, nodes and arrowheads remain visible. Magenta is a paint proposal, not source ink.',
                           'Choose the backing pigment from source_palette, measured inside this label. The backing is below every text glyph; do not change text, geometry or fix an unrelated misplaced shape.'],
                  'output_schema':{'type':'object','additionalProperties':False,
                      'required':['decision','fill','occluded_parts','evidence'],
                      'properties':{'decision':{'enum':['transparent','fill','unresolved']},
                          'fill':{'anyOf':[{'type':'null'},{'enum':palette}],
                                  'description':'Use JSON null without quotes for transparent or unresolved.'},
                          'occluded_parts':{'type':'array','uniqueItems':True,'items':{'enum':list(aliases)+list(arrow_aliases)}},
                          'evidence':{'type':'string','minLength':1}}}}
        record['model_calls'] = 1
        try:
            reply = runtime.call(root/'calls', 'text_fill', 'Judge local text/frame occlusion from CLEAN. Return one JSON only.',
                                 prompt, [('CLEAN',root/'clean.png'), ('IDENTITY',root/'identity.png')],
                                 lambda raw:validate(raw, {**aliases,**arrow_aliases}, palette), 4096)
            record['review'] = reply
            if reply['decision']=='fill':
                approved = [aliases[k] for k in reply['occluded_parts'] if k in aliases]
                approved_arrows=[arrow_aliases[k] for k in reply['occluded_parts'] if k in arrow_aliases]
                backing=stroke_backing(row,approved,approved_arrows,scene,styles,arrows)
                rejection = safe_fill(backing, approved, reply['fill'], scene, styles, arrows, result, images,
                    approved_arrows=approved_arrows,source_pixels=source_pixels,glyph_pixels=glyph_pixels)
                if rejection:record['unresolved'] = rejection
                else:
                    if backing.get('background_box_px'):row['background_box_px']=backing['background_box_px']
                    row['background_fill'] = reply['fill'].upper(); row['background_evidence'] = reply['evidence']
                    record.update(applied=True, occluded_parts=approved,occluded_arrows=approved_arrows, fill=row['background_fill'])
        except (RuntimeError, ValueError) as exc:
            record['unresolved'] = str(exc)
        return record

    with ThreadPoolExecutor(max_workers=min(8, max(1,len(result)))) as pool:
        audit = list(pool.map(review, result))
    report = {'protocol':PROTOCOL, 'background_policy':'transparent_unless_source_confirmed_and_geometry_safe',
              'model_calls':sum(r['model_calls'] for r in audit), 'filled':sum(r['applied'] for r in audit),
              'reviews':audit, 'shape_arrow_geometry_and_styles_unchanged':True,
              'text_geometry_and_typography_unchanged':True}
    save(folder/'selection.json', report)
    return result, report
