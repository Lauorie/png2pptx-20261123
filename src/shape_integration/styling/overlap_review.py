"""One isolated source-based coverage plan for partially overlapping faces.

Only fill and paint order are proposed. Geometry and strokes remain owned by
the ordinary style pipeline. No review is needed for disjoint/contained faces.
"""
import copy
import math
from concurrent.futures import as_completed
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.ops import unary_union
from . import views, occlusion
from .contracts import fill, number
from .geometry import parts, footprint
from ..common import save
from ..fitting.paths import sample_commands

PROTOCOL = 'shape_integration.surface_overlap.3'


def groups(table, geoms, options):
    """Geometry nominates an overlap; it does not decide opacity or order."""
    keys = [k for k in table if geoms[k].area > 0]
    parent = {k:k for k in keys}; pairs = []
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    tol = options['geometry_tolerance_px']
    for i,a in enumerate(keys):
        ga = geoms[a]
        for b in keys[i+1:]:
            gb = geoms[b]
            if not ga.intersects(gb):continue
            if ga.buffer(tol).covers(gb) or gb.buffer(tol).covers(ga):continue
            overlap = ga.intersection(gb)
            if overlap.area <= options['minimum_overlap_area_px']:continue
            parent[find(b)] = find(a)
            pairs.append({'a':a, 'b':b, 'box':list(overlap.bounds),
                          'fraction':overlap.area/min(ga.area,gb.area)})
    components = {}
    for pair in pairs:components.setdefault(find(pair['a']),[]).append(pair)
    return [{'keys':[k for k in keys if find(k)==root], 'pairs':members}
            for root,members in components.items()]


def stroke_support(geometry, style, pixels):
    """Observe source pigment along a fixed contour; never select/delete a part."""
    color = style.get('stroke')
    if not isinstance(color,str) or not style.get('width_px'):return None
    pigment = np.array([int(color[i:i+2],16) for i in (1,3,5)])
    radius = max(1.,style['width_px']/2+1.)
    offsets = np.linspace(-radius,radius,max(3,int(math.ceil(4*radius))+1))
    fragments = list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
    hits = [];length = 0.
    for fragment in fragments:
        if fragment.is_empty or not hasattr(fragment,'coords') or fragment.length<1:continue
        distance = np.linspace(0.,fragment.length,max(3,int(math.ceil(2*fragment.length))+1))
        points = np.array([fragment.interpolate(float(d)).coords[0] for d in distance])
        tangent = np.gradient(points,axis=0);normal = np.c_[-tangent[:,1],tangent[:,0]]
        normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-9)
        xy = np.rint(points[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
        valid = (xy[:,:,0]>=0)&(xy[:,:,0]<pixels.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<pixels.shape[0])
        xy[:,:,0] = np.clip(xy[:,:,0],0,pixels.shape[1]-1)
        xy[:,:,1] = np.clip(xy[:,:,1],0,pixels.shape[0]-1)
        rgb = pixels[xy[:,:,1],xy[:,:,0]].astype(float)
        hits.extend(np.any((np.linalg.norm(rgb-pigment,axis=2)<45)&valid,axis=1).tolist())
        length += fragment.length
    if not hits:return None
    return {'length_px':round(length,2),'source_pigment_fraction':round(float(np.mean(hits)),3),
            'samples':len(hits),'normal_radius_px':round(radius,2),'probe_colour':color}


def source_support(group, geoms, styles, mapping, pixels):
    aliases = {k:a for a,k in mapping.items()};result = []
    for pair in group['pairs']:
        a,b = pair['a'],pair['b']
        margin = max(1.,max(styles[k].get('width_px') or 0 for k in (a,b))/2+1.)
        others = unary_union([geoms[k].buffer(margin) for k in group['keys'] if k not in (a,b)])
        observations = []
        for key,peer in [(a,b),(b,a)]:
            edge = geoms[key].boundary
            inside = edge.intersection(geoms[peer].buffer(-margin)).difference(others)
            outside = edge.difference(geoms[peer].buffer(margin).union(others))
            observations.append({'contour':aliases[key],'inside_face':aliases[peer],
               'inside':stroke_support(inside,styles[key],pixels),
               'exposed_outside':stroke_support(outside,styles[key],pixels)})
        if any(o['inside'] for o in observations):result.extend(observations)
    return result


def source_views(source, table, mapping, pairs, folder, options):
    """One board: untouched source pixels beside a separate geometry reference."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    aliases = {k:a for a,k in mapping.items()}
    cards = []; records = []; font = ImageFont.truetype(views.FONT,22)
    with Image.open(source) as original:
        image = original.convert('RGB')
        whole = views.crop_box([table[k] for k in mapping.values()],image.size,
                              options['crop_padding'],options['crop_min_padding_px'])
        windows = [(whole,list(mapping.values()),'Whole overlap group')]
        # One representative intersection adds detail without repeated panels.
        for pair in sorted(pairs,key=lambda p:-p['fraction'])[:1]:
            b = np.asarray(pair['box'])
            pad = max(8., .25*max(b[2:]-b[:2]))
            box = [max(0,math.floor(b[0]-pad)), max(0,math.floor(b[1]-pad)),
                   min(image.width,math.ceil(b[2]+pad)), min(image.height,math.ceil(b[3]+pad))]
            windows.append((box,[pair['a'],pair['b']], 'Intersection: '+aliases[pair['a']]+' / '+aliases[pair['b']]))
        for box,keys,title in windows:
            size = (480,480); scale,offset = views.crop_transform(box,size)
            clean = views.resample_crop(image,box,size,scale,offset)
            locator = Image.new('RGB',size,'white')
            draw = ImageDraw.Draw(locator)
            labels = []; occupied = []
            for i,key in enumerate(keys):
                color = ['#E6007E','#0071BC','#00A651','#D35400'][i%4]
                alias = aliases[key]; labels.append(alias); anchors = []
                for path in views.subpaths(table[key]):
                    points = sample_commands(path,96)
                    xy = [((x-box[0]+.5)*scale-.5+offset[0],
                           (y-box[1]+.5)*scale-.5+offset[1]) for x,y in points]
                    if len(xy)>1:draw.line(xy,fill=color,width=2)
                    anchors.extend(xy[::max(1,len(xy)//12)])
                tb=draw.textbbox((0,0),alias,font=font)
                anchor,rect=views.label_position(anchors,size,(tb[2]-tb[0]+8,tb[3]-tb[1]+8),occupied)
                occupied.append(rect);draw.line([anchor,(rect[0],rect[1])],fill=color,width=1)
                draw.rectangle(rect,fill='white');draw.text((rect[0]+4-tb[0],rect[1]+4-tb[1]),alias,font=font,fill=color)
            card = Image.new('RGB',(980,552),'#f3f4f5');draw = ImageDraw.Draw(card)
            draw.text((6,3),title,font=font,fill='black')
            for i,(label,tile) in enumerate([('SOURCE: appearance evidence',clean),('GEOMETRY: IDs only',locator)]):
                draw.text((i*490+4,32),label,font=font,fill='black');card.paste(tile,(i*490+4,68))
            records.append({'parts':labels,'source_box':box,'scale':scale,'offset':offset})
            cards.append(card)
    sheet = Image.new('RGB',(980,552*len(cards)),'white')
    for i,card in enumerate(cards):sheet.paste(card,(0,i*552))
    path = folder/'source_board.png';sheet.save(path);save(folder/'frames.json',records)
    return [('LEFT: original SOURCE pixels. RIGHT: separate fitted geometry reference; '
             'complete contours identify IDs and do not represent source visibility or fill.',path)],records


def validate(raw, mapping, styles):
    rows = raw.get('_source_rows',raw.get('parts')) if isinstance(raw,dict) else None
    if not isinstance(rows,list):raise ValueError('Expected parts list')
    result = {}; edges = []
    for row in rows:
        alias = row.get('id') if isinstance(row,dict) else None
        if alias not in mapping or alias in result:raise ValueError('Unknown or repeated part ID')
        status = row.get('status')
        if status not in ('supported','uncertain'):raise ValueError('Expected supported/uncertain')
        if not isinstance(row.get('source_evidence'),str) or not row['source_evidence'].strip():
            raise ValueError('Source evidence is required')
        old = styles[mapping[alias]]
        if status=='supported' and 'fill' not in row:raise ValueError('A source plan needs an explicit fill')
        if status=='uncertain':
            value = {'fill':old.get('fill'),'fill_opacity':old.get('fill_opacity',1)}
        else:
            f = fill(row.get('fill'));opacity = row.get('fill_opacity',1)
            value = {'fill':f,'fill_opacity':1. if f is None and opacity is None else number(opacity,'fill_opacity')}
        if not 0<=value['fill_opacity']<=1 or value['fill'] is None and value['fill_opacity']!=1:
            raise ValueError('Invalid fill opacity')
        behind = row.get('behind',[])
        if not isinstance(behind,list) or any(a not in mapping or a==alias for a in behind):
            raise ValueError('behind must identify other group parts')
        if status=='uncertain':
            value = {k:old.get(k,1 if k=='fill_opacity' else None) for k in value}
        if status=='uncertain':behind = []
        result[alias] = {**value,'status':status,'behind':list(dict.fromkeys(behind)),
                         'source_evidence':row['source_evidence']}
        edges.extend((alias,a) for a in behind)
    if set(result)!=set(mapping):raise ValueError('Every overlap part needs one decision')
    if occlusion.stable_order(list(mapping),edges)[1]:raise ValueError('Cyclic source paint order')
    return {'parts':result,'_source_rows':rows}


def review(scene, styles, relations, source, folder, runtime, options, pool):
    """Return a proposal and its evidence; the caller owns final native export."""
    from .controller import observed_order
    table = parts(scene); geoms = {k:footprint(p)[0] for k,p in table.items()}
    nominated = groups(table,geoms,options)
    pixels = None
    if nominated:
        with Image.open(source) as image:pixels = np.asarray(image.convert('RGB'))
    proposed = copy.deepcopy(styles); ordered = copy.deepcopy(relations)
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def ask(index,group):
        root = folder/f'group_{index:03d}'
        mapping = {f'P{i}':k for i,k in enumerate(group['keys'])}
        images,records = source_views(source,table,mapping,group['pairs'],root/'input',options)
        prompt = {'protocol':PROTOCOL,
          'task':'Plan fill, opacity and back-to-front order of the identified overlapping faces '
                 'from the original SOURCE. The upper row shows the whole group; the lower row '
                 'enlarges one intersection with context on both sides.',
          'rules':[
            'Trace each contour in SOURCE. A rear contour interrupted by another face supports '
            'an opaque foreground fill and that face\'s front position. White is an opaque fill; '
            'stroke dashes and surface transparency are independent.',
            'Use fill:null when rear contours remain visible unchanged through a face; use partial '
            'opacity for visible blending. A plain white interior alone does not establish opacity.',
            'A behind:[B] means draw A first, then B on top. Use the interrupted source contours '
            'to determine order. Contained details belong above their own enclosing surface.',
            'source_stroke_support measures source pigment near the fitted contours, excluding '
            'other faces in this group. Compare inside support with the same contour exposed '
            'outside and with the reverse contour. If A stays visible inside B while B '
            'disappears inside A, A covers B: output B behind:[A]. Short spans or weak exposed '
            'support may be inconclusive; the source pixels remain the appearance evidence.',
            'Return one source-supported plan per ID. Use uncertain for unresolved coverage. '
            'Geometry, strokes, text and image assets are fixed.'],
          'targets':[{'id':a,'contour':views.contour_identity(table[k])} for a,k in mapping.items()],
          'detail_windows':records,
          'source_stroke_support':source_support(group,geoms,styles,mapping,pixels),
          'output':{'parts':[{'id':'P0','status':'supported|uncertain',
                     'fill':'#RRGGBB|null|linear_gradient object', 'fill_opacity':1,'behind':[],
                     'source_evidence':'visible contour evidence'}]}}
        save(root/'job.json',{'mapping':mapping,'prompt':prompt})
        result = runtime.for_stage(shape_schema_retries=0,reasoning_effort='low').call(root/'calls','overlap',
                   'Judge only source-supported surface coverage and paint order. Return one concise JSON.',
                   prompt,images,lambda raw:validate(raw,mapping,styles),min(options['max_tokens'],12288))
        return {'mapping':mapping,**result}
    futures = {pool.submit(ask,i,g):i for i,g in enumerate(nominated)}; reports = {}; touched = set()
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result(); reports[str(index)] = result
            for alias,row in result['parts'].items():
                key = result['mapping'][alias]
                if row['status']=='supported':
                    if (row['fill'],row['fill_opacity'])!=(styles[key].get('fill'),styles[key].get('fill_opacity',1)):
                        proposed[key].update({k:row[k] for k in ('fill','fill_opacity')})
                        touched.add(table[key]['object_id'])
                if observed_order(table,geoms,ordered,key,
                    behind=[result['mapping'][a] for a in row['behind']],origin='surface_overlap',scene=scene):
                    touched.add(table[key]['object_id'])
                    touched.update(table[result['mapping'][a]]['object_id'] for a in row['behind'])
        except Exception as exc:reports[str(index)] = {'error':str(exc)}
    if touched:
        with Image.open(source) as image:
            ordered = occlusion.order_internal_by_paint(scene,table,ordered,proposed,geoms,
                             options['geometry_tolerance_px'],source=image)
        for oid in ordered['part_orders']:
            if oid not in touched:ordered['part_orders'][oid] = relations['part_orders'][oid]
        ordered['flat_order'] = [k for oid in ordered['object_order'] for k in ordered['part_orders'][oid]]
    report = {'protocol':PROTOCOL,'model_calls':len(nominated),'groups':reports,
              'changed_fills':[k for k in table if styles[k]!=proposed[k]],
              'order_changed':relations['flat_order']!=ordered['flat_order'],
              'unresolved_groups':[k for k,v in reports.items() if v.get('error') or
                    any(p['status']=='uncertain' for p in v.get('parts',{}).values())]}
    save(folder/'audit.json',report)
    return proposed,ordered,report
