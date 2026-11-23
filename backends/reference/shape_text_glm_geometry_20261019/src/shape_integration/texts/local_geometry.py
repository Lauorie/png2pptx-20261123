"""Compact neighborhood localization with explicit reading-direction evidence."""
from concurrent.futures import as_completed
from copy import deepcopy
import math
import statistics

from PIL import Image, ImageDraw, ImageFont
from ..common import read, save, valid_box, intersect

SYSTEM = ('You localize existing text in scientific figures. Preserve every supplied ID, '
          'transcription and mathematical expression. Return JSON geometry only. '
          'Use source glyphs as evidence, not the rough colored boxes.')


def refine_inventory(inventory, source, folder, runtime, opt, progress=lambda *a: None):
    """Crop review may correct content/colors; failed crops retain prior data."""
    from concurrent.futures import ThreadPoolExecutor
    from .numeric_routing import inventory as route_inventory
    result=route_inventory(inventory,'before_crop_review')
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows,errors=refine(deepcopy(result['texts']),source,folder,runtime,
                {**opt,'content_review_enabled':True,'geometry_max_targets':6},pool,progress)
    except Exception as exc:
        result['regions_needing_review']=result.get('regions_needing_review',[])+[
            {'stage':'glm_crop_review','policy':'previous_valid_inventory','reason':str(exc)}]
        save(folder/'fallback.json',{'reason':str(exc),'policy':'previous_valid_inventory'})
        return result
    before={r['id']:r for r in inventory['texts']}
    if len({r['id'] for r in rows})!=len(rows) or any(
            r['id'] not in before and r.get('parent_id') not in before for r in rows):
        result['regions_needing_review']=result.get('regions_needing_review',[])+[
            {'stage':'glm_crop_review','reason':'Invalid identity in new reply; retained previous inventory'}]
        return result
    result['texts']=rows
    summary=read(folder/'summary.json')
    warnings=[{'stage':'glm_crop_geometry','reason':e} for e in errors]
    if summary.get('unresolved_geometry',0):
        warnings.append({'stage':'glm_crop_geometry','reason':'Some targets retain unverified initial boxes',
                         'count':summary['unresolved_geometry']})
    if summary.get('unresolved_ids'):
        warnings.append({'stage':'glm_crop_review','reason':'Unresolved targets retained',
                         'ids':summary['unresolved_ids']})
    result['regions_needing_review']=result.get('regions_needing_review',[])+warnings
    result['crop_geometry_summary']=summary
    return route_inventory(result,'after_crop_review')


def center(row):
    a,b,c,d=row['bbox_2d'];return ((a+c)/2,(b+d)/2)


def compact_groups(rows, size, max_targets=6):
    """Complete-link grouping from the 20260928 compact-neighborhood trial."""
    groups=[[r] for r in rows]
    def cost(a,b):
        group=a+b
        if len(group)>max_targets:return math.inf
        heights=[min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group]
        radius=max(160*max(size)/2048,statistics.median(heights)*7)
        distance=max(math.dist(center(x),center(y)) for x in group for y in group)
        if distance>radius:return math.inf
        box=union(group)
        return distance+.05*math.hypot(box[2]-box[0],box[3]-box[1])
    while len(groups)>1:
        value,i,j=min((cost(a,b),i,j) for i,a in enumerate(groups) for j,b in enumerate(groups) if i<j)
        if not math.isfinite(value):break
        groups[i]+=groups.pop(j)
    return groups


def union(rows):
    return [min(r['bbox_2d'][0] for r in rows),min(r['bbox_2d'][1] for r in rows),
            max(r['bbox_2d'][2] for r in rows),max(r['bbox_2d'][3] for r in rows)]


def direction_board(source, rows, path):
    """Unrotate source crops at candidate angles: a readable word resolves ±90."""
    tiles=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    with Image.open(source) as image:
        for row in rows:
            x0,y0,x1,y1=row['bbox_2d']
            if abs(row['rotation_deg'])<20 and y1-y0<1.3*(x1-x0):continue
            pad=max(6,min(x1-x0,y1-y0)*.3)
            crop=image.crop((max(0,math.floor(x0-pad)),max(0,math.floor(y0-pad)),
                             min(image.width,math.ceil(x1+pad)),min(image.height,math.ceil(y1+pad)))).convert('RGB')
            angles=list(dict.fromkeys([row['rotation_deg'],(row['rotation_deg']+360)%360-180,0,90,-90]))
            panel=Image.new('RGB',(310*len(angles),160),'white');draw=ImageDraw.Draw(panel)
            for i,angle in enumerate(angles):
                # PIL positive rotates CCW: undo the candidate clockwise angle.
                view=crop.rotate(angle,expand=True,fillcolor='white',resample=Image.Resampling.BICUBIC)
                factor=min(290/view.width,120/view.height,4)
                view=view.resize((max(1,round(view.width*factor)),max(1,round(view.height*factor))),Image.Resampling.LANCZOS)
                panel.paste(view,(i*310+(310-view.width)//2,32+(120-view.height)//2))
                draw.text((i*310+6,6),f'{row["id"]}: angle {angle:g} deg',font=font,fill='black')
            tiles.append(panel)
    if not tiles:return False
    board=Image.new('RGB',(max(t.width for t in tiles),sum(t.height for t in tiles)),'white')
    y=0
    for tile in tiles:board.paste(tile,(0,y));y+=tile.height
    board.save(path);return True


def validate(value, ids, size):
    # Accept an unambiguous list envelope; never repair or invent coordinates.
    if isinstance(value,list):value={'items':value}
    if not isinstance(value,dict) or not isinstance(value.get('items'),list):raise ValueError('Expected items array')
    items=value['items']
    if len(items)!=len(ids) or {r.get('id') for r in items}!=set(ids):raise ValueError('Return every target ID exactly once')
    for row in items:
        if set(row)-{'id','status','bbox_2d','rotation_deg','direction_status','reason'}:raise ValueError('Only geometry fields allowed')
        if row.get('status') not in ('located','clipped','ambiguous'):raise ValueError('Invalid location status')
        if row['status']=='located' and not valid_box(row.get('bbox_2d'),size):raise ValueError('Invalid numeric bbox')
        if row['status']!='located' and row.get('bbox_2d') is not None:raise ValueError('Unresolved bbox must be null')
        if row.get('direction_status') not in ('confirmed','ambiguous'):raise ValueError('Direction status required')
        angle=row.get('rotation_deg')
        if row['direction_status']=='confirmed':
            if type(angle) not in (int,float) or not math.isfinite(angle) or not -180<=angle<=180:raise ValueError('Invalid clockwise angle')
        elif angle is not None:raise ValueError('Uncertain direction must be null')
    return value


def refine(rows,source,folder,runtime,opt,pool,progress=lambda *a:None):
    if opt.get('content_review_enabled',False):
        from .local_content import refine as joint_refine
        return joint_refine(rows,source,folder,runtime,opt,pool,progress)
    from .controller import frame, mapped_box, local_hints
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:size=list(im.size)
    groups=compact_groups(rows,size,opt.get('geometry_max_targets',6))
    save(folder/'input_rows.json',rows)
    save(folder/'plan.json',{'groups':[[r['id'] for r in g] for g in groups],
        'max_targets':opt.get('geometry_max_targets',6),'strategy':'complete_link_compact',
        'content_frozen':True,'allowed_updates':['bbox_2d','rotation_deg']})
    stage=runtime.for_stage(reasoning_effort=opt.get('geometry_reasoning_effort','low'),shape_schema_retries=1)
    def job(index,group):
        root=folder/f'C{index:03d}';root.mkdir(exist_ok=True)
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group)
        pad=max(32*max(size)/2048,short*2.5)
        f=frame(source,root/'input',{**opt,'crop_min_padding_px':pad,'crop_padding':0,'crop_scale':100},union(group))
        clean=root/'input/clean.png'
        local_hints(clean,group,f,root/'input/ids.png')
        images=[('CLEAN crop. Bboxes refer only to this image.',clean),('Rough identity hints, not ground truth geometry.',root/'input/ids.png')]
        if direction_board(source,group,root/'input/directions.png'):
            images.append(('Reading-direction candidates only; NEVER use these panel coordinates for bbox.',root/'input/directions.png'))
        prompt={'task':'Refine complete glyph ink bounding boxes and verify reading direction.',
            'image_size':f['size'],
            'targets':[{'id':r['id'],'text':r['text'],'runs':r['runs'],'kind':r.get('kind','text'),
                        'latex':r.get('latex',''),'rough_rotation_deg':r['rotation_deg']} for r in group],
            'rules':['Return each fixed ID once. Do not change content, formula structure, style, split or merge.',
                     'bbox_2d uses CLEAN crop pixel centers: [xmin,ymin,xmax,ymax]. Include every dot, accent, subscript, superscript, radical and fraction bar. Exclude neighboring words and node boundaries.',
                     'rotation_deg is CLOCKWISE from normal upright horizontal left-to-right text. +90 means first character above last (top-to-bottom). -90 means first character below last (bottom-to-top). 180 is upside-down horizontal.',
                     'Determine orientation by reading actual glyphs, NEVER from the tall bbox or rough_rotation alone. If direction candidates are supplied, choose the angle whose unrotated crop reads normally left-to-right with upright glyphs.',
                     'Adjacent rotated columns belong to distinct IDs. Preserve their physical identity even when their initial bboxes are inaccurate.',
                     'For arbitrary slanted text use the actual clockwise angle. If orientation is unclear, direction_status=ambiguous and rotation_deg=null; do not guess.',
                     'For clipped or ambiguous location, bbox_2d=null and status=clipped or ambiguous. Do not invent invisible glyph extents.'],
            'example':{'items':[{'id':group[0]['id'],'status':'located','bbox_2d':[20,30,180,90],
                                 'rotation_deg':0,'direction_status':'confirmed','reason':''}]}}
        save(root/'prompt.json',prompt)
        validator=lambda v:validate(v,[r['id'] for r in group],f['size'])
        def repair(previous,error):
            content=(previous/'content.txt').read_text() if (previous/'content.txt').exists() else ''
            return {'system':SYSTEM,'prompt':{**prompt,'format_repair':{'error':error,'previous':content}},'images':images,'validator':validator}
        try:
            if (root/'result.json').exists():
                return index,validator(read(root/'result.json')),f,None
            # A complete API response can fail only due to its envelope. Reuse
            # the saved parsed data after validation, without calling the model.
            for parsed in sorted((root/'calls').glob('locate_*/parsed_response.json'),reverse=True):
                status=read(parsed.parent/'stream_status.json')
                if status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
                try:recovered=validator(read(parsed))
                except ValueError:continue
                save(root/'result.json',recovered)
                save(root/'recovery.json',{'source':str(parsed),'policy':'validate completed response; list envelope only','api_calls':0})
                return index,recovered,f,None
            result=stage.call(root/'calls','locate',SYSTEM,prompt,images,validator,opt['max_tokens'],retry_builder=repair)
            save(root/'result.json',result)
            return index,result,f,None
        except Exception as exc:
            save(root/'error.json',{'error':str(exc)})
            return index,None,f,str(exc)
    results={};futures=[pool.submit(job,i,g) for i,g in enumerate(groups)]
    for fut in as_completed(futures):
        index,result,f,error=fut.result();results[index]=(result,f,error)
        progress('local_geometry',len(results),len(groups))
    output=deepcopy(rows);by_id={r['id']:r for r in output};audit=[];errors=[]
    for index,group in enumerate(groups):
        result,f,error=results[index]
        if error:errors.append(f'C{index:03d}: {error}')
        proposals={r['id']:r for r in result['items']} if result else {}
        for old in group:
            proposal=proposals.get(old['id']);new=by_id[old['id']]
            record={'id':old['id'],'group':f'C{index:03d}','old_bbox':old['bbox_2d'],'old_angle':old['rotation_deg'],'proposal':proposal,'bbox_updated':False,'angle_updated':False}
            if proposal:
                if proposal['status']=='located':
                    box=mapped_box(proposal['bbox_2d'],f,True)
                    if valid_box(box,size):new['bbox_2d']=box;record['bbox_updated']=True
                    else:record['rejected']='outside source after coordinate mapping'
                if proposal['direction_status']=='confirmed':
                    new['rotation_deg']=(proposal['rotation_deg']+180)%360-180
                    record['angle_updated']=new['rotation_deg']!=old['rotation_deg']
            assert {k:v for k,v in new.items() if k not in ('bbox_2d','rotation_deg')}=={k:v for k,v in old.items() if k not in ('bbox_2d','rotation_deg')}
            record.update(bbox=new['bbox_2d'],angle=new['rotation_deg']);audit.append(record)
    save(folder/'audit.json',audit)
    save(folder/'summary.json',{'groups':len(groups),'targets':len(rows),'bbox_updated':sum(r['bbox_updated'] for r in audit),
        'angles_changed':sum(r['angle_updated'] for r in audit),'failed_groups':len(errors),'errors':errors,
        'unresolved_geometry':sum(not r['bbox_updated'] for r in audit),'content_unchanged':True})
    return output,errors
