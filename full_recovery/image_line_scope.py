"""Source-covered image interiors participate in line ownership too.

An original-pixel icon with a measured rim already covers its internal native
linework. External/rim strokes and any cleanup intersection remain independent.
"""
import math
import numpy as np
from PIL import Image,ImageDraw


def footprint(row):
    return np.concatenate([row['points'],*[np.asarray(p) for p in row.get('attachment_bounds',[])]])


def contained(row,asset,*,inset=0.):
    points=footprint(row);b=np.asarray(asset['source_box_half_open'],float)
    if np.any(points<b[:2]+inset) or np.any(points>b[2:]-1-inset):return False
    e=asset.get('native_enclosure')
    if e:
        angle=np.deg2rad(e['rotation_deg']);delta=points-e['center']
        uv=delta@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        radii=np.asarray(e['size'])/2-e['width_px']/2-inset
        if min(radii)<=0 or np.any(np.sum((uv/radii)**2,axis=1)>=1):return False
    return True


def resolve(rows,assets):
    from .image_edit import cleanup_mask
    choices=[];receipts=[];claimed=set();units=[]
    for asset in assets:
        request=asset.get('edit_request',{});meta=asset.get('edit',{})
        if (asset.get('role')!='icon' or not asset.get('native_enclosure') or
                not request.get('source_pixels_policy') or meta.get('separate_foreground_parts') or
                meta.get('preserved_native_frames')):continue
        context=asset.get('edit_context_half_open',asset['source_box_half_open'])
        size=tuple(int(round(context[i+2]-context[i])) for i in (0,1))
        mask=np.asarray(cleanup_mask(size,request));covered=[]
        for row in rows:
            if row['layer']!='arrows' or row['id'] in claimed:continue
            if any(h not in (None,'none') for h in row.get('heads',[])):continue
            width=float(row['style'].get('width_px') or 1)
            if not contained(row,asset,inset=width/2+1):continue
            owned={**asset,'source_box_half_open':meta.get('owned_box',asset['source_box_half_open'])}
            if not contained(row,owned,inset=width/2+1):continue
            native=Image.new('L',size,0);draw=ImageDraw.Draw(native)
            for path in row.get('paths',[row['points']]):
                pts=np.asarray(path)-np.array(context[:2])
                draw.line([tuple(p) for p in pts],fill=255,width=max(1,math.ceil(width+4)))
            for path in row.get('attachment_bounds',[]):
                pts=np.asarray(path)-np.array(context[:2]);draw.polygon([tuple(p) for p in pts],fill=255)
            if np.any((np.asarray(native)>0)&(mask>0)):continue
            covered.append(row['id']);claimed.add(row['id'])
        if not covered:continue
        identity='source_image_interior/'+asset['id']
        choices.append({'source_object_id':identity,'source_kind':'image_content','asset_id':asset['id'],
            'candidate_ids':covered,'keep_ids':[],
            'reason':'Source-locked image preserves this measured enclosure interior outside all possible cleanup masks'})
        receipts.extend({'candidate_id':k,'classification':'source_covered_image_interior','source_kind':'image_content',
            'source_object_id':identity,'asset_id':asset['id'],'removed':True} for k in covered)
        units.append({'id':identity,'asset_id':asset['id'],'kind':'image_content','candidate_ids':covered,
            'source_pixels_policy':request['source_pixels_policy'],'cleanup_intersection':False,
            'measured_enclosure':asset['native_enclosure']})
    return {'choices':choices,'receipts':receipts,'claimed_ids':claimed,'source_objects':units,
            'protocol':'source_covered_image_interiors.1','arrow_geometry_changed':False}
