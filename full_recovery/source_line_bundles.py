"""Source-pixel identities for simple parallel strokes between filled nodes.

Node pairs only nominate original-image inspection windows. Visible original
stroke profiles define the physical members. Selection requires a unique
native survivor for every member; residual misalignment remains explicit.
No new line or arrow geometry is generated.
"""
from collections import defaultdict

import numpy as np


def simple(row):
    style=row.get('style',{});dash=style.get('dash','solid')
    if isinstance(dash,dict):dash=dash.get('type',dash.get('preset','solid'))
    if (row.get('closed') or 'segment' in row or row.get('attachment_refs')
            or any(h not in (None,'none') for h in row.get('heads',[]))
            or dash!='solid'):
        return None
    points=np.asarray(row['points'],float); span=np.ptp(points,axis=0); axis=int(np.argmax(span))
    if span[axis]<8 or span[1-axis]>max(1.5,.025*span[axis]):return None
    colour=style.get('stroke',style.get('color'))
    if not isinstance(colour,str) or len(colour)!=7:return None
    rgb=np.array([int(colour[i:i+2],16) for i in (1,3,5)])
    if rgb.mean()>100 or np.ptp(rgb)>45:return None
    return axis


def resolve(scene,rows,source):
    rgb=np.asarray(source.convert('RGB')); height,width=rgb.shape[:2]
    groups=defaultdict(list)
    for row in rows:
        pair=row.get('endpoint_nodes'); axis=simple(row)
        if row['layer']=='shape' and axis is not None and pair and len(pair)==2:
            groups[(tuple(sorted(pair)),axis)].append(row)
    choices=[]; receipts=[]; inventories=[]; claimed=set()
    for (pair,axis),shape_rows in sorted(groups.items()):
        if any(k not in scene['parts'] for k in pair):continue
        nodes=[scene['parts'][k] for k in pair]
        if any(n.get('kind') not in ('rectangle','rounded_rectangle') for n in nodes):continue
        boxes=sorted((np.asarray(n['bbox_2d'],float) for n in nodes),key=lambda b:b[axis])
        a,b=boxes; other=1-axis
        low=max(a[other],b[other])+2; high=min(a[other+2],b[other+2])-2
        start=a[axis+2]+3; end=b[axis]-3
        if end-start<8 or high-low<6 or high-low<.45*min(a[other+2]-a[other],b[other+2]-b[other]):continue
        lo=np.zeros(2,int);hi=np.zeros(2,int)
        lo[axis]=int(np.ceil(start));hi[axis]=int(np.floor(end))+1
        lo[other]=int(np.ceil(low));hi[other]=int(np.floor(high))+1
        lo=np.maximum(0,lo);hi=np.minimum([width,height],hi)
        tile=rgb[lo[1]:hi[1],lo[0]:hi[0]]
        if not tile.size:continue
        ink=(tile.mean(2)<100)&(np.ptp(tile.astype(float),axis=2)<45)
        # Canonical array: cross-line positions by along-line samples.
        aligned=ink if axis==0 else ink.T
        if aligned.shape[1]<8:continue
        strength=aligned.mean(1)
        reliable=strength>=.72
        for chunk in np.array_split(aligned,3,axis=1):reliable&=chunk.mean(1)>=.5
        intervals=np.flatnonzero(np.diff(np.r_[False,reliable,False].astype(int))).reshape(-1,2)
        if len(intervals)<2:continue
        if any(stop-begin>max(4,.12*aligned.shape[0]) for begin,stop in intervals):continue
        positions=np.array([np.average(np.arange(begin,stop),weights=strength[begin:stop])+lo[other]
                            for begin,stop in intervals])
        if np.any(np.diff(positions)<3):continue
        eligible=[]
        for row in rows:
            if row['id'] in claimed or simple(row)!=axis:continue
            points=np.asarray(row['points']); ends=points[[0,-1]]
            pmin,pmax=points[:,axis].min(),points[:,axis].max()
            cross=float(np.median(points[:,other]))
            if not low<=cross<=high or pmin>start+2 or pmax<end-2:continue
            # Full paths spanning several node gaps are not local edit units.
            if pmin<a[axis]-2 or pmax>b[axis+2]+2:continue
            if pmax-pmin>1.4*(b[axis]-a[axis+2])+6:continue
            if row.get('endpoint_nodes') and set(row['endpoint_nodes'])!=set(pair):continue
            eligible.append((row,cross))
        if not any(row['layer']=='shape' for row,c in eligible):continue
        kept=[]; assignments=[]
        identity_tolerance=.45*float(np.diff(positions).min())
        for position in positions:
            scored=sorted(((abs(c-position),row['id'],row) for row,c in eligible if row['id'] not in kept),
                          key=lambda v:(v[0],v[1]))
            # Half the actual source separation is the identity boundary.
            # A fit may be displaced yet still unambiguously name this path;
            # requiring an already perfect fit reintroduced overlap failures.
            if not scored or scored[0][0]>identity_tolerance:break
            distance,ident,row=scored[0];kept.append(ident);assignments.append({'candidate_id':ident,
                'source_position_px':float(position),'native_position_error_px':float(distance)})
        if len(kept)!=len(positions):continue
        ids=[row['id'] for row,cross in eligible]
        if not set(ids)-set(kept):continue
        identity='source_pixel_bundle_%04d'%len(inventories)
        error=max(a['native_position_error_px'] for a in assignments)
        inventories.append({'id':identity,'kind':'bundle','count_status':'exact','visible_count':len(positions),
            'endpoint_nodes':list(pair),'axis':axis,'inspection_box_half_open':[*lo.tolist(),*hi.tolist()],
            'source_member_positions_px':positions.tolist(),'source_member_widths_px':(intervals[:,1]-intervals[:,0]).tolist(),
            'selected_members':assignments,'native_alignment_status':'aligned' if error<=1.5 else 'incomplete',
            'maximum_native_position_error_px':error})
        choices.append({'candidate_ids':ids,'keep_ids':kept,'source_object_id':identity,
            'reason':'Original gap profiles establish physical members; every member has a unique existing survivor. Residual native displacement is recorded separately'})
        claimed.update(ids)
        for row,cross in eligible:
            receipts.append({'candidate_id':row['id'],'source_object_id':identity,
                'classification':'source_pixel_bundle','removed':row['id'] not in kept,
                'native_position_px':cross,'nearest_source_position_px':float(positions[np.argmin(abs(positions-cross))])})
    return {'choices':choices,'receipts':receipts,'source_objects':inventories,'claimed_ids':claimed,
            'arrow_geometry_changed':False,'protocol':'source_pixel_parallel_bundle.1'}
