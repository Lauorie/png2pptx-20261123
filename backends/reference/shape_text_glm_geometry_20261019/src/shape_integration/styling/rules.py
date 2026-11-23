"""Pixel evidence proposes styles; its scores are routing diagnostics, not truth."""
import numpy as np
from .geometry import footprint, raster, crop_box, bounds as geometric_bounds
from ..fitting.paths import sample_commands
from .gradient import rgb, hexcolor, estimate as gradient_estimate

def poly(part): return np.array(sample_commands(part['commands'], 96), float)
def bounds(parts, size, padding=12): return crop_box(parts,size,padding=0,minimum=padding)

def profile_visibility(coords, image_size, excluded):
    valid=(coords[:,:,0]>=0)&(coords[:,:,0]<=image_size[0]-1)&(coords[:,:,1]>=0)&(coords[:,:,1]<=image_size[1]-1)
    if not excluded.is_empty:
        lo=np.floor(coords.min(axis=(0,1))).astype(int);hi=np.ceil(coords.max(axis=(0,1))).astype(int)+1
        lo=np.maximum(lo,0);hi=np.minimum(hi,image_size)
        if np.all(hi>lo):
            mask=raster(excluded,[*lo.tolist(),*hi.tolist()])
            x=np.clip(np.rint(coords[:,:,0]).astype(int)-lo[0],0,hi[0]-lo[0]-1)
            y=np.clip(np.rint(coords[:,:,1]).astype(int)-lo[1],0,hi[1]-lo[1]-1)
            valid &= ~mask[y,x]
    return valid.all(axis=1)

import cv2
from scipy.ndimage import map_coordinates, median_filter

def palette(pixels, limit=8):
    if len(pixels)==0: return []
    pixels=np.asarray(pixels,dtype=np.uint8).reshape(-1,3)
    # Quantized buckets tolerate antialiasing; representative RGB is an observed median.
    bins=(pixels.astype(np.int32)//12)
    codes=bins[:,0]*1024+bins[:,1]*32+bins[:,2]
    vals,counts=np.unique(codes,return_counts=True); order=np.argsort(-counts)
    out=[]
    for ix in order:
        c=np.median(pixels[codes==vals[ix]],axis=0)
        if any(np.linalg.norm(c-rgb(x['color']))<18 for x in out): continue
        out.append({'color':hexcolor(c),'fraction':round(float(counts[ix]/len(pixels)),4)})
        if len(out)>=limit: break
    return out


def ranked_source_colors(image, parts, exclusions, limit=15):
    """Count real original RGB pixels once across the target regions.

    The rule estimator's interpolated profiles and proposed answers are never
    candidates. Overlapping target masks do not multiply a pixel's vote.
    """
    if not 1 <= limit <= 15: raise ValueError('At most 15 source color candidates')
    box=bounds(list(parts.values()),image.size,7)
    mask=np.zeros((box[3]-box[1],box[2]-box[0]),bool)
    for key,part in parts.items():
        geometry,_=footprint(part)
        region=geometry.buffer(7).difference(exclusions(key))
        mask |= raster(region,box)
    pixels=np.asarray(image)[box[1]:box[3],box[0]:box[2]][mask]
    if not len(pixels): return {'colors':[], 'pixel_count':0,'sample_box':box}
    colors,counts=np.unique(pixels.reshape(-1,3),axis=0,return_counts=True)
    order=sorted(range(len(counts)),key=lambda i:(-int(counts[i]),tuple(colors[i])))[:limit]
    return {'colors':[{'color':hexcolor(colors[i]),'pixel_count':int(counts[i])} for i in order],
            'pixel_count':int(len(pixels)),'unique_colors':int(len(colors)),
            'sample_box':box,'method':'exact_original_rgb_counts_union_of_visible_target_regions'}

def contour_samples(part):
    q=poly(part); delta=np.diff(q,axis=0); lengths=np.linalg.norm(delta,axis=1)
    keep=np.r_[True,lengths>1e-6];q=q[keep]
    if len(q)<2: return np.empty((0,2)),np.empty((0,2)),1.
    lengths=np.linalg.norm(np.diff(q,axis=0),axis=1);cum=np.r_[0,np.cumsum(lengths)]
    n=min(1400,max(40,int(cum[-1]))); t=np.linspace(0,cum[-1],n,endpoint=False)
    ind=np.clip(np.searchsorted(cum,t,side='right')-1,0,len(lengths)-1)
    a=(t-cum[ind])/lengths[ind];pts=q[ind]+a[:,None]*(q[ind+1]-q[ind])
    tangent=(q[ind+1]-q[ind])/lengths[ind,None]; normals=np.c_[-tangent[:,1],tangent[:,0]]
    return pts,normals,float(cum[-1]/n)

def estimate(image,part,excluded,options):
    arr=np.asarray(image); w,h=image.size; box=bounds([part],image.size,12);x0,y0,x1,y1=box
    roi=arr[y0:y1,x0:x1]; mask=np.zeros(roi.shape[:2],np.uint8)
    closed=part['commands'][-1][0]=='Z'
    q=poly(part)-[x0,y0]
    geometry, geometry_issue = footprint(part)
    if closed and geometry.area > 0: mask=raster(geometry,box).astype(np.uint8)
    visible=~raster(excluded,box)
    interior=cv2.erode(mask,np.ones((7,7),np.uint8))
    if interior.sum()<8: interior=cv2.erode(mask,np.ones((3,3),np.uint8))
    interior=(interior>0)&visible
    fp=palette(roi[interior])
    if closed and not fp: fp=palette(roi[(mask>0)&visible])
    fill=fp[0]['color'] if fp else None
    pts,normals,spacing=contour_samples(part)
    offsets=np.arange(-7.,7.01,.5)
    coords=pts[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    profiles=np.stack([map_coordinates(arr[:,:,k].astype(float),[coords[:,:,1],coords[:,:,0]],order=1,mode='nearest') for k in range(3)],axis=-1)
    valid=profile_visibility(coords,image.size,excluded)
    lum=profiles.mean(axis=-1)
    # A stroke is a narrow color band distinct from BOTH sides of an edge.
    ends=np.minimum(np.median(lum[:,:4],axis=1),np.median(lum[:,-4:],axis=1))
    idx=np.argmin(lum[:,6:-6],axis=1)+6
    dark=profiles[np.arange(len(pts)),idx]
    contrast=ends-dark.mean(axis=1)
    contrast[~valid]=0
    cp=palette(dark[contrast>14])
    coverage=float(np.mean(contrast[valid]>14)) if valid.any() else 0.
    stroke=cp[0]['color'] if cp and coverage>.13 else None
    if not closed:
        # Open line on a colored background can itself be light/colored.
        bg=(np.median(profiles[:,:3],axis=1)+np.median(profiles[:,-3:],axis=1))/2
        dist=np.linalg.norm(profiles-bg[:,None,:],axis=2)
        ix=np.argmax(dist[:,5:-5],axis=1)+5
        picked=profiles[np.arange(len(pts)),ix]
        cp=palette(picked[(np.max(dist[:,5:-5],axis=1)>20)&valid])
        stroke=cp[0]['color'] if cp else None
        coverage=float(np.mean(np.max(dist[valid,5:-5],axis=1)>20)) if valid.any() else 0.
    widths=[]; support=np.zeros(len(pts),bool)
    if stroke:
        dist=np.linalg.norm(profiles-rgb(stroke),axis=2)
        hit=dist<max(32, .25*np.linalg.norm(rgb(stroke)-np.array([255,255,255])))
        for i,row in enumerate(hit):
            if not valid[i]: continue
            runs=np.where(np.diff(np.r_[False,row,False].astype(int)))[0].reshape(-1,2)
            runs=[(a,b) for a,b in runs if a>0 and b<len(offsets)]
            if runs:
                a,b=min(runs,key=lambda ab:abs((offsets[ab[0]]+offsets[ab[1]-1])/2))
                if abs((offsets[a]+offsets[b-1])/2)<5:
                    widths.append((b-a)*.5);support[i]=True
    width=round(float(np.median(widths)),2) if widths else 1.5
    width=float(np.clip(width,.5,12))
    sm=median_filter(support,size=3,mode='wrap' if closed else 'nearest')
    runs=np.where(np.diff(np.r_[False,sm,False].astype(int)))[0].reshape(-1,2)
    on=np.array([b-a for a,b in runs if valid[max(0,a-1):min(len(valid),b+1)].all()])*spacing
    gaps=np.array([runs[i+1,0]-runs[i,1] for i in range(len(runs)-1) if valid[runs[i,0]:runs[i+1,1]].all()])*spacing
    dashed=bool(len(on)>=4 and len(gaps)>=3 and .15<sm.mean()<.88 and
                np.median(gaps)>1.5 and np.std(on)/max(np.mean(on),1)<.9)
    dash='dot' if dashed and np.median(on)<width*2 else 'dash' if dashed else 'solid'
    fill_purity=fp[0]['fraction'] if fp else 0.
    bg=palette(roi[(cv2.dilate(mask,np.ones((13,13),np.uint8))>mask)&visible]) if closed else []
    # White inside a white scene is visually indistinguishable from transparent.
    if fill and np.min(rgb(fill))>248 and (not bg or np.min(rgb(bg[0]['color']))>248): fill=None
    colors=[]
    for entry in fp+cp+palette(profiles[valid].reshape(-1,3),10):
        c=entry['color']
        if not any(np.linalg.norm(rgb(c)-rgb(v))<12 for v in colors): colors.append(c)
    # V1 audit found the deduplication step could remove the exact rule color,
    # making "keep the rule" conflict with the choice schema. Pin both observed
    # rule colors into the palette before truncation. V1 frozen outputs stay intact.
    pinned=[c for c in [fill,stroke] if c]
    colors=list(dict.fromkeys(pinned+colors))[:20]
    reasons=[]
    if geometry_issue: reasons.append(geometry_issue)
    if valid.mean()<.5: reasons.append('occlusion_or_boundary_sampling_insufficient')
    if closed and not fp: reasons.append('no_visible_interior')
    if closed and fill is None and fp: reasons.append('white_or_unfilled_ambiguous')
    if closed and fill_purity<.62: reasons.append('heterogeneous_interior')
    if closed and min(x1-x0,y1-y0)<28: reasons.append('small_interior')
    if stroke and coverage<.7 and not dashed: reasons.append('partial_boundary_support')
    if stroke and (not widths or np.std(widths)>max(1.5,width*.7)): reasons.append('width_ambiguous')
    if not stroke: reasons.append('no_separate_stroke_evidence')
    if dashed: reasons.append('dash_needs_visual_check')
    yy,xx=np.where(interior)
    grad,grad_audit=gradient_estimate(part,np.c_[xx+x0,yy+y0],roi[interior],options) if closed else (None,{})
    if grad: reasons.append('gradient_needs_visual_check')
    fills=[None]+colors if closed else [None]
    if grad: fills.append(grad)
    return {'candidates':{'fills':fills,'strokes':[None]+colors},
            'gradient_evidence':grad_audit,
            'style':{'fill':fill if closed else None,'stroke':stroke,'width_px':width if stroke else 0,'dash':dash},
            'closed':closed,'palette':colors,'fill_candidates':fp,'stroke_candidates':cp,
            'evidence':{'fill_purity':fill_purity,'stroke_contrast_coverage':round(coverage,3),
                        'stroke_support':round(float(sm.mean()),3),'width_samples':len(widths),
                        'dash_on_px':round(float(np.median(on)),2) if len(on) else None,
                        'dash_off_px':round(float(np.median(gaps)),2) if len(gaps) else None},
            'needs_model':bool(reasons),'reasons':reasons,'crop_original_xyxy':box}
