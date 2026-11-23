"""Pixel measurements on fixed arrow geometry. No model requests or identity edits."""
import numpy as np
from scipy.ndimage import map_coordinates,median_filter
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion


def points(row):return np.asarray([row['start'],*row.get('via',[]),row['end']],float)


def sample(row,step=.5):
    p=points(row);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-6]]
    d=np.diff(p,axis=0);length=np.linalg.norm(d,axis=1);cum=np.r_[0,np.cumsum(length)]
    s=np.linspace(0,cum[-1],max(2,int(cum[-1]/step)+1));index=np.minimum(np.searchsorted(cum,s,side='right')-1,len(d)-1)
    direction=d[index]/length[index,None];q=p[index]+((s-cum[index])/length[index])[:,None]*d[index]
    return s,q,direction


def pixels(image,q,order=1):
    if image.ndim==2:return map_coordinates(image.astype(float,copy=False),[q[...,1],q[...,0]],order=order,mode='constant',cval=0)
    return np.stack([map_coordinates(image[...,k].astype(float),[q[...,1],q[...,0]],order=order,mode='nearest') for k in range(3)],axis=-1)


def runs(binary):return np.where(np.diff(np.r_[False,binary,False].astype(int)))[0].reshape(-1,2)


def color_mode(colors):
    bins=np.floor(np.asarray(colors)/12).astype(int)
    _,inverse,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
    return np.median(np.asarray(colors)[inverse==np.argmax(counts)],axis=0)


def filled(image,mask,row):
    """Fill color inside the confirmed contour; no shaft-width proxy."""
    polygon=Image.new('L',(image.shape[1],image.shape[0]))
    ImageDraw.Draw(polygon).polygon([tuple(p) for p in points(row)],fill=255)
    interior=np.asarray(polygon)>0
    if mask is not None and mask.any():interior &= mask
    eroded=binary_erosion(interior)
    samples=image[eroded if eroded.any() else interior]
    if not len(samples):raise ValueError('No fill-color pixel evidence; do not default to black')
    color=color_mode(samples)
    return dict(id=row['id'],representation='filled_contour',
                color='#'+''.join(f'{int(round(c)):02x}' for c in color),
                width_px=None,border_width_px=0,start=None,end=None,
                evidence=dict(fill_samples=len(samples),source='CLEAN inside associated mask and contour' if mask is not None else 'CLEAN inside model contour',
                              border_estimated=False))


def dash_fit(s,support,body,width):
    """Measure complete marks; exclude only marks clipped by the sample window."""
    ids=np.flatnonzero(body)
    if len(ids)<4:return dict(type='unknown',preset='solid',reason='Too little visible shaft')
    signal=median_filter(support[ids[0]:ids[-1]+1].astype(int),size=3)>0
    on=runs(signal);step=s[1]-s[0];coverage=float(signal.mean())
    result=dict(type='solid',preset='solid',support_fraction=round(coverage,3))
    complete=on[(on[:,0]>0)&(on[:,1]<len(signal))]
    lengths=(complete[:,1]-complete[:,0])*step;gaps=(on[1:,0]-on[:-1,1])*step
    result['complete_marks']=len(complete)
    if len(lengths)>=2 and len(gaps)>=2:
        mark=float(np.median(lengths));gap=float(np.median(gaps))
        variability=max(float(np.median(abs(lengths-mark))/max(mark,.5)),float(np.median(abs(gaps-gap))/max(gap,.5)))
        if variability<.3 and gap>=.75:
            # Distance from the geometry start into an ink/gap cycle. Only use
            # observed starts, not a run cut off by the head exclusion window.
            starts=s[ids[0]]+(on[on[:,0]>0,0]-.5)*step
            angles=-starts/(mark+gap)*2*np.pi
            phase=float((np.angle(np.mean(np.exp(1j*angles)))%(2*np.pi))/(2*np.pi)*(mark+gap))
            predicted=(s[ids[0]:ids[-1]+1]+phase)%(mark+gap)<mark
            agreement=float(np.mean(predicted==signal))
            # A few similar defects do not establish a repeating pattern for
            # an otherwise solid shaft. Compare against the entire measured
            # signal, including long continuous marks the median ignores.
            if agreement>max(coverage,1-coverage):
                dotted=mark<=1.8*width
                result.update(type='dotted' if dotted else 'dashed',preset='sysDot' if dotted else 'dash',
                    mark_px=mark,gap_px=gap,period_px=mark+gap,phase_px=phase,
                    pattern_source='measured normal-profile support',pattern_agreement=round(agreement,3))
                return result
    if coverage<.85:result.update(type='unknown',reason='Insufficient or inconsistent periodic evidence; solid fallback',
        measured_mark_lengths_px=lengths.tolist(),measured_gaps_px=gaps.tolist())
    return result


def head_profile_body(s,spans,width,row):
    """Trim the first broad head lobe, measured on the associated ink.

    Compare the head and shaft within the same mask-width measurement. Only an endpoint that GLM marked as
    headed is considered. No fixed percentage of arrow length is removed.
    Gaps are left intact: this changes the measurement window, never the mask.
    """
    span=median_filter(np.asarray(spans),size=3)
    visible=span[span>0]
    if len(visible):
        mode=np.argmax(np.bincount(np.rint(visible*2).astype(int)))/2
        width=float(np.median(visible[abs(visible-mode)<=.5]))
    body=np.ones(len(s),bool);regions={}
    for end,reverse in [('start',False),('end',True)]:
        if not row.get(end+'_head'):continue
        values=span[::-1] if reverse else span
        ink=runs(values>0)
        if not len(ink):continue
        a,b=ink[0]
        broad=runs(values[a:b]>1.5*width)
        if not len(broad):continue
        stop=a+int(broad[0,1])
        boundary=float(s[min(stop,len(s)-1)])
        body &= s<s[-1]-boundary if reverse else s>boundary
        regions[end]=dict(extent_px=boundary,mask_shaft_width_px=width,
                          method='first broad ink lobe relative to the same mask shaft profile')
    return body,regions


def apply_line_style(measured,line_style):
    """GLM supplies the category; pixels supply parameters when measurable."""
    if line_style not in ('solid','dashed','dotted','unknown'):
        raise ValueError('Unknown model line_style')
    result=dict(measured,category_source='pixels',parameter_source='measured')
    if line_style=='unknown':
        if measured['type']=='unknown':
            result.update(preset=None,parameter_source='unresolved')
        return result
    preset={'solid':'solid','dashed':'dash','dotted':'sysDot'}[line_style]
    if line_style=='solid':
        return dict(type='solid',preset=preset,category_source='glm',parameter_source='not_applicable',measurement=measured)
    if measured['type'] in ('dashed','dotted'):
        result.update(type=line_style,preset=preset,category_source='glm',measured_type=measured['type'])
    else:
        result=dict(type=line_style,preset=preset,category_source='glm',parameter_source='preset',
                    reason='Model category preserved; pixel period unresolved',measurement=measured)
    return result


def pending(row):
    dash=row.get('style',{}).get('dash',{})
    return dash.get('type')=='unknown' and dash.get('preset') is None


def shaft(image,mask,row,adaptive_heads=False):
    s,q,direction=sample(row);normal=np.column_stack([-direction[:,1],direction[:,0]])
    radius=max(16.,min(128.,.75*s[-1]),2*(row.get('width_estimate') or 2))
    offsets=np.arange(-radius,radius+.01,.5);coords=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    rgb=pixels(image,coords);background=np.median(rgb,axis=1)
    contrast=np.linalg.norm(rgb-background[:,None,:],axis=-1)
    support_mask=pixels(mask,coords,0)>0 if mask is not None else np.ones(contrast.shape,bool)
    proximity=np.exp(-abs(offsets)/max(8.,2*(row.get('width_estimate') or 2)))
    score=np.where(support_mask,contrast*proximity[None,:],-1);peak=np.argmax(score,axis=1)
    candidates=rgb[np.arange(len(q)),peak]
    guard=min(max(8.,4*(row.get('width_estimate') or 2)),s[-1]*.18)
    body=(s>(guard if row.get('start_head') else -1))&(s<(s[-1]-guard if row.get('end_head') else s[-1]+1))
    usable=body & (score.max(axis=1)>15)
    if usable.sum()<4:
        return dict(color='#000000',width_px=None,dash=dict(type='unknown',preset='solid'),
                    evidence=dict(reason='Insufficient shaft pixel evidence')), (s,q,direction)
    color=color_mode(candidates[usable]);delta=color-background;denom=np.maximum(np.sum(delta*delta,axis=1),1)
    # Resampling can overshoot the foreground color. Test color-line agreement
    # BEFORE clamping coverage: otherwise a darker, same-color ringing pixel
    # splits one genuine shaft cross-section into several thin intervals.
    projection=np.sum((rgb-background[:,None,:])*delta[:,None,:],axis=-1)/denom[:,None]
    alpha=np.clip(projection,0,1)
    reconstructed=background[:,None,:]+projection[...,None]*delta[:,None,:]
    residual=np.linalg.norm(rgb-reconstructed,axis=-1)
    hit=(alpha>.35)&(residual<30)&support_mask
    widths=[];centers=[];found=np.zeros(len(q),bool)
    spans=np.zeros(len(q))
    for i in range(len(q)):
        # A large head can dominate a normal profile's background estimate.
        # Measure its extent directly from its SAM evidence, not that color fit.
        occupied=np.flatnonzero(support_mask[i] if mask is not None else hit[i])
        if len(occupied):spans[i]=offsets[occupied[-1]]-offsets[occupied[0]]+.5
    for i in range(len(q)):
        intervals=[(a,b) for a,b in runs(hit[i]) if a>0 and b<len(offsets)]
        if not intervals:continue
        a,b=min(intervals,key=lambda ab:abs((offsets[ab[0]]+offsets[ab[1]-1])/2))
        # Preserve v3's native-resolution measurement behavior: a fixed local
        # pixel cutoff drops the shaft after upsampling even for small offsets.
        # Record the offset as evidence; it does not change geometry.
        found[i]=True
        if body[i]:
            widths.append(float(alpha[i,max(0,a-1):min(b+1,len(offsets))].sum()*.5))
            centers.append(float((offsets[a]+offsets[b-1])/2))
    # A broad filled head can occupy most of the route. The persistent shaft
    # width is the mode of normal cross-sections, not the median head diameter.
    if widths:
        bins=np.rint(np.asarray(widths)*2).astype(int)
        mode=int(np.argmax(np.bincount(bins)))
        width=float(np.median(np.asarray(widths)[abs(bins-mode)<=1]))
    else:width=None
    # SAM can omit periodic strips from a genuinely solid shaft, or bridge a
    # real dash gap. Use its pixels to locate the shaft and measure color/width;
    # verify dash occupancy against CLEAN in that narrow shaft corridor.
    center=float(np.median(centers)) if centers else 0.
    band=abs(offsets-center)<=max(1.,(width or row.get('width_estimate') or 2)/2)
    clean_found=((alpha[:,band]>.35)&(residual[:,band]<30)).mean(axis=1)>.5
    regions={}
    if adaptive_heads and width is not None:
        body,regions=head_profile_body(s,spans,width,row)
    dash=dash_fit(s,clean_found,body,width or 1)
    result=dict(color='#'+''.join(f'{int(round(c)):02x}' for c in color),width_px=round(width,2) if width else None,
        dash=dash,evidence=dict(width_samples=len(widths),
        width_evidence='native local CLEAN cross-sections within associated mask; shaft width mode; unclipped color-line residual',
        dash_evidence='CLEAN color support in the mask-measured shaft corridor',
        head_exclusion=regions if adaptive_heads else {'method':'legacy fixed guard'},
        mask_support_fraction=round(float(found[body].mean()),3) if body.any() else None,
        clean_support_fraction=round(float(clean_found[body].mean()),3) if body.any() else None,
        width_mad_px=round(float(np.median(abs(np.asarray(widths)-width))),2) if width else None,
        median_geometry_offset_px=round(float(np.median(abs(np.asarray(centers)))),2) if centers else None))
    return result,(s,q,direction)


def fit_head(image,row,style,samples,end,templates):
    if not row.get(end+'_head'):return None
    supplied=row.get(end+'_head_type')
    aliases={'open':'arrow','filled_triangle':'triangle','circle':'oval'}
    if supplied:
        supplied=aliases.get(supplied,supplied)
        if supplied not in ('triangle','stealth','arrow','diamond','oval'):
            raise ValueError('Unsupported supplied native head type: '+str(supplied))
        return dict(type=supplied,width='med',length='med',source='supplied_type',needs_review=False)
    if templates is None:raise ValueError('Native head templates required when type is not supplied')
    atlas,meta=templates;width=style['width_px'] or row.get('width_estimate') or 2
    s,q,_=samples;tip=q[0] if end=='start' else q[-1]
    distance=min(max(8.,4*width),s[-1]*.4)
    index=np.argmin(abs(s-(distance if end=='start' else s[-1]-distance)))
    direction=q[index]-tip;direction/=max(np.linalg.norm(direction),1e-9);normal=np.array([-direction[1],direction[0]])
    # Fixed dimensionless field, five fixed med/med shapes. Only align position
    # for measurement; the saved geometry and exported anchor are untouched.
    u,v=np.meshgrid(np.arange(-3.5,6.01,.125),np.arange(-3.5,3.51,.125))
    rgb=pixels(image,tip+width*(u[...,None]*direction+v[...,None]*normal))
    color=np.array([int(style['color'][i:i+2],16) for i in (1,3,5)])
    background=np.median(np.concatenate([rgb[:3].reshape(-1,3),rgb[-3:].reshape(-1,3)]),axis=0)
    delta=color-background;alpha=np.clip(np.sum((rgb-background)*delta,axis=-1)/max(float(delta@delta),1),0,1)
    # Use CLEAN evidence, including pixels missing from the SAM head mask.
    residual=np.linalg.norm(rgb-(background+alpha[...,None]*delta),axis=-1)
    observed=(alpha>.4)&(residual<35);candidates=[]
    for i,kind in enumerate(meta['types']):
        if 'positions' in meta:
            anchor=meta['positions'][row['id']][kind];ref=width
        else:
            anchor=[meta['tip'][0],meta['tip'][1]+i*meta['row_spacing']];ref=meta['reference_width_px']
        best=None
        for du in np.arange(-3,3.01,.5):
            for dv in np.arange(-1,1.01,.5):
                coords=np.stack([anchor[0]-(u-du)*ref,anchor[1]+(v-dv)*ref],axis=-1)
                predicted=pixels(atlas,coords)<.5
                iou=float((observed&predicted).sum()/max((observed|predicted).sum(),1))
                if best is None or iou>best['pixel_iou']:
                    best=dict(type=kind,pixel_iou=iou,geometry_tip_offset_uv_px=[round(float(du*width),2),round(float(dv*width),2)])
        candidates.append(best)
    candidates.sort(key=lambda c:-c['pixel_iou']);best=candidates[0]
    margin=best['pixel_iou']-candidates[1]['pixel_iou']
    best.update(pixel_iou=round(best['pixel_iou'],3),width='med',length='med',source='clean_native_template',
                needs_review=bool(best['pixel_iou']<.6 or margin<.025 or np.linalg.norm(best['geometry_tip_offset_uv_px'])>max(2,width)),
                alternatives=[dict(c,pixel_iou=round(c['pixel_iou'],3)) for c in candidates[1:3]])
    return best


def recover(image,mask,row,templates=None,measured_shaft=None,adaptive_heads=False):
    style,samples=measured_shaft if measured_shaft is not None else shaft(image,mask,row,adaptive_heads)
    if 'line_style' in row:style['dash']=apply_line_style(style['dash'],row['line_style'])
    style.update(id=row['id'],start=fit_head(image,row,style,samples,'start',templates),end=fit_head(image,row,style,samples,'end',templates))
    reasons=[]
    if 'unknown' in row.get('heads',[]):reasons.append('endpoint_head_type_unresolved')
    if style['width_px'] is None:reasons.append('insufficient_shaft_evidence')
    if style['dash']['type']=='unknown':reasons.append('dash_or_occlusion_ambiguous')
    if style['dash'].get('parameter_source')=='preset':reasons.append('dash_parameters_use_preset')
    if (style['evidence'].get('median_geometry_offset_px') or 0)>2:reasons.append('geometry_offset')
    if any(style[e] and style[e]['needs_review'] for e in ['start','end']):reasons.append('head_type_or_anchor_ambiguous')
    style.update(needs_review=bool(reasons),review_reasons=reasons)
    return style
