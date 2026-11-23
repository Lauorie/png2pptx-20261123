"""Optional model-selected style evidence; never edits the target geometry."""
import copy,math
import numpy as np
from lib import style


def add_prompt(prompt):
    prompt['instructions'].append(
        'For each non-block route also return shaft_sample: "auto" to use reliable associated mask evidence, '
        '[[x0,y0],[x1,y1]] to select a short straight stretch of its visible SHAFT in CLEAN, or null if no shaft can be measured. '
        'Choose a sample when the mask is head-only, mixed, or misses the body. It may span several dashes and gaps, '
        'but exclude arrowheads, corners, text and neighbors. These two local points only locate style evidence: '
        'they do not change the route, infer a connection, or prescribe width/color. Do not guess an invisible shaft.')
    prompt['output_schema']['routes']=prompt['output_schema']['routes'].replace('line_style}', 'line_style,shaft_sample}')


def validate(value,size):
    if value is None or isinstance(value,str) and value=='auto':return
    if not isinstance(value,list) or len(value)!=2:raise ValueError('shaft_sample is auto, null, or two local [x,y] points')
    for p in value:
        if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p):
            raise ValueError('shaft_sample needs two finite [x,y] points')
        if any(not -.5<=v<n-.5 for v,n in zip(p,size)):raise ValueError('shaft_sample outside CLEAN')
    if value[0]==value[1]:raise ValueError('shaft_sample endpoints must differ')


def unresolved(row,reason):
    return dict(id=row['id'],color='#000000',width_px=None,start=None,end=None,
                dash=dict(type='unknown',preset=None),requested_line_style=row.get('line_style'),
                evidence=dict(reason=reason,shaft_sample=row.get('shaft_sample'),source='glm_shaft_sample'),
                needs_review=True,review_reasons=['insufficient_shaft_evidence'])


def complete_sample_marks(image,sample,width,color):
    """Measure complete ink runs intersecting the selected straight stretch.

    Sample endpoints locate evidence, not dash edges. Extend observation along
    the same axis to the image boundary, retaining only runs that intersect the
    selected interval. No neighboring disjoint mark becomes a new target.
    The semantic dash category is already supplied by GLM.
    """
    a,b=np.asarray(sample,float);length=float(np.linalg.norm(b-a));direction=(b-a)/length
    lo,hi=-np.inf,np.inf
    for origin,delta,size in zip(a,direction,[image.shape[1],image.shape[0]]):
        if abs(delta)<1e-10:continue
        limits=sorted([(-.5-origin)/delta,(size-.5-origin)/delta]);lo=max(lo,limits[0]);hi=min(hi,limits[1])
    s=np.arange(lo+.5,hi,.5);q=a+s[:,None]*direction
    normal=np.array([-direction[1],direction[0]]);radius=max(16.,4*width)
    offsets=np.arange(-radius,radius+.01,.5);rgb=style.pixels(image,q[:,None,:]+normal*offsets[None,:,None])
    background=np.median(rgb,axis=1);foreground=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    delta=foreground-background;denom=np.maximum(np.sum(delta*delta,axis=1),1)
    alpha=np.sum((rgb-background[:,None,:])*delta[:,None,:],axis=-1)/denom[:,None]
    residual=np.linalg.norm(rgb-(background[:,None,:]+alpha[...,None]*delta[:,None,:]),axis=-1)
    band=abs(offsets)<=max(1.,width/2)
    support=((alpha[:,band]>.35)&(residual[:,band]<30)).mean(axis=1)>.5
    marks=[]
    for begin,end in style.runs(support):
        if begin==0 or end==len(s):continue
        start=s[begin]-.25;stop=s[end-1]+.25
        if stop>=0 and start<=length:marks.append([float(start),float(stop)])
    if len(marks)<2:return None
    marks=np.array(marks);lengths=marks[:,1]-marks[:,0];gaps=marks[1:,0]-marks[:-1,1]
    mark=float(np.median(lengths));gap=float(np.median(gaps));period=mark+gap
    phase=float((np.angle(np.mean(np.exp(-2j*np.pi*marks[:,0]/period)))%(2*np.pi))/(2*np.pi)*period)
    return dict(type='dashed',preset='dash',mark_px=mark,gap_px=gap,period_px=period,phase_px=phase,
        complete_marks=len(marks),observed_gaps=len(gaps),measured_mark_lengths_px=lengths.tolist(),
        measured_gaps_px=gaps.tolist(),pattern_source='complete CLEAN ink runs intersecting model shaft sample',
        category_source='glm',parameter_source='measured',sample_boundaries_are_not_dash_boundaries=True)


def recover(image,row):
    sample=row['shaft_sample']
    if sample is None:return unresolved(row,'Model could not identify measurable shaft; geometry retained separately')
    validate(sample,[image.shape[1],image.shape[0]])
    part=copy.deepcopy(row)
    part.update(kind='straight',start=sample[0],end=sample[1],via=[],beziers=[],width_estimate=None,
                heads=['none','none'],start_head=False,end_head=False,start_head_type=None,end_head_type=None)
    part.pop('points',None)
    measured=style.shaft(image,None,part,adaptive_heads=False)
    if measured[0]['width_px'] is None:return unresolved(row,'Selected shaft sample has insufficient pixel evidence')
    result=style.recover(image,None,row,measured_shaft=measured)
    if row.get('line_style') in ('dashed','dotted') and not result['dash'].get('period_px'):
        pattern=complete_sample_marks(image,sample,result['width_px'],result['color'])
        if pattern:
            pattern.update(type=row['line_style'],preset='sysDot' if row['line_style']=='dotted' else 'dash')
            result['dash']=pattern
            result['review_reasons']=[r for r in result['review_reasons'] if r!='dash_parameters_use_preset']
            result['needs_review']=bool(result['review_reasons'])
    result['evidence']['shaft_sample']=copy.deepcopy(sample)
    result['evidence']['source']='glm_selected_shaft_clean_pixels'
    result['evidence']['width_evidence']='CLEAN cross-sections of the model-selected shaft; no head mask restriction'
    result['evidence']['sample_coordinate_frame']='local CLEAN; geometry unchanged'
    # A measured dash phase belongs to the sampled stretch, not to the route's
    # first point. Transfer the offset/order without modifying either geometry.
    if result['dash'].get('period_px') and 'phase_px' in result['dash']:
        s,q,d=style.sample(row);index=int(np.argmin(np.linalg.norm(q-np.asarray(sample[0]),axis=1)))
        forward=float(d[index]@(np.asarray(sample[1])-sample[0]))>=0
        phase=result['dash']['phase_px']
        result['dash']['phase_px']=((phase if forward else result['dash']['mark_px']-phase)-float(s[index]))%result['dash']['period_px']
        result['evidence']['phase_transfer']=dict(route_distance_px=float(s[index]),forward=forward)
    return result
