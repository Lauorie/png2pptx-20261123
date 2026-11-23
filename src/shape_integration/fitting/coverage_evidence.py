"""Carry unresolved source proposals into the existing local repair call.

These are fallible locations, not inferred part counts or geometry. Coverage
already paid for the evidence; dropping it at the repair boundary makes a
compound target fall back to an unaided whole-image count.
"""
from copy import deepcopy


def repair_context(context, repair):
    """Carry the discovery, its search scope and its evidence as one repair input."""
    from .observation import sanitize_feedback
    result=deepcopy(context)
    for source,target in [('source_anchors','coverage_source_anchors'),
                          ('source_extent_original','coverage_source_extent_original')]:
        if repair.get(source):result[target]=deepcopy(repair[source])
    requests=[]
    for issue in repair.get('issues',[]):
        request={'kind':issue['kind']}
        for key in ('evidence','action'):
            if value:=sanitize_feedback(issue.get(key,'')):request[key]=value
        if len(request)>1:requests.append(request)
    if requests:result['coverage_source_requests']=requests
    return result


def bind(repairs, existing, evidence, decisions):
    boxes={row['id']:row['bbox_2d'] for row in existing}
    pending={row['sam_id'] for row in decisions if row['action']=='uncertain'}
    result=deepcopy(repairs)
    for repair in result:
        b=repair.get('source_extent_original',boxes[repair['object_id']]);anchors=[]
        for row in evidence.get('shown',[]):
            if row['id'] not in pending:continue
            q=row['bbox_original'];x,y=(q[0]+q[2])/2,(q[1]+q[3])/2
            if not (b[0]<=x<=b[2] and b[1]<=y<=b[3]):continue
            anchors.append({'id':row['id'],'bbox_original':list(q),
                'origin':'unresolved_fitting_coverage_proposal',
                'association':'inside_target_neighborhood; ownership_unverified'})
        if anchors:repair['source_anchors']=anchors
    return result


def local_anchors(job):
    anchors=job.get('input_context',{}).get('coverage_source_anchors',[])
    if not anchors:return []
    from .frames import FitFrame
    from .coordinates import coordinate_mode
    frame=FitFrame.from_metadata(job['frame']);w,h=job['frame']['actual_crop_input_size_px']
    result=[]
    for row in anchors:
        b=row['bbox_original'];q=frame.to_local(b[:2])+frame.to_local(b[2:])
        if q[2]<0 or q[3]<0 or q[0]>w-1 or q[1]>h-1:continue
        clipped=not (0<=q[0]<=q[2]<=w-1 and 0<=q[1]<=q[3]<=h-1)
        values=([v*1000/((w,h)[i%2]-1) for i,v in enumerate(q)]
                if coordinate_mode(job)=='norm1000' else q)
        result.append({'id':row['id'],'bbox_2d':[round(v,3) for v in values],
            'bbox_local_px':q,'source_extent_clipped':clipped,'origin':row['origin']})
    return result


def augment_prompt(prompt,job):
    requests=job.get('input_context',{}).get('coverage_source_requests',[])
    if requests:
        prompt['coverage_requested_corrections']={
            'requests':deepcopy(requests),
            'rule':'These requests come from the later whole-source coverage check. '
                   'Resolve every request against CLEAN, including members outside the old subgroup. '
                   'Coverage itself is fallible: these are hypotheses, not mandatory part counts. '
                   'If CLEAN disproves a request, explicitly dismiss it rather than inventing geometry. '
                   'Source visibility alone is insufficient: rules.scope still decides eligibility. '
                   'Dismiss or delegate an out-of-scope request, including arrowheads and their shafts '
                   'even when the request calls them intrinsic icon strokes. An earlier shape copy '
                   'does not establish ownership or authorize another out-of-scope copy. '
                   'Earlier target descriptions and counts are fallible, not exclusion rules. '
                   'Recover the complete requested source-visible object and preserve supported parts. '
                   'A peer ID, label, containment or bounding-box overlap alone cannot transfer ownership '
                   'of a distinct contour. If the supplied evidence cannot resolve ownership, report '
                   'unresolved instead of deleting that contour or declaring the request satisfied.'}
    extent=job.get('input_context',{}).get('coverage_source_extent_original')
    if extent:
        from .frames import FitFrame
        from .coordinates import coordinate_mode
        frame=FitFrame.from_metadata(job['frame']);w,h=job['frame']['actual_crop_input_size_px']
        q=frame.to_local(extent[:2])+frame.to_local(extent[2:])
        if coordinate_mode(job)=='norm1000':q=[v*1000/((w,h)[i%2]-1) for i,v in enumerate(q)]
        prompt['coverage_source_scope']={'bbox_2d':q,
            'rule':'Coverage observed source-visible members beyond the previous localization. '
                   'Inspect this complete physical object in CLEAN, including the outlying members. '
                   'Prior descriptions, counts and localization boxes are fallible and must not clip the object. '
                   'This is an observation scope, not a contour or permission to include neighbouring independent objects.'}
    rows=local_anchors(job)
    if rows:
        prompt['coverage_source_anchors']={
            'candidates':[{k:v for k,v in row.items() if k!='bbox_local_px'} for row in rows],
            'coordinate_mode':job.get('coordinate_mode','pixel'),
            'rule':'These are the SAME unresolved source proposals shown to fitting coverage, '
                   'also labeled in teal on the target locator. Their boxes are fallible search hints, '
                   'not contours, confirmed owners, missing-part counts or a required uniform layout. '
                   'Match each against CLEAN and actual fitted parts. Several proposals can refer '
                   'to one source part. Inspect the entire target beyond them. Recover only source-visible '
                   'missing content, and correct displaced existing parts instead of adding duplicates.'}
    return prompt


def annotate(image,job):
    from PIL import ImageDraw,ImageFont
    from .views import FONT
    rows=local_anchors(job)
    if not rows:return
    draw=ImageDraw.Draw(image);edge=max(image.size)
    font=ImageFont.truetype(FONT,max(12,edge//85))
    for row in rows:
        q=row['bbox_local_px']
        draw.rectangle(q,outline='#00836b',width=max(1,edge//700))
        draw.text((max(0,q[0]),max(0,q[1]-edge//65)),row['id'],font=font,
                  fill='#00836b',stroke_width=1,stroke_fill='white')
