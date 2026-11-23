"""Lightweight identity references original observations; never estimates bboxes."""
import copy
from pathlib import Path
import numpy as np
import contracts
import evidence as ev
import group_bbox_identity
import protocol
import target_locator
from transport.common import write_json as write


def build(crop,group,view,context_size,out,config=None):
    config=config or {};out=Path(out)
    boxes=[dict(id=b['id'],bbox=b['bbox']) for b in target_locator.boxes(group,view)]
    annotations=target_locator.configured_draw(crop,boxes,out/'input/target_boxes.png',config)
    write(out/'input/box_display.json',annotations)
    q=dict(task='Identify the visible line/arrow targets indicated by this observation. Geometry is fitted in the next stage.',
        size=list(crop.size),context_size=list(context_size),crop_in_context=ev.crop_in_context(view,context_size),
        group_boxes=boxes,instructions=[
            protocol.SCOPE,
            'CLEAN is unmarked local source pixels. GROUP BOXES are approximate original SAM candidates; CONTEXT locates CLEAN. '
            'Merge duplicate candidates, split mixed candidates or exclude non-linework according to CLEAN. '
            'Do not reconstruct unrelated neighbors. A candidate can be partial and a target may extend beyond its box.',
            'Return separate targets for distinguishable independent arrows. A dense connected band may be one or several '
            'coherent recovery tasks: do not enumerate all edges, match every mask, explain the architecture, '
            'or assume a complete bipartite network. The number of boxes does not determine the number of targets.',
            'source_boxes references only enough existing G IDs to locate a target; do not enumerate overlapping fragments. '
            'Use [] when the observation as a whole locates a dense band or a visible target lacks a useful single candidate. '
            'Do NOT estimate or return a new bbox, endpoints or coordinates for a target.',
            'locator is a short appearance identifier: color, nearby printed label, or nearby objects. '
            'Do not describe route order, direction, estimated coordinates, invisible connections, trunks or branches. '
            'kind is provisional. geometry=mask asks the next stage to inspect pixels for a simple line/curve or block arrow; '
            'geometry=glm requests visible centerline geometry for elbows, dense networks or ambiguous masks. '
            'No mask matching, head types, width, dash measurements or geometry proof is needed in this request.',
            'Only a continuation of the same target or a visibly shared branch outside CLEAN belongs in followups; '
            'give its region in CONTEXT coordinates. Otherwise followups=[]. Return complete with targets=[] for excluded linework; '
            'uncertain means an in-scope target remains unresolved.',
            target_locator.display_instruction(config),
        ],output_schema=dict(status='complete|uncertain',
            targets='[{id,source_boxes:[existing G IDs] or [],kind:straight|curve|polyline|network|block_arrow,geometry:mask|glm,locator:short appearance identifier}]',
            followups='[{context_bbox:[left,top,right,bottom],description:short visible continuation}] or []'))
    q['instructions']=[x for x in q['instructions'] if x]
    if config.get('identity_geometry_mode')=='glm_points':
        import point_identity
        q=point_identity.prompt(crop,boxes,view,context_size,config)
        q['instructions']=[x for x in q['instructions'] if x]
    return q,[('CLEAN: local original',out/'input/clean.png'),
              ('GROUP BOXES: original candidates; not fitted geometry',out/'input/target_boxes.png'),
              ('CONTEXT: whole figure',out/'input/context.png')]


def validate(raw,size,context_size,source_ids,config=None):
    point_mode=(config or {}).get('identity_geometry_mode')=='glm_points'
    if not isinstance(raw,dict) or raw.get('status') not in ('complete','uncertain'):
        raise ValueError('Identity status must be complete or uncertain')
    if not isinstance(raw.get('targets'),list):raise ValueError('targets must be a list')
    targets=[];seen=set()
    for t in raw['targets']:
        tid=t.get('id')
        if not isinstance(tid,str) or not tid or '/' in tid or tid in seen:raise ValueError('Unique target IDs required')
        seen.add(tid);refs=t.get('source_boxes',[])
        if not isinstance(refs,list) or any(x not in source_ids for x in refs):raise ValueError('Unknown original G reference')
        if t.get('kind') not in ('straight','curve','polyline','network','block_arrow'):raise ValueError('Invalid kind')
        if not point_mode and t.get('geometry') not in ('mask','glm'):raise ValueError('Invalid geometry route')
        if not isinstance(t.get('locator'),str) or not t['locator'].strip():raise ValueError('A short visual identifier is required')
        # Even an unsolicited model bbox can never replace original evidence.
        targets.append({k:copy.deepcopy(t[k]) for k in ('id','kind','locator')})
        targets[-1]['geometry']='glm' if point_mode else t['geometry']
        targets[-1]['source_boxes']=list(dict.fromkeys(refs))
    followups=raw.get('followups',[])
    if not isinstance(followups,list):raise ValueError('followups must be a list')
    for f in followups:
        contracts.bbox(f.get('context_bbox'),context_size)
        if not isinstance(f.get('description'),str) or not f['description'].strip():raise ValueError('Describe visible continuation')
    return dict(status=raw['status'],targets=targets,followups=copy.deepcopy(followups))


def resolve(raw,labels,crop,proposals,candidates,info,out,model,config=None,source_boxes=()):
    internal=copy.deepcopy(raw);origins=[]
    for t in internal['targets']:
        refs=[b for b in source_boxes if b['id'] in t['source_boxes']]
        bounds=np.asarray(ev.bounds([b['bbox'] for b in refs]) if refs else [0,0,*crop.size],float)
        bounds=np.clip(bounds,[0,0,0,0],[crop.width,crop.height,crop.width,crop.height])
        t['bbox']=bounds.tolist()
        t['bbox_source']='original_candidate_union' if refs else 'original_observation'
        origins.append(dict(target=t['id'],source_boxes=t['source_boxes'],bbox=t['bbox'],
                            source='original_candidate_union' if refs else 'original_observation',model_estimated=False))
    write(Path(out)/'target_reference_extents.json',origins)
    if (config or {}).get('identity_geometry_mode')=='glm_points':
        import point_identity
        answer,calls,issues=point_identity.deferred_routes(internal,labels,source_boxes,out)
    else:
        answer,calls,issues=group_bbox_identity.resolve(internal,labels,crop,proposals,candidates,info,out,model,config,source_boxes)
    answer['identity_targets']=copy.deepcopy(raw['targets'])
    extent_sources={t['id']:t['bbox_source'] for t in internal['targets']}
    for r in answer['routes']:r['target_bbox_source']=extent_sources[r['id']]
    return answer,calls,issues
