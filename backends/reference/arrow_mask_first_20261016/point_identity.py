"""Scoped visual identity, followed directly by model-authored geometry.

Mask colors indicate evidence provenance, never confirmed target ownership.
No M/D assignment or numerical mask fitting occurs in this identity mode.
"""
import copy
from pathlib import Path
import numpy as np
from PIL import Image
import evidence as ev
import protocol
import target_locator
from transport.common import write_json as write


def prompt(crop, boxes, view, context_size, config):
    return dict(
        task='Resolve ONLY the complete physical line/arrow targets indicated by this group of G candidates. The next request draws their geometry with points.',
        size=list(crop.size), context_size=list(context_size), crop_in_context=ev.crop_in_context(view,context_size),
        group_boxes=boxes,
        instructions=[
            protocol.SCOPE,
            'CLEAN is the unmarked source. GROUP BOXES locate the original G candidates. '
            'EVIDENCE uses orange for this group primary mask pixels and faint cyan for surrounding evidence. '
            'These are fallible location hints, not fitted paths, ownership truth or already reconstructed objects. '
            'Recover the complete indicated targets from CLEAN even where masks or boxes are partial.',
            'Keep the group task distinct from the surrounding crop. Merge duplicate candidates, split mixed candidates, '
            'or exclude ineligible content. Do not inventory every visible neighbor. Sharing a module or attachment point '
            'does not make all neighboring arrows part of this task. A crop-edge fragment alone is not a new complete target.',
            'Use separate targets for distinguishable independent arrows. A dense band may be one or several coherent '
            'geometry tasks; do not enumerate all edges, match each mask, account for every G, or explain the architecture. '
            'The next request can draw multiple independent paths for one network target.',
            'source_boxes needs only enough existing G IDs to locate the target. [] is allowed when the GROUP collectively '
            'locates one target, not as permission to add unrelated neighbors. locator is a short appearance identifier '
            'using color or printed labels; omit coordinates, turn sequences and inferred connections.',
            'Return identities only: no new bbox, points, mask IDs, mask/GLM choice, head types or style. '
            'kind is provisional; block_arrow means a broad closed arrow shape, including hollow arrows, not a normal shaft with a head.',
            'For a continuation of the SAME target or a true shared branch outside CLEAN, request followup with its '
            'region in CONTEXT coordinates. Do not emit its cropped stub as another target. Unrelated new neighbors '
            'belong to the later whole-figure missing-content stage. Otherwise followups=[]. '
            'complete with targets=[] excludes this group; uncertain means an indicated target remains unresolved.',
            target_locator.display_instruction(config),
        ],
        output_schema=dict(status='complete|uncertain',
            targets='[{id,source_boxes:[existing G IDs] or [],kind:straight|curve|polyline|network|block_arrow,locator:short appearance identifier}]',
            followups='[{context_bbox:[left,top,right,bottom],description:short visible continuation}] or []'))


def evidence_image(crop, group, view, candidates, gid, pool, out):
    """Restore primary evidence without promoting all related masks to primary."""
    out=Path(out);primary=np.zeros((crop.height,crop.width),bool);surrounding=primary.copy()
    primary_ids=[];context_ids=[];sources=[]
    for c in candidates:
        m=c.get('mask')
        if m is None:m=ev.mask(c['path'])
        if c.get('owner')==gid:
            primary|=m;primary_ids.append(c['id'])
        else:
            surrounding|=m;context_ids.append(c['id'])
    # Global pixels remain a locator when the native observation has no primary
    # match. This never crops or constrains model output to those pixels.
    origin='native_primary'
    if not primary.any():
        for g in group:primary|=ev.project(g,view)
        origin='global_candidate_pixels'
    for c in pool or []:
        if c['observation']==gid or ev.gap(c['bbox_global'],view['bbox_global'])>0:continue
        m=ev.project(c,view)
        if m.any():
            surrounding|=m;sources.append(dict(observation=c['observation'],id=c['id']))
    surrounding&=~primary
    pixels=np.asarray(crop).astype(float).copy()
    pixels[surrounding]=.8*pixels[surrounding]+.2*np.array([0,185,215])
    pixels[primary]=.45*pixels[primary]+.55*np.array([230,95,35])
    folder=out/'input';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'identity_evidence.png';Image.fromarray(np.rint(pixels).astype('uint8')).save(path)
    Image.fromarray(primary.astype('uint8')*255).save(folder/'identity_primary_mask.png')
    Image.fromarray(surrounding.astype('uint8')*255).save(folder/'identity_context_mask.png')
    write(out/'identity_evidence.json',dict(primary_ids=primary_ids,context_ids=context_ids,
        primary_origin=origin,primary_pixels=int(primary.sum()),context_pixels=int(surrounding.sum()),
        shared_sources=sources,role='visual location evidence only',per_mask_matching_requested=False))
    return path


def deferred_routes(raw, labels, source_boxes, out):
    """Internal handoff only: identity never selects a numerical geometry source."""
    routes=[]
    for t in raw['targets']:
        refs=[b for b in source_boxes if b['id'] in t['source_boxes']]
        routes.append(dict(id=t['id'],kind='polyline' if t['kind']=='network' else t['kind'],
            identity_kind=t['kind'],geometry='glm',masks=[],proposal=None,nodes=[],network=None,
            sources=list(t['source_boxes']) or list(labels),source_linking='original_candidate_locator',
            locator=t['locator'],target_bbox=t['bbox'],target_bbox_source=t['bbox_source'],
            source_boxes=refs,heads=['unknown','unknown'],line_style='unknown'))
    write(Path(out)/'mask_geometry_calls.json',dict(calls=0,identity_saw_local_masks=True,
        geometry_entry='glm_points_or_shape',mask_assignment_requested=False))
    return dict(nodes=[],routes=routes,followups=copy.deepcopy(raw['followups']),
        excluded=list(labels) if not routes and raw['status']=='complete' else [],
        uncertain=list(labels) if not routes and raw['status']=='uncertain' else [],
        observation_incomplete=raw['status']=='uncertain',identity_targets=copy.deepcopy(raw['targets'])),0,[]


def recover_block(answer,candidates,gid,crop,out,model):
    """A closed arrow uses shape points directly; never publish the locator box."""
    import block_shapes,bbox_geometry_context,mask_vectors
    out=Path(out);route=answer['routes'][0]
    if len(answer['routes'])!=1 or route['kind']!='block_arrow':
        raise ValueError('Independent shape target required')
    folder=out/'shape_geometry'/'S001';related=bbox_geometry_context.related_masks(route,candidates)
    # A mask is an optional outline fallback only when its category and original
    # candidate association agree. Ambiguous neighbors are not silently selected.
    refs={b['id'] for b in route.get('source_boxes',[])}
    matching=[c for c in related if c.get('category')=='block_arrow' and c.get('best_global') in refs]
    support=np.logical_or.reduce([c['mask'] for c in matching]) if matching else None
    x0,y0,x1,y1=route['target_bbox'];rid=gid+':'+route['id']
    seed=dict(id=rid,label=route['id'],kind='block_arrow',start=[x0,y0],via=[[x1,y0],[x1,y1]],end=[x0,y1],
        closed=True,heads=['none','none'],line_style='unknown',node_ids=[],shared_nodes=[],identity_index=1,
        globals=route['sources'],source_masks=[c['id'] for c in matching],source_mask_paths=[c['path'] for c in matching],
        identity_locator=route['locator'],target_bbox=route['target_bbox'],target_bbox_source=route['target_bbox_source'],
        locator_only=True,context_image=str((out/'input/context.png').resolve()))
    record=dict(route=route['id'],requested='glm',path='shape',state='running',request=str(folder/'request'))
    write(out/'routing.json',[record]);write(out/'head_calls.json',dict(calls=0,ordinary_mask_extra_calls=0))
    try:
        row=block_shapes.recover(crop,seed,support,folder,model)
        row['physical_target_id']=rid;row['geometry_owner_id']=rid
        mask_vectors.assign_parents([row],gid)
        record.update(state='completed',drawable_paths=[row['id']]);write(out/'routing.json',[record])
        return [row],{row['id']:support},[],1
    except Exception as exc:
        record.update(state='failed',error=str(exc));write(out/'routing.json',[record])
        write(folder/'error.json',dict(error=str(exc),locator_not_published=True))
        return [],{},[dict(stage='glm_geometry',routes=[route['id']],error=str(exc),fallback='unresolved shape; no locator rectangle published')],1
