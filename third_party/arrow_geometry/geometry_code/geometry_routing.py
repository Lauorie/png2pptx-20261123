"""Identity-selected mask fitting or GLM geometry, with one shared model client.

Networks are grouped only by the identity model's network/node references.
No overlap score, residual threshold, or final geometry gate changes its decision.
"""
import copy,traceback
from pathlib import Path
import numpy as np
import mask_vectors,segmented_geometry,visual,evidence as ev,head_semantics
from PIL import Image
from transport.common import write_json as write,read_json as read,sha256 as sha


def measurements(mids,index,proposals):
    """Carry the measurements already seen by identity into geometry unchanged."""
    result=[]
    for mid in mids:
        y,x=np.nonzero(index[mid]['mask'])
        if not len(x):continue
        axes=[]
        for pid,p in proposals.items():
            if mid in p.get('masks',[]):
                axes.append(dict(id=pid,kind=p['kind'],points=[
                    [round(float(a),1),round(float(b),1)] for a,b in [p['start'],*p.get('via',[]),p['end']]]))
        result.append(dict(id=mid,bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],straight_estimates=axes))
    return result


def components(routes):
    pending=list(routes);result=[]
    while pending:
        group=[pending.pop(0)];changed=True
        while changed:
            changed=False
            networks={r.get('network') for r in group if r.get('network')}
            nodes={n for r in group for n in r['nodes']}
            for r in list(pending):
                if (r.get('network') in networks or nodes.intersection(r['nodes'])):
                    group.append(r);pending.remove(r);changed=True
        result.append(group)
    return result


def recover(answer,proposals,candidates,gid,crop,out,model,locks,config=None):
    config=config or {}
    bbox_mode=config.get('identity_input_mode') in ('group_boxes','group_refs')
    union_evidence=config.get('geometry_mask_view')=='union'
    out=Path(out);index={c['id']:c for c in candidates};routes=answer['routes']
    groups=components(routes)
    if bbox_mode and config.get('bbox_geometry_batch')=='independent' and len(groups)>1:
        import bbox_geometry_context
        return bbox_geometry_context.independent(answer,proposals,candidates,gid,crop,out,model,locks,config,groups)
    if config.get('identity_geometry_mode')=='glm_points' and any(r['kind']=='block_arrow' for r in routes):
        import point_identity
        return point_identity.recover_block(answer,candidates,gid,crop,out,model)
    # A network label may name an off-crop continuation. A lone, explicitly
    # mask-selected segment has no second route to jointly solve in this call.
    deferred={r['id'] for g in groups if any(r['geometry']=='glm' for r in g)
              or (len(g)>1 and any(r.get('network') for r in g)) for r in g}
    fast=copy.deepcopy(answer)
    for r in fast['routes']:
        if r['id'] in deferred:r['geometry']='glm'
    with locks['cpu']:
        rows,supports,issues=mask_vectors.physical_rows(fast,proposals,candidates,gid,defer_errors=True)
    failures={i['route'] for i in issues if i.get('fallback')=='deferred_glm'}
    # A numerical failure in a confirmed network defers that whole semantic
    # component, so its incident branches cannot acquire inconsistent junctions.
    deferred.update(r['id'] for g in groups if any(r['id'] in failures for r in g) for r in g)
    rows=[r for r in rows if r['id'].split(':',1)[-1] not in deferred]
    supports={r['id']:supports[r['id']] for r in rows}
    records={r['id']:dict(route=r['id'],requested=r['geometry'],network=r.get('network'),
        path='glm' if r['id'] in deferred else 'mask' if r['geometry']=='mask' else 'legacy_points',
        reason='numerical_mask_failure' if any(x['id'] in failures for g in groups if r in g for x in g)
        else 'model_shared_network' if r.get('network') and r['id'] in deferred else 'model_choice',
        state='pending' if r['id'] in deferred else 'completed') for r in routes}
    calls=0;head_calls=0
    write(out/'routing.json',list(records.values()))
    # Preserve the original observation scope. Batch its confirmed GLM targets
    # without asserting that they share nodes or form one physical network.
    fitting_groups=([[r for r in routes if r['id'] in deferred]] if deferred else []) if bbox_mode else groups
    for group in fitting_groups:
        if not any(r['id'] in deferred for r in group):continue
        calls+=1;folder=out/'geometry'/f'F{calls:03d}';inputs=folder/'input';inputs.mkdir(parents=True,exist_ok=True)
        try:
            targets=[];baseline={};layers=[];mask_by_target={};identity_context=[];optional_evidence={}
            for i,r in enumerate(group,1):
                rid=f'R{i}';mids=r['masks']
                support=np.logical_or.reduce([index[m]['mask'] for m in mids]) if mids else None
                mask_by_target[rid]=support
                layers.append((rid,support if support is not None else np.zeros((crop.height,crop.width),bool)))
                baseline[rid]=dict(id=gid+':'+r['id'],label=r['id'],heads=r['heads'],line_style=r['line_style'],
                    source_masks=mids,source_mask_paths=[index[m]['path'] for m in mids],globals=r['sources'],
                    identity_index=routes.index(r)+1,geometry_request=str(folder/'request'))
                if r.get('source_linking'):baseline[rid]['discovery_source_linking']=r['source_linking']
                # Carry a visual locator, never the old free-form route narrative.
                # In particular, old guessed coordinates must not become measured
                # evidence or be restored by a compatibility fallback.
                locator=r.get('locator','').strip()
                targets.append(dict(id=rid,identity_kind=r['kind'],global_targets=r['sources'],
                    description=locator,network=r.get('network'),
                    mask_measurements=measurements(mids,index,proposals)))
                if config.get('identity_source_mode')=='observation':
                    targets[-1].pop('identity_kind',None)
                    targets[-1]['observation_sources']=targets[-1].pop('global_targets')
                if bbox_mode:
                    # Identity bbox is an approximate locator, not a mask gate.
                    targets[-1]=dict(id=rid,bbox=r['target_bbox'],description=locator)
                    if r.get('target_bbox_source')=='original_observation':
                        targets[-1].pop('bbox')
                        targets[-1]['locator_scope']='whole observation; no original target bbox available'
                    if r.get('source_boxes'):targets[-1]['source_boxes']=r['source_boxes']
                    if config.get('bbox_geometry_mask_evidence'):
                        import bbox_geometry_context
                        related=bbox_geometry_context.related_masks(r,candidates,include_network=union_evidence)
                        optional_evidence.update({c['id']:c for c in related})
                        if not union_evidence:targets[-1]['optional_mask_evidence']=[c['id'] for c in related]
                identity_context.append(dict(target=rid,source_route=r['id'],locator=locator,
                    omitted_description=r.get('description','')))
                records[r['id']].update(state='running',request=str(folder/'request'),target=rid)
            with locks['cpu']:
                if bbox_mode:
                    import target_locator
                    annotations=target_locator.configured_draw(crop,[t for t in targets if 'bbox' in t],inputs/'target_boxes.png',config)
                    write(inputs/'box_display.json',annotations)
                else:visual.masks(crop,layers,inputs/'masks.png')
            with_heads=config.get('geometry_endpoint_heads',False)
            prompt=segmented_geometry.prompt(targets,crop.size,locator_mode='boxes' if bbox_mode else 'masks',with_heads=with_heads)
            if bbox_mode:
                note=target_locator.display_instruction(config)
                if note:prompt['instructions'].append(note)
                if config.get('identity_neutral_handoff'):
                    prompt['instructions'].append('source_boxes retain the original SAM candidate extents that located this target. '
                        'They may be partial or overlap neighbors; neither they nor the identity bbox restrict the visible route. '
                        'Determine all endpoints and bends independently from CLEAN, including portions beyond those boxes.')
                if config.get('identity_input_mode')=='group_refs':
                    prompt['instructions'].append('The target bbox is computed from the original SAM references, not estimated by identity. '
                        'It locates evidence only. Read the actual source pixels to determine the complete visible route; '
                        'nearby boxes, repeated architecture and the locator text do not establish a branch or connection.')
                    if any(t.get('locator_scope') for t in targets):
                        prompt['instructions'].append('A target with locator_scope has no associated original candidate box. '
                            'No target rectangle is drawn for it. Locate the intended ink from CLEAN, CONTEXT and its short appearance identifier; '
                            'never substitute another line merely because it has a visible candidate box.')
            if (out/'input/transform.json').exists():
                with Image.open(out/'input/context.png') as context:context_size=context.size
                prompt['context_size']=list(context_size)
                prompt['crop_in_context']=ev.crop_in_context(read(out/'input/transform.json'),context_size)
                prompt['instructions'].append('crop_in_context gives the exact crop location in CONTEXT pixels. Its magenta rectangle is an annotation; do not guess the image correspondence.')
            if not bbox_mode:
                prompt['instructions'] += ['mask_measurements are in this same CLEAN coordinate frame. Bboxes locate ink, not endpoints or corner positions. Straight estimates may span neighbors or only part of a route. Use useful measurements to locate the visible centerline; CLEAN determines each endpoint and turn.']
            prompt['instructions'] += ['These R target identities are fixed. Draw their shaft centerlines without forcing separate network branches into a single-chain traversal or drawing arrowhead wings as paths.']
            images=[('CLEAN: local pixel coordinates',out/'input/clean.png'),
                    (('TARGET BOXES: confirmed R targets only',inputs/'target_boxes.png') if bbox_mode else
                     ('MASKS: selected target evidence, R labels',inputs/'masks.png')),
                    ('CONTEXT: whole figure',out/'input/context.png')]
            if optional_evidence:
                with locks['cpu']:
                    evidence_layers=[('',np.logical_or.reduce([c['mask'] for c in optional_evidence.values()]))] if union_evidence else [(mid,c['mask']) for mid,c in optional_evidence.items()]
                    visual.masks(crop,evidence_layers,inputs/'optional_masks.png')
                images.append(('OPTIONAL MASK EVIDENCE: nearby measured pixels, not confirmed targets',inputs/'optional_masks.png'))
                if not union_evidence:prompt['optional_masks']=measurements(list(optional_evidence),optional_evidence,{})
                else:write(inputs/'optional_mask_sources.json',measurements(list(optional_evidence),optional_evidence,{}))
                prompt['instructions'].append('OPTIONAL MASK EVIDENCE shows cached local SAM pixels near this target. '
                    'Use useful pixels to locate thin shafts and bends; ignore missing or unrelated fragments. '
                    'The highlight is pixel evidence, not additional targets or a complete fitted route. No mask assignment or explanation is requested. '
                    'Only output the supplied R targets. CLEAN decides the geometry when evidence conflicts.')
            if config.get('identity_source_mode')=='observation' and rows:
                import coverage_visual
                with locks['cpu']:coverage_visual.render(crop,rows,ev.identity_view(crop.size),inputs/'already_recovered.png')
                images.append(('ALREADY RECOVERED in this observation: blue paths; gray is source only',inputs/'already_recovered.png'))
                prompt['already_recovered']=dict(path_ids=[r['id'] for r in rows],geometry='blue paths on the paired image')
                prompt['instructions'].append('The blue ALREADY RECOVERED paths are supplied to locate portions already drawn in this observation. For a remaining-linework target, recover its visible missing paths without copying those blue paths. A crossing does not imply connectivity. These paths are not geometric anchors that may bend other lines.')
            write(folder/'targets.json',targets);write(folder/'baseline_metadata.json',baseline)
            write(folder/'identity_context_not_sent.json',identity_context)
            write(folder/'prompt.json',prompt)
            write(inputs/'images.json',[dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
            write(out/'routing.json',list(records.values()))
            fitted=model.ask(folder/'request',prompt,images,lambda value:segmented_geometry.validate(value,baseline,crop.size,require_heads=with_heads))
            write(folder/'answer.json',fitted)
            with locks['cpu']:
                newrows=segmented_geometry.geometry(fitted,baseline)
                segmented_geometry.draw_points(fitted,out/'input/clean.png',folder/'points.png')
            support_by_id={row['id']:mask_by_target[row['geometry_target_id']] for row in newrows}
            if with_heads:
                import geometry_heads
                head_issues=[];nhead=0
                for row in newrows:geometry_heads.attach(row)
                if config.get('geometry_native_preview',False):
                    try:
                        with locks['cpu']:geometry_heads.render(crop,newrows,folder/'native_preview')
                    except Exception as exc:
                        head_issues.append(dict(stage='geometry_preview',error=str(exc),geometry_retained=True))
                        write(folder/'native_preview_error.json',head_issues[-1])
            elif config.get('separate_style',False):
                head_issues=[];nhead=0
                for row in newrows:
                    row['heads']=['unknown','unknown'];row['head_source']='deferred_to_fixed_geometry_style'
            else:newrows,head_issues,nhead=head_semantics.recover(crop,newrows,support_by_id,folder/'heads',model,cpu_slot=locks['cpu'])
            issues.extend(head_issues);head_calls+=nhead
            for target,r in zip(targets,group):
                owned=[row for row in newrows if row['geometry_target_id']==target['id']]
                for row in owned:
                    refs=list(dict.fromkeys(n for s in row['model_segments'] for n in s['nodes']))
                    # Namespace the actual point references, while retaining
                    # physical target ownership separately from drawable paths.
                    namespace=gid+':'+config.get('geometry_namespace','')+folder.name+':'
                    row['node_ids']=[namespace+str(n) for n in refs]
                    row['shared_nodes']=[namespace+str(n) for n in row['shared_nodes']]
                    for end,h in zip(('start','end'),row['heads']):
                        row[end+'_head']=h not in ('none','unknown');row[end+'_head_type']=h if h not in ('none','unknown') else None
                    supports[row['id']]=mask_by_target[target['id']];rows.append(row)
                records[r['id']]['state']='completed'
                records[r['id']]['drawable_paths']=[row['id'] for row in owned]
                records[r['id']]['head_request']=str(folder/'heads/request')
            write(folder/'geometry_local.json',newrows)
        except Exception as exc:
            write(folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            issues.append(dict(stage='glm_geometry',routes=[r['id'] for r in group],error=str(exc),fallback='unresolved; no fabricated geometry'))
            for r in group:records[r['id']].update(state='failed',error=str(exc))
        write(out/'routing.json',list(records.values()))
    # A style sample remains in the original local frame even when GLM replaces
    # the centerline. It is evidence, not an endpoint constraint or mask gate.
    by_route={gid+':'+r['id']:r for r in routes}
    for row in rows:
        source=by_route[row.get('geometry_owner_id',row['id'])]
        if 'shaft_sample' in source and not config.get('separate_style',False):row['shaft_sample']=copy.deepcopy(source['shaft_sample'])
    rows.sort(key=lambda r:r['identity_index']);mask_vectors.assign_parents(rows,gid)
    write(out/'head_calls.json',dict(calls=head_calls,ordinary_mask_extra_calls=0))
    return rows,supports,issues,calls
