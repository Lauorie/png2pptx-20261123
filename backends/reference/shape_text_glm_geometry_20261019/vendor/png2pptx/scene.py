"""Persistent inventory with explicit identities, observations, fits and audit history."""
import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .contracts import check_refinement, check_audit
from .fitting import fit_object
from .frames import scale_point


def now():return datetime.now(timezone.utc).isoformat()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def new_scene(image,candidates):
    return {'schema_version':'png2pptx.scene.1','image':image,'coordinate_frame':'original image pixel centers; Pillow half-pixel resize mapping',
            'scope':'M1 candidate-seeded representative shape sample; not full-image reconstruction',
            'objects':{c['id']:{**c,'category':'shape','status':'candidate_unselected','selected':False,'history':[]} for c in candidates},
            'points':{},'paths':{},'issues':[],'crop_plan':None,'audit':None,
            'pending_categories':{'images':'not_started','arrows':'not_started','text':'not_started'},
            'created_utc':now()}


def prompt_candidates(scene):
    return [{'id':o['id'],'label':o['label'],'proposed_type':o['proposed_type'],'bbox_global_pixels':o['bbox_global_pixels'],
             'status':o['status'],'fitted_bbox_global_pixels':o.get('fitted_bbox_global_pixels')} for o in scene['objects'].values()]


def apply_refinement(scene,object_id,raw,crop,provenance):
    # Compute and validate before changing live state. A failed fit cannot leave half an object.
    check_refinement(raw,object_id,crop.input_size,crop.global_size)
    obj=copy.deepcopy(scene['objects'][object_id]);new_points={};new_paths={}
    event={'utc':now(),'decision':raw['decision'],'provenance':provenance,'reason':raw['reason'],'uncertain':raw['uncertain']}
    if raw['decision']=='refined':
        source_size=scene['image']['source_size'];global_size=scene['image']['input_size']
        global_points={p['id']:crop.to_global(p['xy']) for p in raw['points']}
        source_points={pid:scale_point(xy,global_size,source_size) for pid,xy in global_points.items()}
        fit=fit_object(raw,source_points)
        revision=len(obj['history'])+1;prefix=f'{object_id}/r{revision}/'
        for p in raw['points']:
            pid=prefix+p['id'];new_points[pid]={'id':pid,'object_id':object_id,'local_id':p['id'],
                'xy_original':source_points[p['id']],'xy_global_input':global_points[p['id']],'xy_crop_input':p['xy'],'provenance':provenance}
        for part in fit['parts']:
            part['global_point_ids']=[prefix+p for p in part['observed_point_ids']]
            path_id=prefix+part['id'];new_paths[path_id]={'id':path_id,'object_id':object_id,'part_id':part['id'],
                'point_ids':part['global_point_ids'],'commands_original_pixels':part['commands']}
        obj.update(status='refined_unverified',shape_type=raw['shape_type'],fit=fit,
                   active_point_ids=list(new_points),active_path_ids=list(new_paths),
                   fitted_bbox_global_pixels=scale_point(fit['bbox_2d'][:2],source_size,global_size)+scale_point(fit['bbox_2d'][2:],source_size,global_size))
    else:
        obj['status']={'reject':'rejected','unresolved':'unresolved','needs_context':'needs_context'}[raw['decision']]
    obj['history'].append(event)
    scene['objects'][object_id]=obj;scene['points'].update(new_points);scene['paths'].update(new_paths)


def apply_audit(scene,raw):
    refined=[o['id'] for o in scene['objects'].values() if o['status']=='refined_unverified']
    check_audit(raw,list(scene['objects']),refined,scene['image']['input_size'])
    updated=copy.deepcopy(scene['objects']);issues=[]
    for review in raw['object_reviews']:
        o=updated[review['id']]
        o['status']={'accept':'accepted_by_model','revise':'needs_revision','reject':'rejected_by_model','duplicate':'duplicate'}[review['decision']]
        o['duplicate_of']=review['duplicate_of'];o['history'].append({'utc':now(),'stage':'shape_audit',**review})
        if review['decision']!='accept':issues.append({'type':'object_review',**review})
    # Candidate duplicate links remain explicit issues unless adjudicated as an object review.
    # This prevents overwriting accepted fits when a global response contains inconsistent opinions.
    for pair in raw['duplicate_candidates']:issues.append({'type':'duplicate_candidate_proposal','status':'pending_adjudication',**pair})
    for region in raw['missing_regions']:issues.append({'type':'missing_shape_region','status':'pending_model_selected_refinement',**region})
    scene['objects']=updated;scene['issues'].extend(issues);scene['audit']=raw


def summary(scene):
    objects=list(scene['objects'].values())
    return {'case':scene['image']['id'],'candidates':len(objects),'selected':sum(o['selected'] for o in objects),
            'fitted_objects':sum('fit' in o for o in objects),'fitted_parts':sum(len(o.get('fit',{}).get('parts',[])) for o in objects),
            'observed_points':len(scene['points']),'status_counts':dict(Counter(o['status'] for o in objects)),
            'audit_completed':scene['audit'] is not None,'missing_regions':sum(x['type']=='missing_shape_region' for x in scene['issues']),
            'duplicate_proposals':sum(x['type']=='duplicate_candidate_proposal' for x in scene['issues'])}
