"""Full-inventory experiments. Batching limits request size, never candidate coverage."""
import copy
from . import prompts as base
from .contracts import obj, arr, ident, string, enum, null, bbox_schema, validate, refine_schema, check_refinement, check_plan
from .frames import Crop

VERSION='shape_full.2'
SYSTEM=base.SYSTEM


def round_coordinates(value, key=''):
    """Round coordinate payloads only; scale ratios and stored transforms retain precision."""
    if isinstance(value,dict):return {k:round_coordinates(v,k) for k,v in value.items()}
    if isinstance(value,list):return [round_coordinates(v,key) for v in value]
    coordinate=('bbox' in key or key in ('xy','origin','maximum_xy','minimum_xy','output_min_xy','output_max_xy','offset_xy'))
    if coordinate and isinstance(value,float):return round(value,2)
    return value


def coordinate_prompt(prompt):
    result=round_coordinates(prompt)
    result['coordinate_precision']='Return all point, bbox and crop coordinates as JSON numbers with at most 2 decimal places. Integers are fine; do not pad trailing zeros. Coordinate hints are rounded to 2 decimals. Use integer image/crop dimensions for exact transformations; scale ratios are not coordinates and retain precision.'
    return result


def full_plan(size, targets, inventory, crop_scale=3, crop_max_edge=2048):
    p=base.planning_prompt(size,targets,len(targets))
    p.pop('experiment_sampling')
    p.update(prompt_version=VERSION,stage='full_shape_crop_planning',
        task='Plan a crop for EVERY required target ID in this batch. This is full-inventory processing, not representative sampling.',
        required_target_ids=[c['id'] for c in targets],
        inventory_context=inventory,
        coverage_requirement='Return every required target ID exactly once across regions. No target may be dropped because it is simple, repetitive, hard, small, or similar to another. Other batches process all remaining inventory IDs. Batch size only limits request size, never the number of objects reconstructed. Candidates may be wrong; still give a useful crop so the independent refinement can verify identity. Regions are chosen by you, not a fixed tiling grid.')
    p['crop_rules']=p['crop_rules'].replace('The controller records all other candidates as not selected; do not imply they are absent.',
        'Every other candidate belongs to another batch and will also be processed.')
    if crop_scale is not None:
        p['crop_rules']=p['crop_rules'].replace('resizes each crop to long edge 2048',
            f'upsamples each crop by min({crop_scale:g}, {crop_max_edge}/crop_long_edge) relative to this global image, preserving aspect ratio with integer dimensions. The global input has long edge 2048; a crop has variable dimensions and long edge at most {crop_max_edge}')
        p['crop_scale_relative_to_global_input']=crop_scale
        p['crop_max_long_edge']=crop_max_edge
    return p


def check_full_plan(raw,targets,size):
    check_plan(raw,targets,len(targets),size)
    actual=[i for r in raw['regions'] for i in r['target_ids']]
    if set(actual)!={c['id'] for c in targets}:raise ValueError('Full planning omitted required candidate IDs')
    return raw


def full_refine_schema(ident_value,local_size,global_size,other_ids):
    schema=copy.deepcopy(refine_schema(ident_value,local_size,global_size))
    schema['properties']['decision']['enum']+=['duplicate','represented_by_children']
    ids=enum(other_ids)
    schema['properties']['duplicate_of']={'oneOf':[ids,null()]} if other_ids else null()
    schema['properties']['represented_by']=arr(ids) if other_ids else {'type':'array','maxItems':0}
    schema['required']+=['duplicate_of','represented_by']
    return schema


def check_full_refine(raw,ident_value,crop,other_ids):
    validate(raw,full_refine_schema(ident_value,crop.input_size,crop.global_size,other_ids))
    if len(raw['represented_by'])!=len(set(raw['represented_by'])):raise ValueError('Repeated represented_by identity')
    decision=raw['decision']
    if decision in ('duplicate','represented_by_children'):
        if raw['points'] or raw['parts'] or raw['suggested_global_crop'] is not None:raise ValueError('Identity delegation requires empty own geometry and null crop')
        if decision=='duplicate' and (not raw['duplicate_of'] or raw['represented_by']):raise ValueError('Duplicate needs exactly one existing owner')
        if decision=='represented_by_children' and (raw['duplicate_of'] is not None or not raw['represented_by']):raise ValueError('Delegation needs existing child IDs')
    else:
        if raw['duplicate_of'] is not None:raise ValueError('duplicate_of is only valid for duplicate decision')
        if decision!='refined' and raw['represented_by']:raise ValueError('Unresolved geometry must not claim child coverage')
        stripped={k:v for k,v in raw.items() if k not in ('duplicate_of','represented_by')}
        check_refinement(stripped,ident_value,crop.input_size,crop.global_size)
    return raw


def full_refine(candidate,frame,peers,feedback=None):
    p=base.refine_prompt(candidate,frame)
    other_ids=[c['id'] for c in peers]
    p.update(prompt_version=VERSION,stage='full_single_shape_geometry',other_candidates_in_this_image=peers,
        ownership_rules='Every candidate in this image is processed independently. A container owns its boundary, not all the contents in its bounding box. If distinct existing candidates own visible subparts, do not reconstruct those same subparts again here: list their IDs in represented_by and return only this target\'s own remaining geometry. If a grouping has no own geometry and is completely represented by existing children, use represented_by_children. A compound with no separately enumerated children may return its visible parts in its own point table. Overlap/containment alone does not mean duplicate. Use duplicate only for the SAME visual object and identify its existing owner; do not confuse a container and its contents.',
        observation_rules='Trace actual supplied pixels. Do not replace the contour with a memorized vector template, a familiar symbolic approximation, invented symmetry, or unobserved repeated parts. Adapt on-contour sampling to actual detail. Be concise and return a complete result; if the target cannot be traced reliably, explicitly return unresolved instead of pretending an enclosing box is its geometry.',
        extended_decisions='Normal geometry decisions follow the stated rules. duplicate: empty geometry, duplicate_of names the same object, represented_by empty. represented_by_children: empty geometry, nonempty represented_by, duplicate_of null. refined: own geometry nonempty, duplicate_of null, represented_by may list separate owners of excluded child geometry. Other decisions have null duplicate_of and empty represented_by.',
        output_schema=full_refine_schema(candidate['id'],frame['actual_crop_input_size_px'],frame['global_image_size_px'],other_ids))
    if feedback:p['previous_review_feedback']=feedback
    return p


def local_check_schema(ident_value):
    return obj({'candidate_id':{'const':ident_value},'decision':enum(['accept','revise','reject']),
                'reason':string(),'uncertain':arr(string())})


def local_check(ident_value,frame,shape_type,represented_by,candidate=None,children=None):
    return {'prompt_version':VERSION,'stage':'single_shape_geometry_check',
        'task':'Review ONLY the named target\'s fitted geometry against clean local pixels.',
        'candidate_id':ident_value,'shape_type':shape_type,'frame':frame,'represented_by_other_candidates':represented_by,
        'target_identity':candidate,'excluded_child_inventory':children or [],
        'images':'Image 1 is the clean crop. Image 2 is the same aligned crop with fitted contour and observed point labels. Colored annotations are predictions, not evidence.',
        'scope':base.SCOPE,
        'rules':'Check target identity, contour, type, placement, real corners, curvature and missing visible geometric parts owned by this target. Excluded child candidates are handled separately. Do not check whole-image coverage, other candidates, arrow completeness or OCR. Do not accept a coarse symbolic approximation when the actual contour has unrepresented details. For outlined shapes the fitted curve represents stroke center, so compare its centerline, not both stroke boundaries. A single outline must not become two independently counted inner/outer shapes just to encode width. accept means adequate geometry with only minor deviations; revise means geometry or ownership/scope needs correction; reject means this is not a shape target. Explain visible evidence briefly.',
        'serialization':base.SERIALIZATION,'output_schema':local_check_schema(ident_value)}


def coverage_schema(size,ids):
    return obj({'missing_regions':arr(obj({'bbox_2d':bbox_schema(size),'reason':string()})),
        'duplicate_candidates':arr(obj({'keep':enum(ids),'other':enum(ids),'reason':string()})),
        'coverage_assessment':enum(['no_additional_regions_found','search_regions_proposed','uncertain']),
        'uncertain':arr(string())})


def coverage(size,inventory,round_index):
    return {'prompt_version':VERSION,'stage':'shape_inventory_coverage','round':round_index,
        'task':'Find shape regions absent from the GLOBAL inventory and duplicate identities. Do not re-review the geometry of every known object.',
        'coordinate_system':base.pixel_frame(size),'scope':base.SCOPE,'inventory':inventory,
        'images':'Image 1 is clean global source. Image 2 shows inventory boxes and successful geometry; both have identical global alignment.',
        'rules':'Every listed candidate has already entered full processing. Failed/unresolved/revision-needed known objects are NOT new missing discoveries. A compound bbox alone does not prove its internal shape parts were recovered: use its fitted-part inventory and recorded child owners. Propose model-chosen coherent crop regions containing additional unrepresented shapes, using as many regions as visual density warrants, not a fixed grid. A later independent detection request will enumerate each region. Do not enumerate every missing point or fit curves here. Duplicate pairs must name the same visual object, not overlapping objects or parent/child geometry. No arrow/line completeness, natural-image extraction or OCR checks.',
        'crop_bounds':f'Inclusive pixel centers: 0 <= xmin < xmax <= {size[0]-1}; 0 <= ymin < ymax <= {size[1]-1}.',
        'serialization':base.SERIALIZATION,'output_schema':coverage_schema(size,[x['id'] for x in inventory])}


def check_coverage(raw,size,ids):
    validate(raw,coverage_schema(size,ids));seen=set()
    for r in raw['missing_regions']:Crop.from_model(r['bbox_2d'],size)
    for pair in raw['duplicate_candidates']:
        key=frozenset([pair['keep'],pair['other']])
        if len(key)!=2 or key in seen:raise ValueError('Invalid/repeated duplicate pair')
        seen.add(key)
    if bool(raw['missing_regions'])!=(raw['coverage_assessment']=='search_regions_proposed') and raw['coverage_assessment']!='uncertain':
        raise ValueError('Coverage assessment contradicts proposed regions')
    return raw


def discovery_schema(size,existing_ids):
    link={'oneOf':[enum(existing_ids),null()]} if existing_ids else null()
    return obj({'objects':arr(obj({'local_id':ident(),'label':string(),'shape_type':string(),'bbox_2d':bbox_schema(size),
        'matches_existing':link,'parent_id':link,'reason':string()})),
        'subregions':arr(obj({'bbox_2d':bbox_schema(size),'reason':string()})), 'uncertain':arr(string())})


def discovery(frame,inventory,reason):
    size=frame['actual_crop_input_size_px'];ids=[x['id'] for x in inventory]
    return {'prompt_version':VERSION,'stage':'shape_region_discovery','task':'Enumerate previously unrepresented shape objects in this model-selected crop; return boxes/identity only, not precise geometry points.',
        'coordinate_system':base.pixel_frame(size),'frame':frame,'scope':base.SCOPE,'search_reason':reason,'inventory':inventory,
        'rules':'Use actual crop pixels for bbox_2d. For an existing object or a part already in its fitted-part inventory, set matches_existing to that owner ID; do not rediscover it as new. A new distinct child inside a known container can have matches_existing=null and parent_id equal to that container; containment alone is not duplication. For dense or complicated regions you may choose smaller coherent subregions for another detection request; do not also enumerate the same contents in objects. Choose subdivisions from the image, never a fixed grid or fixed shape count. Every genuinely new object enters the refinement queue. Do not infer unseen internals or count source photographs/text/independent linework as shapes.',
        'serialization':base.SERIALIZATION,'output_schema':discovery_schema(size,ids)}


def check_discovery(raw,crop,existing_ids):
    validate(raw,discovery_schema(crop.input_size,existing_ids));seen=set()
    for o in raw['objects']:
        if o['local_id'] in seen:raise ValueError('Repeated discovery identity')
        seen.add(o['local_id']);Crop.from_model(o['bbox_2d'],crop.input_size)
        if o['matches_existing'] is not None and o['matches_existing']==o['parent_id']:raise ValueError('Same object cannot also be its parent')
    for r in raw['subregions']:Crop.from_model(r['bbox_2d'],crop.input_size)
    return raw
