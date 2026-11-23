"""Response schemas and checks are the exact contracts embedded in prompts."""
import math
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def obj(props):return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
def arr(item, minimum=0, maximum=None):
    out={'type':'array','items':item,'minItems':minimum}
    if maximum is not None:out['maxItems']=maximum
    return out
def num(n):return arr({'type':'number'},n,n)
def bounded_coordinates(maxima):
    return {'type':'array','prefixItems':[{'type':'number','minimum':0,'maximum':v} for v in maxima],
            'items':False,'minItems':len(maxima),'maxItems':len(maxima)}
def bbox_schema(size=None):return bounded_coordinates([size[0]-1,size[1]-1]*2) if size else num(4)
def string():return {'type':'string'}
def enum(values):return {'type':'string','enum':list(values)}
def ident():return {'type':'string','minLength':1}
def null():return {'type':'null'}
def validate(raw,schema):
    try:Draft202012Validator(schema).validate(raw)
    except ValidationError as exc:raise ValueError(f'{list(exc.absolute_path)}: {exc.message}') from exc
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('Non-finite JSON number')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(raw);return raw


def plan_schema(candidates,max_targets,size=None):
    return obj({'regions':arr(obj({'id':ident(),'target_ids':arr(enum([c['id'] for c in candidates]),1,max_targets),
                        'bbox_2d':bbox_schema(size),'reason':string()}),0,max_targets),
                'selection_reason':string(),'uncertain':arr(string())})


def part_schema():
    common={'id':ident()}
    return {'oneOf':[
        obj({**common,'kind':{'const':'rectangle'},'frame_point_ids':arr(ident(),4,4)}),
        obj({**common,'kind':{'const':'rounded_rectangle'},'frame_point_ids':arr(ident(),4,4),'tangent_point_ids':arr(ident(),2,2)}),
        obj({**common,'kind':{'const':'ellipse'},'center_id':ident(),'axis1_id':ident(),'axis2_id':ident()}),
        obj({**common,'kind':{'const':'path'},'closed':{'type':'boolean'},'segments':arr({'oneOf':[
            obj({'kind':{'const':'line'},'point_ids':arr(ident(),2,2)}),
            obj({'kind':{'const':'curve'},'point_ids':arr(ident(),3)})]},1)})]}


def refine_schema(candidate_id,input_size=None,global_size=None):
    return obj({'candidate_id':{'const':candidate_id},'decision':enum(['refined','needs_context','reject','unresolved']),
                'shape_type':string(),'points':arr(obj({'id':ident(),'xy':bounded_coordinates([v-1 for v in input_size]) if input_size else num(2)})),
                'parts':arr(part_schema()),'suggested_global_crop':{'oneOf':[bbox_schema(global_size),null()]},
                'reason':string(),'uncertain':arr(string())})


def part_refs(part):
    if part['kind'] in ('rectangle','rounded_rectangle'):return part['frame_point_ids']+part.get('tangent_point_ids',[])
    if part['kind']=='ellipse':return [part['center_id'],part['axis1_id'],part['axis2_id']]
    return [x for s in part['segments'] for x in s['point_ids']]


def check_refinement(raw,candidate_id,input_size,global_size):
    validate(raw,refine_schema(candidate_id,input_size,global_size))
    from .frames import Crop
    if raw['suggested_global_crop'] is not None:Crop.from_model(raw['suggested_global_crop'],global_size)
    if raw['decision']!='refined':
        if raw['points'] or raw['parts']:raise ValueError('Non-refined decision needs empty geometry')
        if raw['decision']=='needs_context' and raw['suggested_global_crop'] is None:raise ValueError('needs_context must select a new global crop')
        if raw['decision']!='needs_context' and raw['suggested_global_crop'] is not None:raise ValueError('Only needs_context may propose a crop')
        return raw
    if not raw['parts']:raise ValueError('refined decision requires nonempty parts')
    if raw['suggested_global_crop'] is not None:raise ValueError('refined decision cannot request another crop')
    points={};used=set();parts=set()
    for p in raw['points']:
        if p['id'] in points:raise ValueError('Duplicate local point ID')
        if any(not 0<=v<=input_size[i]-1 for i,v in enumerate(p['xy'])):raise ValueError('Local point out of actual crop bounds')
        points[p['id']]=p['xy']
    for part in raw['parts']:
        if part['id'] in parts:raise ValueError('Duplicate part ID')
        parts.add(part['id']);refs=part_refs(part)
        if any(p not in points for p in refs):raise ValueError('Undefined point reference')
        used.update(refs)
        if part['kind'] in ('rectangle','rounded_rectangle'):
            ids=part['frame_point_ids']
            if len(set(ids))!=4:raise ValueError('Box frame requires four distinct corner IDs')
            xy=[points[i] for i in ids]
            edges=[[b[i]-a[i] for i in (0,1)] for a,b in zip(xy,xy[1:]+xy[:1])]
            cross=[a[0]*b[1]-a[1]*b[0] for a,b in zip(edges,edges[1:]+edges[:1])]
            if not all(c>1e-6 for c in cross):raise ValueError('Box corners must be clockwise with nonzero convex area in image coordinates')
        elif part['kind']=='ellipse':
            c,a,b=[points[i] for i in refs]
            if abs((a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0]))<1e-6:raise ValueError('Degenerate ellipse')
        else:
            segments=part['segments']
            for left,right in zip(segments,segments[1:]):
                if left['point_ids'][-1]!=right['point_ids'][0]:raise ValueError('Adjacent segments must share endpoint IDs')
            if part['closed'] and segments[-1]['point_ids'][-1]!=segments[0]['point_ids'][0]:raise ValueError('Closed path must share final/initial ID')
            for segment in segments:
                xy=[points[p] for p in segment['point_ids']]
                if any(math.dist(a,b)<1e-6 for a,b in zip(xy,xy[1:])):raise ValueError('Zero-length path segment')
    if used!=set(points):raise ValueError('Unused local points')
    return raw


def audit_schema(all_ids,refined_ids,size=None):
    ids=enum(all_ids)
    reviews=arr(obj({'id':enum(refined_ids),'decision':enum(['accept','revise','reject','duplicate']),
                    'duplicate_of':{'oneOf':[ids,null()]},'reason':string()})) if refined_ids else {'type':'array','maxItems':0}
    return obj({'object_reviews':reviews,'missing_regions':arr(obj({'bbox_2d':bbox_schema(size),'shape_type':string(),'reason':string()})),
                'duplicate_candidates':arr(obj({'keep':ids,'other':ids,'reason':string()})),
                'uncertain':arr(string())})


def check_plan(raw,candidates,max_targets,size):
    from .frames import Crop
    validate(raw,plan_schema(candidates,max_targets,size))
    regions=set();selected=set()
    for region in raw['regions']:
        if region['id'] in regions:raise ValueError('Duplicate crop region ID')
        regions.add(region['id']);Crop.from_model(region['bbox_2d'],size)
        for target in region['target_ids']:
            if target in selected:raise ValueError('Candidate selected more than once')
            selected.add(target)
    if len(selected)>max_targets:raise ValueError('Selection exceeds experiment budget')
    return raw


def check_audit(raw,all_ids,refined_ids,size):
    from .frames import Crop
    validate(raw,audit_schema(all_ids,refined_ids,size))
    review_ids=[r['id'] for r in raw['object_reviews']]
    if len(review_ids)!=len(set(review_ids)) or set(review_ids)!=set(refined_ids):raise ValueError('Audit must review each refined object exactly once')
    links={};pairs=set()
    for review in raw['object_reviews']:
        duplicate=review['duplicate_of']
        if (review['decision']=='duplicate')!=(duplicate is not None):raise ValueError('duplicate_of required only for duplicate decision')
        if duplicate is not None:links[review['id']]=duplicate
    for pair in raw['duplicate_candidates']:
        identity=frozenset([pair['keep'],pair['other']])
        if len(identity)<2 or identity in pairs:raise ValueError('Invalid/repeated duplicate pair')
        pairs.add(identity)
        if pair['other'] in links and links[pair['other']]!=pair['keep']:raise ValueError('Conflicting duplicate identity')
        links[pair['other']]=pair['keep']
    for origin in links:
        seen=set();node=origin
        while node in links:
            if node in seen:raise ValueError('Cyclic duplicate identities')
            seen.add(node);node=links[node]
    for region in raw['missing_regions']:Crop.from_model(region['bbox_2d'],size)
    return raw
