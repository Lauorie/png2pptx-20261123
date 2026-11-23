"""Visual style groups across repeated single-part objects; compounds excluded."""
import math
from concurrent.futures import as_completed
from .geometry import bounds, closed
from . import views
from ..common import save
from ..transport import RequestTimeout

PROTOCOL='shape_integration.component_style_groups.1'


def candidate_sets(scene, table, selected):
    """Geometry proposes candidates only; the model must confirm appearance."""
    entries=[]
    for obj in scene['objects'].values():
        # Count original membership, not just successful/selected parts.
        if len(obj['part_keys'])!=1: continue
        key=obj['part_keys'][0]
        if key not in table or key not in selected: continue
        p=table[key];b=bounds(p);w=max(b[2]-b[0],1e-6);h=max(b[3]-b[1],1e-6)
        entries.append((key,p['kind'],closed(p),w/h,math.sqrt(w*h)))
    groups=[]
    for item in entries:
        # Complete-link proposal avoids a chain of dissimilar sizes/ratios.
        for group in groups:
            if all(item[1:3]==other[1:3] and
                   max(item[3],other[3])/min(item[3],other[3])<=1.4 and
                   max(item[4],other[4])/min(item[4],other[4])<=2.5 for other in group):
                group.append(item);break
        else: groups.append([item])
    return [[x[0] for x in group] for group in groups if len(group)>1]


def validate(raw, mapping):
    groups=raw.get('groups') if isinstance(raw,dict) else None
    if not isinstance(groups,list):raise ValueError('groups must be an array, possibly empty')
    used=set();out=[]
    for i,g in enumerate(groups):
        if not isinstance(g,dict):raise ValueError(f'groups.{i}: expected object')
        members=g.get('members')
        if not isinstance(members,list) or len(members)<2 or any(not isinstance(k,str) or k not in mapping for k in members):
            raise ValueError(f'groups.{i}.members: at least two supplied IDs required')
        if len(set(members))!=len(members) or used.intersection(members):
            raise ValueError(f'groups.{i}.members: duplicate membership')
        if g.get('representative') not in members:raise ValueError(f'groups.{i}.representative must be a member')
        if not isinstance(g.get('evidence'),str) or not g['evidence'].strip():
            raise ValueError(f'groups.{i}.evidence: visible source evidence required')
        used.update(members)
        out.append({k:g[k] for k in ('members','representative','evidence')})
    return {'groups':out}


def discover(scene, source, table, selected, folder, runtime, options, pool, assets=()):
    proposals=candidate_sets(scene,table,selected) if options['component_grouping_enabled'] else []
    audit={'protocol':PROTOCOL,'enabled':options['component_grouping_enabled'],
           'compound_objects_excluded':True,'proposals':proposals,'groups':[], 'errors':[]}
    def classify(index, keys):
        root=folder/f'candidate_{index:03d}';mapping={f'C{i}':k for i,k in enumerate(keys)}
        images,units,_=views.prepare(source,table,mapping,root/'input',options,assets=assets)
        prompt={'protocol':PROTOCOL,'task':'Identify repeated components that can share one complete style. Only the listed independent single-part objects are eligible.',
                'targets':[{'id':a} for a in mapping],
                'rules':[
                    'Judge CLEAN source appearance. Labels, contours and IMG boxes in the separate target view only identify objects; ignore text content and arrows inside/around them.',
                    'Group only visibly matching fill (including no-fill or gradient), stroke color, stroke weight and line pattern. Similar shape, nearby position or repeated labels alone are insufficient. Separate clearly different colors, highlights and border weights.',
                    'Return only definite repeated groups of at least two IDs. Omit unique or uncertain components; they will follow the ordinary recovery pipeline. Each ID belongs to at most one group.',
                    'Choose the clearest, least-occluded member as representative. Give brief source evidence. Do not estimate numerical colors, widths or coordinates.'],
                'output':{'groups':[{'members':['C0','C1'],'representative':'C0','evidence':'brief visible evidence of the same complete style'}]}}
        save(root/'job.json',{'mapping':mapping,'prompt':prompt})
        stage=runtime.for_stage(shape_schema_retries=0)
        for attempt in range(options['protocol_retries']+1):
            try:
                result=stage.call(root/'calls',f'group_{attempt:02d}',
                                  'Compare source appearances of existing components. Return one JSON object only.',
                                  prompt,images,lambda raw:validate(raw,mapping),options['max_tokens'])
                return [{'members':[mapping[a] for a in g['members']],
                         'representative':mapping[g['representative']], 'evidence':g['evidence'],
                         'request':str(root)} for g in result['groups']]
            except RequestTimeout:raise
            except (ValueError,RuntimeError) as exc:
                if attempt==options['protocol_retries'] or 'Interrupted prior call' in str(exc):raise
                prompt={**prompt,'format_correction':str(exc)}
    jobs={pool.submit(classify,i,keys):i for i,keys in enumerate(proposals)} if runtime else {}
    results={}
    for future in as_completed(jobs):
        index=jobs[future]
        try:results[index]=future.result()
        except Exception as exc:audit['errors'].append({'proposal':index,'error':str(exc),'fallback':'ordinary_pipeline'})
    for index in sorted(results):
        for group in results[index]:
            audit['groups'].append({'id':f'S{len(audit["groups"])+1:03d}',**group})
    save(folder/'groups.json',audit)
    return audit
