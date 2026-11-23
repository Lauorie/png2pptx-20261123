"""Discovery provenance is not a per-line matching problem.

Adapted from the successful 2026-10-20 run_observation_sources.py replay.
Pixel support remains explicit in M/D; pooled G IDs are context provenance only.
"""
import copy


def prompt(value):
    q=copy.deepcopy(value)
    q['task']='Identify visible linework in this local observation and choose usable mask evidence or a separate geometry request.'
    q['observation_sources']=q.pop('sources')
    q['instructions'][1]=(
        'CLEAN is the local pixel frame; CONTEXT locates it. G boxes collectively locate discovery evidence, not confirmed objects. '
        'Do not match G IDs to individual paths, M masks or D proposals, or account for every G. The program stores observation provenance. '
        'Inspect visible linework supported by the supplied local observation, including eligible newly found local M masks with no G owner. '
        'M/D are optional pixel measurements, not a required number of targets. Use a matching D for a complete simple line; otherwise select masks or request GLM geometry. '
        'A dense region with ambiguous per-line masks may be one geometry=glm target: supply a short visual locator, not a complete edge census or architecture explanation. '
        'The geometry request can return multiple independent paths; grouping for observation does not imply physical connection. '
        'Only true visible junctions need network labels. Keep crossings independent. Ignore unrelated photographic texture and text.')
    for m in q['masks']:m.pop('global_overlap',None)
    q['output_schema']['routes']=q['output_schema']['routes'].replace('sources:[G IDs],','')
    q['output_schema'].pop('excluded',None);q['output_schema'].pop('uncertain',None)
    q['output_schema']['status']='complete or uncertain; uncertain means eligible linework remains unresolved'
    q['instructions'].append('For geometry=glm, nodes=[], proposal=null; omit heads/line_style because these are judged on final paths in the style stage. Do not spend time guessing their types now. For ordinary mask/points routes retain ordered heads and line_style as rough evidence.')
    return q


def answer(raw,labels):
    a=copy.deepcopy(raw)
    for route in a.get('routes',[]):
        route['sources']=list(labels);route['source_linking']='observation_context_only'
        if route.get('geometry')=='glm':
            route.setdefault('heads',['unknown','unknown']);route.setdefault('line_style','unknown')
            route.setdefault('nodes',[]);route.setdefault('proposal',None);route.setdefault('network',None)
    unresolved=a.get('status')=='uncertain'
    a['excluded']=[] if a.get('routes') or unresolved else list(labels)
    a['uncertain']=list(labels) if unresolved and not a.get('routes') else []
    a['observation_incomplete']=unresolved
    a['source_linking']='observation_context_only'
    return a
