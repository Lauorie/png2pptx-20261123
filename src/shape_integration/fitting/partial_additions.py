"""Retain independently reviewed additions without replacing old geometry."""
import copy

from .adapter import union_bounds, shared_point_diagnostics
from .contracts import part_refs

PROTOCOL='shape_integration.partial_additions.1'


def select(before,after,review,request):
    """Select only additions outside the existing critic's repair targets.

    Old commands and native parameters must be identical. This cannot accept
    a displaced/retyped replacement or infer source ownership from a label.
    The complete object remains unresolved and the criticized additions stay
    in the original attempted revision, outside the deliverable fit.
    """
    if (not request or not request.get('allow_new_parts') or
            review.get('decision')!='revise' or not review.get('issues') or
            before['frame']!=after['frame'] or
            any(f['fit_status']!='ok' or f['observation_status']!='ok' for f in (before,after))):
        return None
    old={p['id']:p for p in before['parts']};new={p['id']:p for p in after['parts']}
    if not old.keys()<new.keys():return None
    for pid,p in old.items():
        q=new[pid]
        if any(p.get(k)!=q.get(k) for k in ('status','kind','commands','parameters')):
            return None
    added=set(new)-set(old);held=set()
    for issue in review['issues']:
        ids=issue.get('part_ids') or [review.get('part_id_map',{}).get(p,p) for p in issue.get('parts',[])]
        if issue['kind']not in ('geometry','missing') or not ids or not set(ids)<=added:
            return None
        held.update(ids)
    # A shared point is a dependency, even if another part was not named in
    # the issue. Keep the whole dependent addition out of this partial commit.
    while True:
        refs={pid for key in held for pid in part_refs(new[key]['observed_part'])}
        linked={key for key in added if refs.intersection(part_refs(new[key]['observed_part']))}
        if linked<=held:break
        held.update(linked)
    accepted=added-held
    if not accepted:return None
    out=copy.deepcopy(after)
    out['parts']=[p for p in out['parts']if p['id']not in held]
    observation=out['observation']
    observation['parts']=[p for p in observation['parts']if p['id']not in held]
    refs={pid for p in observation['parts']for pid in part_refs(p)}
    observation['points']=[p for p in observation['points']if p['id']in refs]
    out['points']=[p for p in out['points']if p['id']in refs]
    observation['status']='partial'
    observation['reason']='Independent additions retained; unresolved additions held in attempted revision: '+', '.join(sorted(held))
    out.update(observation_status='partial',fit_status='partial',completeness_status='incomplete',
        bbox_2d=union_bounds(out['parts']),
        part_counts={'observed':len(out['parts']),'succeeded':len(out['parts']),'failed':0},
        shared_point_diagnostics=shared_point_diagnostics(out['parts'],{p['id']:p['xy']for p in observation['points']}))
    audit={'protocol':PROTOCOL,'retained_additions':sorted(accepted),'held_additions':sorted(held),
           'old_native_geometry_unchanged':True,'whole_object_accepted':False}
    out['partial_additions']=audit
    return out,audit
