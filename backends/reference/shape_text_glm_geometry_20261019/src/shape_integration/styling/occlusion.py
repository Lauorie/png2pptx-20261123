"""Whole-component occlusion; internal part order belongs to geometry recovery."""
import heapq
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from .geometry import footprint

PROTOCOL = 'shape_integration.component_occlusion.1'


def relation_batches(pairs, limit):
    """Never shrink unrelated overlap components into one model input crop."""
    parent={}
    def find(key):
        parent.setdefault(key,key)
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    for pair in pairs:
        a,b=find(pair['a']),find(pair['b'])
        if a!=b:parent[b]=a
    components={}
    for pair in pairs:components.setdefault(find(pair['a']),[]).append(pair)
    return [group[i:i+limit] for group in components.values() for i in range(0,len(group),limit)]


def discover(table, options):
    geoms={}; issues={}; edges=[]; pending=[]; members={}
    for key,part in table.items():
        geoms[key],error=footprint(part)
        if error: issues[key]=error
        members.setdefault(part['object_id'],[]).append(key)
    components={oid:unary_union([geoms[k] for k in keys]) for oid,keys in members.items()}
    component_issues={oid:[k for k in keys if k in issues] for oid,keys in members.items()
                      if any(k in issues for k in keys)}
    keys=list(components); tol=options['geometry_tolerance_px']
    for i,a in enumerate(keys):
        ga=components[a]
        if ga.is_empty: continue
        for b in keys[i+1:]:
            gb=components[b]
            if gb.is_empty or not ga.intersects(gb): continue
            if a in component_issues or b in component_issues:
                pending.append({'a':a,'b':b,'kind':'unsupported_geometry'}); continue
            # Equal outlines do not establish either direction.
            if ga.hausdorff_distance(gb)<=tol:
                pending.append({'a':a,'b':b,'kind':'coincident'}); continue
            if ga.area>0 and ga.buffer(tol).covers(gb):
                edges.append({'back':a,'front':b,'origin':'contains'}); continue
            if gb.area>0 and gb.buffer(tol).covers(ga):
                edges.append({'back':b,'front':a,'origin':'contains'}); continue
            inner=(ga.buffer(-tol).intersection(gb.buffer(-tol)).area
                   if ga.area>0 and gb.area>0 else
                   ga.buffer(-tol).intersection(gb).length if ga.area>0 else
                   gb.buffer(-tol).intersection(ga).length if gb.area>0 else 0)
            if inner > options['minimum_overlap_area_px']:
                pending.append({'a':a,'b':b,'kind':'partial_overlap'})
    return geoms, {'protocol':PROTOCOL,'scope':'components','component_parts':members,
                   'part_owners':{k:p['object_id'] for k,p in table.items()},
                   'edges':edges,'pending':pending,'geometry_issues':issues,
                   'component_geometry_issues':component_issues}


def stable_order(ids, edges):
    rank={v:i for i,v in enumerate(ids)}; adj={v:set() for v in ids}; degree={v:0 for v in ids}
    for a,b in edges:
        if a==b or a not in rank or b not in rank: continue
        if b not in adj[a]: adj[a].add(b); degree[b]+=1
    ready=[rank[v] for v in ids if degree[v]==0]; heapq.heapify(ready); out=[]
    while ready:
        a=ids[heapq.heappop(ready)];out.append(a)
        for b in sorted(adj[a],key=rank.get):
            degree[b]-=1
            if degree[b]==0: heapq.heappush(ready,rank[b])
    return (out,[]) if len(out)==len(ids) else (list(ids),[v for v in ids if degree[v]>0])


def order(scene, table, relations):
    if relations.get('scope')!='components':
        raise ValueError('Expected component occlusion relations; old part-level runs must stay frozen')
    owners={k:p['object_id'] for k,p in table.items()}; ids=[k for k in scene['objects'] if k in set(owners.values())]
    edges=relations['edges']; conflicts=[]
    if any(e['back'] not in ids or e['front'] not in ids or e['back']==e['front'] for e in edges):
        raise ValueError('Occlusion endpoints must be two different whole components')
    object_order,cycle=stable_order(ids,[(e['back'],e['front']) for e in edges])
    if cycle: conflicts.append({'type':'cross_group_cycle','objects':cycle})
    # Preserve the fitting result's internal ordering, including inferred hidden parts.
    part_orders={oid:[k for k in scene['objects'][oid]['part_keys'] if k in table] for oid in object_order}
    flat=[k for oid in object_order for k in part_orders[oid]]; rank={k:i for i,k in enumerate(object_order)}
    effective=[e for e in edges if e['back'] not in cycle and e['front'] not in cycle
               and rank[e['back']]<rank[e['front']]]
    rejected=[{**e,'reason':'component_order_cycle' if e['back'] in cycle or e['front'] in cycle
               else 'cannot_apply_component_order'} for e in edges if e not in effective]
    return {**relations,'object_order':object_order,'part_orders':part_orders,'flat_order':flat,
            'effective_edges':effective,'unapplied_edges':rejected,'order_conflicts':conflicts}


def paint(geom, style, unknown=False):
    if geom.is_empty: return geom
    if style is None or unknown:
        return geom if geom.area>0 else geom.buffer(7)
    items=[]
    if style.get('fill') is not None and geom.area>0: items.append(geom)
    if style.get('stroke') and style.get('width_px',0)>0:
        edge=geom.boundary if geom.area>0 else geom
        # Entire stroked band is conservatively unavailable, including unknown dash gaps.
        items.append(edge.buffer(style['width_px']/2+.5))
    return unary_union(items) if items else GeometryCollection()


def exclusions(key, geoms, relations, styles):
    """Only confirmed foreground components block source evidence.

    Internal parts, unknown relations and failed requests never create mutual
    masks. A compound's shared palette must retain all its visible colors.
    """
    owner=relations['part_owners'][key]
    adj={}
    for e in relations['effective_edges']: adj.setdefault(e['back'],set()).add(e['front'])
    ahead=set(); todo=list(adj.get(owner,()))
    while todo:
        v=todo.pop()
        if v in ahead: continue
        ahead.add(v);todo.extend(adj.get(v,()))
    blocks=[paint(geoms[k],styles.get(k)) for oid in ahead if oid!=owner
            for k in relations['component_parts'][oid] if geoms[key].intersects(geoms[k])]
    return unary_union(blocks) if blocks else GeometryCollection()
