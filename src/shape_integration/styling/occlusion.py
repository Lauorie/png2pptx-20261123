"""Whole-component occlusion; internal part order belongs to geometry recovery."""
import heapq
import numpy as np
from shapely.geometry import GeometryCollection, Point, box
from shapely.ops import unary_union
from .geometry import footprint

PROTOCOL = 'shape_integration.component_occlusion.6'


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
    open_lines={oid:unary_union([geoms[k] for k in keys if geoms[k].area==0])
                for oid,keys in members.items()}
    component_issues={oid:[k for k in keys if k in issues] for oid,keys in members.items()
                      if any(k in issues for k in keys)}
    keys=list(components); tol=options['geometry_tolerance_px']
    for i,a in enumerate(keys):
        ga=components[a]
        if ga.is_empty: continue
        for b in keys[i+1:]:
            gb=components[b]
            if gb.is_empty or not ga.intersects(gb): continue
            # Equal outlines do not establish either direction.
            if ga.hausdorff_distance(gb)<=tol:
                pending.append({'a':a,'b':b,'kind':'coincident'}); continue
            if a not in component_issues and ga.area>0 and ga.buffer(tol).covers(gb):
                edges.append({'back':a,'front':b,'origin':'contains'}); continue
            if b not in component_issues and gb.area>0 and gb.buffer(tol).covers(ga):
                edges.append({'back':b,'front':a,'origin':'contains'}); continue
            # An invalid interior path does not invalidate the observed outer
            # container's inclusion of this component. Partial overlaps still
            # abstain: no repaired polygon or guessed silhouette is introduced.
            if a in component_issues or b in component_issues:
                pending.append({'a':a,'b':b,'kind':'unsupported_geometry'}); continue
            inner=(ga.buffer(-tol).intersection(gb.buffer(-tol)).area
                   if ga.area>0 and gb.area>0 else
                   ga.buffer(-tol).intersection(gb).length if ga.area>0 else
                   gb.buffer(-tol).intersection(ga).length if gb.area>0 else 0)
            # Eroding two mixed node/wire components removes their open lines.
            # A connector can still enter the other component's face. Nominate
            # that source-order question through the existing relation review;
            # neither geometry nor component type chooses which one is in front.
            line_contact=(open_lines[a].intersection(gb.buffer(-tol)).length>tol or
                          open_lines[b].intersection(ga.buffer(-tol)).length>tol)
            if inner > options['minimum_overlap_area_px'] or line_contact:
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
    # A contradictory partial-overlap claim must not reset the entire slide.
    # Containment is established before the visual pair review; retain these
    # constraints and add each observed relation only if it stays acyclic.
    effective=[];rejected=[]
    children={oid:{e['front'] for e in edges if e.get('origin')=='contains' and e['back']==oid} for oid in ids}
    # Resolve overlapping containers before incidental overlaps with their
    # children. Otherwise one child-vs-rear-frame claim can reverse the group.
    priority=lambda e:(e.get('origin')!='contains',-len(children[e['back']]|children[e['front']]))
    for edge in sorted(edges,key=priority):
        if any(e['back']==edge['front'] and e['front']==edge['back'] and priority(e)==priority(edge) for e in edges):
            rejected.append({**edge,'reason':'contradictory_pair'})
            conflicts.append({'type':'contradictory_pair','objects':[edge['back'],edge['front']]})
            continue
        _,cycle=stable_order(ids,[(e['back'],e['front']) for e in effective+[edge]])
        if cycle:
            rejected.append({**edge,'reason':'conflicts_with_existing_order'})
            conflicts.append({'type':'cross_group_cycle','objects':cycle,'rejected_edge':edge})
        else:effective.append(edge)
    object_order,_=stable_order(ids,[(e['back'],e['front']) for e in effective])
    # Preserve the fitting result's internal ordering, including inferred hidden parts.
    part_orders={oid:[k for k in scene['objects'][oid]['part_keys'] if k in table] for oid in object_order}
    flat=[k for oid in object_order for k in part_orders[oid]]
    return {**relations,'object_order':object_order,'part_orders':part_orders,'flat_order':flat,
            'effective_edges':effective,'unapplied_edges':rejected,'order_conflicts':conflicts}


def paint(geom, style, unknown=False):
    if geom.is_empty: return geom
    if style is None or unknown:
        return geom if geom.area>0 else geom.buffer(7)
    items=[]
    if style.get('fill') is not None and style.get('fill_opacity',1)==1 and geom.area>0: items.append(geom)
    if style.get('stroke') and style.get('width_px',0)>0:
        edge=geom.boundary if geom.area>0 else geom
        # Entire stroked band is conservatively unavailable, including unknown dash gaps.
        items.append(edge.buffer(style['width_px']/2+.5))
    return unary_union(items) if items else GeometryCollection()


def visible_stroke_in_face(line, face, stroke, fill, pixels):
    """Veto a guessed hidden stroke when its original-source ink is visible."""
    color, background = stroke.get('stroke'), fill.get('fill')
    if not isinstance(color, str) or not isinstance(background, str) or fill.get('fill_opacity', 1) < 1:
        return None
    width = stroke.get('width_px') or 1.
    foreground = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
    backdrop = np.array([int(background[i:i+2], 16) for i in (1, 3, 5)])
    contrast = float(np.linalg.norm(foreground - backdrop))
    if contrast < 30:
        return None
    interior = face.buffer(-max(1., width))
    clipped = line.intersection(interior)
    segments = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]
    radius = max(1., width / 2 + 1.)
    offsets = np.linspace(-radius, radius, max(3, int(np.ceil(4 * radius)) + 1))
    for segment in segments:
        minimum = max(3., 2 * width)
        if not hasattr(segment, 'coords') or segment.length < minimum:
            continue
        distances = np.linspace(0., segment.length, max(3, int(np.ceil(segment.length * 2)) + 1))
        points = np.array([segment.interpolate(float(d)).coords[0] for d in distances])
        tangent = np.gradient(points, axis=0)
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-9)
        xy = np.rint(points[:, None] + normal[:, None] * offsets[None, :, None]).astype(int)
        xy[:, :, 0] = np.clip(xy[:, :, 0], 0, pixels.shape[1] - 1)
        xy[:, :, 1] = np.clip(xy[:, :, 1], 0, pixels.shape[0] - 1)
        rgb = pixels[xy[:, :, 1], xy[:, :, 0]].astype(float)
        ink = np.linalg.norm(rgb - foreground, axis=2)
        face_distance = np.linalg.norm(rgb - backdrop, axis=2)
        hit = np.any((ink < max(20., min(65., contrast * .35))) &
                     (ink + 10 < face_distance), axis=1)
        longest = run = 0
        for supported in hit:
            run = run + 1 if supported else 0
            longest = max(longest, run)
        step = segment.length / (len(distances) - 1)
        if hit.mean() >= .65 and longest * step >= minimum:
            return {'supported_fraction': float(hit.mean()),
                    'supported_run_px': longest * step,
                    'interior_fragment_length_px': segment.length,
                    'normal_radius_px': radius}
    return None


def source_frame_orders(keys, table, styles, geoms, pixels, tolerance, *, closed_faces=False):
    """Observe frame occlusion without changing a contour, pigment or fill."""
    frames=[k for k in keys if (closed_faces or table[k].get('kind') in ('rectangle','rounded_rectangle'))
            and isinstance(styles[k].get('fill'),str) and styles[k].get('fill_opacity',1)==1
            and styles[k].get('stroke') and geoms[k].area>0]
    canvas=box(0,0,pixels.shape[1]-1,pixels.shape[0]-1);result=[]
    for i,a in enumerate(frames):
        for b in frames[i+1:]:
            ga,gb=geoms[a],geoms[b]
            if not ga.intersects(gb) or ga.buffer(tolerance).covers(gb) or gb.buffer(tolerance).covers(ga):continue
            def inside(key,peer):
                radius=max(1.,(styles[key].get('width_px') or 1)/2+1.)
                return visible_stroke_in_face(geoms[key].boundary.intersection(canvas),
                         geoms[peer].buffer(-radius),styles[key],styles[peer],pixels)
            av,bv=inside(a,b),inside(b,a)
            if bool(av)==bool(bv):continue
            front,back=(a,b) if av else (b,a)
            # Missing ink inside alone could be an inaccurate fitted contour.
            # Its surviving exposed part must match the same source pigment.
            outside=canvas.difference(geoms[front].buffer(2))
            exposed=visible_stroke_in_face(geoms[back].boundary,outside,styles[back],styles[back],pixels)
            if exposed is None:continue
            result.append({'back':back,'front':front,'origin':'source_visible_frame_order',
                           'source_visible_front_edge':av or bv,'source_exposed_back_edge':exposed})
    return result


def interleave_named_parts(relations, table, styles, geoms, pixels, tolerance):
    """A relation between two leaves cannot move either whole compound.

    Keep internal order and unrelated component constraints. Only an explicit
    intersecting leaf pair can override its component-level default; visible
    source contours may disprove that pair before it is applied.
    """
    proposed=relations.get('cross_part_model_edges',[])
    if not proposed:return relations
    ids=relations['flat_order'];constraints=[];accepted=[];rejected=[]
    for keys in relations['part_orders'].values():constraints.extend(zip(keys,keys[1:]))
    for edge in proposed:
        a,b=edge['back'],edge['front']
        if a not in table or b not in table or not geoms[a].intersects(geoms[b]):continue
        measured=source_frame_orders([a,b],table,styles,geoms,pixels,tolerance,closed_faces=True) if pixels is not None else []
        selected=measured[0] if measured else edge
        if (selected['back'],selected['front'])!=(a,b):
            rejected.append({**edge,'reason':'contradicts_visible_source_contours','source_order':selected})
        pair=(selected['back'],selected['front'])
        if stable_order(ids,constraints+[pair])[1]:
            rejected.append({**selected,'reason':'conflicts_with_internal_part_order'});continue
        constraints.append(pair);accepted.append(selected)
    named={(e['back'],e['front']) for e in accepted}
    for edge in relations['effective_edges']:
        for a in relations['part_orders'][edge['back']]:
            for b in relations['part_orders'][edge['front']]:
                if not geoms[a].intersects(geoms[b]) or (b,a) in named:continue
                pair=(a,b)
                if not stable_order(ids,constraints+[pair])[1]:constraints.append(pair)
                else:rejected.append({'back':a,'front':b,'origin':edge['origin'],
                                      'reason':'component_default_conflicts_with_named_part_order'})
    flat,cycle=stable_order(ids,constraints)
    if cycle:raise ValueError('Unexpected cross-part order cycle')
    return {**relations,'flat_order':flat,'cross_part_effective_edges':accepted,
            'cross_part_rejected_orders':rejected}


def order_internal_by_paint(scene, table, relations, styles, geoms, tolerance=.2, *, source=None):
    """Opaque containing parts go behind their visible contained peers.

    The fitting stage's part list is not an appearance-aware paint order. A
    background listed after its dots otherwise hides them despite correct styles.
    Geometry and membership stay fixed; unrelated parts keep their stable order.
    """
    part_orders={};audit=[];conflicts=[];rejected=[];pixels=None
    explicit=list(relations.get('internal_model_edges',[]));source_edges=[]
    for oid in relations['object_order']:
        keys=[k for k in scene['objects'][oid]['part_keys'] if k in table]
        observed=[e for e in explicit if e['back'] in keys and e['front'] in keys]
        if source is not None and sum(table[k].get('kind') in ('rectangle','rounded_rectangle')
                and styles[k].get('fill') is not None for k in keys)>1:
            if pixels is None:pixels=np.asarray(source.convert('RGB'))
            measured=source_frame_orders(keys,table,styles,geoms,pixels,tolerance)
            if measured:
                supported=[];constraints=[]
                for edge in measured:
                    candidate=(edge['back'],edge['front'])
                    if stable_order(keys,constraints+[candidate])[1]:
                        rejected.append({**edge,'reason':'conflicting_source_frame_order'});continue
                    supported.append(edge);constraints.append(candidate)
                for edge in observed:
                    candidate=(edge['back'],edge['front'])
                    if stable_order(keys,constraints+[candidate])[1]:
                        rejected.append({**edge,'reason':'contradicts_visible_source_frame_edges'})
                    elif candidate not in constraints:
                        supported.append(edge);constraints.append(candidate)
                observed=supported;source_edges.extend(e for e in observed if e.get('origin')=='source_visible_frame_order')
                explicit=[e for e in explicit if not(e['back'] in keys and e['front'] in keys)]+observed
        edges=[(e['back'],e['front']) for e in observed]
        audit.extend(observed);contains=[];attached=[];dividers={}
        # A boundary-to-boundary line inside a face is a surface divider, not
        # an external connector ending in a node. Use the existing geometry
        # tolerance: independently observed endpoints need not be bit-identical.
        for b in keys:
            gb=geoms[b]
            if gb.is_empty or gb.area>0:continue
            strokes=list(gb.geoms) if hasattr(gb,'geoms') else [gb]
            endpoints=[Point(xy) for line in strokes if hasattr(line,'coords') and len(line.coords)>1
                       for xy in (line.coords[0],line.coords[-1])]
            if not endpoints:continue
            owners=[a for a in keys if styles[a].get('fill') is not None and geoms[a].area>0
                    and geoms[a].buffer(tolerance).covers(gb)
                    and all(geoms[a].boundary.distance(p)<=tolerance for p in endpoints)]
            if owners:dividers[b]=owners
        for edge in observed:
            for line,owners in dividers.items():
                if owners==[edge['back']]:
                    contains.append({'back':line,'front':edge['front'],'origin':'divider_follows_observed_face_order'})
        for a in keys:
            ga=geoms[a]
            if styles[a].get('fill') is None or ga.area<=0:continue
            # Independently fitted line endpoints can fall within the face's
            # visible outline rather than its ideal centreline. That remains
            # containment; otherwise a tiny rounding difference can put an
            # entire multi-stroke detail behind its own opaque face.
            outline=(styles[a].get('width_px') or 0)/2 if styles[a].get('stroke') else 0
            contained_line=ga.buffer(max(tolerance,outline))
            for b in keys:
                if a==b or not (styles[b].get('fill') is not None or styles[b].get('stroke')):continue
                gb=geoms[b]
                if ga.area>gb.area and (contained_line.covers(gb) if gb.area==0 else ga.covers(gb)):
                    if (b,a) in edges:continue
                    contains.append({'back':a,'front':b,'origin':'opaque_internal_contains'})
                elif gb.area==0 and b not in dividers and ga.intersects(gb):
                    strokes=list(gb.geoms) if hasattr(gb,'geoms') else [gb]
                    if any(hasattr(line,'coords') and len(line.coords)>1 and
                           any(ga.contains(Point(xy)) for xy in (line.coords[0],line.coords[-1]))
                           for line in strokes):
                        if source is not None:
                            if pixels is None:
                                pixels = np.asarray(source.convert('RGB'))
                            evidence = visible_stroke_in_face(gb, ga, styles[b], styles[a], pixels)
                            if evidence is not None:
                                rejected.append({'back':b,'front':a,
                                    'origin':'attached_line_behind_opaque_node',
                                    'reason':'source_visible_stroke_inside_face',
                                    'source_support':evidence})
                                continue
                        attached.append({'back':b,'front':a,'origin':'attached_line_behind_opaque_node'})
        # Visible model relations take priority, then full containment, then
        # endpoint heuristics. A weaker guess must not destroy valid ordering
        # by introducing a cycle (e.g. a grid line crossing stacked planes).
        for edge in contains+attached:
            candidate=(edge['back'],edge['front'])
            if candidate in edges:continue
            _,cycle=stable_order(keys,edges+[candidate])
            if cycle:
                rejected.append({**edge,'reason':'conflicts_with_existing_order'})
                continue
            edges.append(candidate);audit.append(edge)
        ordered,cycle=stable_order(keys,edges);part_orders[oid]=ordered
        if cycle:conflicts.append({'type':'internal_paint_cycle','parts':cycle})
    result={**relations,'part_orders':part_orders,
            'flat_order':[k for oid in relations['object_order'] for k in part_orders[oid]],
            'internal_style_edges':audit,'internal_style_conflicts':conflicts,
            'internal_rejected_heuristics':rejected,'internal_model_edges':explicit,
            'source_frame_order_edges':source_edges}
    if source is not None and relations.get('cross_part_model_edges') and pixels is None:
        pixels=np.asarray(source.convert('RGB'))
    return interleave_named_parts(result,table,styles,geoms,pixels,tolerance)


def exclusions(key, geoms, relations, styles):
    """Only confirmed foreground components block source evidence.

    Explicit internal order also applies to an unambiguously attached surface
    divider. Containment or list order alone never establishes occlusion.
    """
    owner=relations['part_owners'][key]
    adj={}
    for e in relations['effective_edges']: adj.setdefault(e['back'],set()).add(e['front'])
    ahead=set(); todo=list(adj.get(owner,()))
    while todo:
        v=todo.pop()
        if v in ahead: continue
        ahead.add(v);todo.extend(adj.get(v,()))
    rank={k:i for i,k in enumerate(relations.get('flat_order',[]))}
    interleaved='cross_part_effective_edges' in relations
    blocks=[paint(geoms[k],styles.get(k)) for oid in ahead if oid!=owner
            for k in relations['component_parts'][oid] if geoms[key].intersects(geoms[k])
            and (not interleaved or rank[k]>rank[key])]
    # An explicit source-observed internal relation is also real occlusion.
    # Unspecified list order never hides evidence; cyclic claims abstain.
    internal={}
    for e in relations.get('internal_model_edges',[]):
        internal.setdefault(e['back'],set()).add(e['front'])
    for e in relations.get('cross_part_effective_edges',[]):
        internal.setdefault(e['back'],set()).add(e['front'])
    # These are inherited from an explicit face order, not the generic
    # containment/endpoint heuristics. Rendering and source sampling must agree:
    # a rear grid divider must not measure the front face's unrelated ink.
    for e in relations.get('internal_style_edges',[]):
        if e.get('origin')=='divider_follows_observed_face_order':
            internal.setdefault(e['back'],set()).add(e['front'])
    foreground=set();todo=list(internal.get(key,()))
    while todo:
        part=todo.pop()
        if part in foreground:continue
        foreground.add(part);todo.extend(internal.get(part,()))
    if key not in foreground:
        blocks.extend(paint(geoms[k],styles.get(k)) for k in foreground
                      if k in geoms and geoms[key].intersects(geoms[k]))
    return unary_union(blocks) if blocks else GeometryCollection()
