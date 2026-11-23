"""Data consistency only. Masks may support multiple routes; no geometry veto."""
import copy,math

HEADS={'none','triangle','stealth','arrow','diamond','oval','unknown'}
KINDS={'straight','polyline','curve','block_arrow'}


def ids(value):
    if not isinstance(value,list):raise ValueError('Expected ID list')
    out=[]
    for x in value:
        if isinstance(x,str):out.append(x)
        elif isinstance(x,dict) and isinstance(x.get('members'),list):out.extend(x['members'])
        else:raise ValueError('IDs must be strings or explicit members lists')
    if any(not isinstance(x,str) for x in out) or len(set(out))!=len(out):raise ValueError('Unique string IDs required')
    return out


def bbox(b,size):
    if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (float,int) or not math.isfinite(x) for x in b):raise ValueError('Finite four-coordinate bbox required')
    if not (0<=b[0]<b[2]<=size[0] and 0<=b[1]<b[3]<=size[1]):raise ValueError('bbox must lie in the stated frame')


def graph(raw,size,sources=(),require_account=True,context_size=None,require_line_styles=False,allow_deferred=False):
    if not isinstance(raw,dict):raise ValueError('JSON object required')
    raw=copy.deepcopy(raw)
    if raw.get('action')=='expand':
        if context_size is None:raise ValueError('Expansion unavailable; return graph or unresolved candidates')
        bbox(raw.get('context_bbox'),context_size);return raw
    if not isinstance(raw.get('nodes'),list) or not isinstance(raw.get('routes'),list):raise ValueError('nodes and routes lists required')
    # Numeric labels carry the same reference semantics as strings. Convert labels
    # and references together, without touching coordinates, heads or topology.
    converted=False
    for p in raw['nodes']:
        if isinstance(p,list) and len(p)==3 and type(p[0]) is int:p[0]=str(p[0]);converted=True
    for r in raw['routes']:
        if not isinstance(r,dict):continue
        # A two-point "line" is an unambiguous spelling of our "straight".
        # Preserve its points, identities, and endpoint heads verbatim.
        if r.get('kind')=='line' and isinstance(r.get('nodes'),list) and len(r['nodes'])==2:
            r['kind']='straight';r['kind_normalization']='line -> straight; two-point geometry unchanged'
        if type(r.get('id')) is int:r['id']=str(r['id']);converted=True
        if isinstance(r.get('nodes'),list):
            if any(type(n) is int for n in r['nodes']):converted=True
            r['nodes']=[str(n) if type(n) is int else n for n in r['nodes']]
    if converted:raw['format_normalization']='Integer node/route labels converted consistently to strings; coordinates and semantics unchanged'
    nodes={};route_ids=set();used=set()
    for p in raw['nodes']:
        if not isinstance(p,list) or len(p)!=3 or not isinstance(p[0],str) or p[0] in nodes:raise ValueError('Unique [node_id,x,y] needed; integer/string label collisions are not allowed')
        if any(type(x) not in (float,int) or not math.isfinite(x) for x in p[1:]):raise ValueError('Finite node coordinates needed')
        if not all(-.5<=x<=n-.5 for x,n in zip(p[1:],size)):raise ValueError('Node outside CLEAN pixel frame')
        nodes[p[0]]=p[1:]
    for r in raw['routes']:
        if not isinstance(r,dict) or not isinstance(r.get('id'),str) or r['id'] in route_ids:raise ValueError('Unique route IDs needed')
        route_ids.add(r['id'])
        if r.get('kind') not in KINDS:raise ValueError('kind must be straight, polyline, curve or block_arrow')
        ns=r.get('nodes');hs=r.get('heads')
        deferred=allow_deferred and r.get('geometry')=='glm'
        if not isinstance(ns,list) or (not deferred and len(ns)<(3 if r['kind']=='block_arrow' else 2)) or any(n not in nodes for n in ns):raise ValueError('Ordered valid node references needed')
        if any(nodes[a]==nodes[b] for a,b in zip(ns,ns[1:])):raise ValueError('Consecutive nodes must differ')
        # A straight route may explicitly reference interior shared junctions.
        # Retain every node; route kind is not permission to erase topology.
        if not isinstance(hs,list) or len(hs)!=2 or any(h not in HEADS for h in hs):raise ValueError('Two ordered endpoint head types required')
        if r['kind']=='block_arrow' and hs!=['none','none']:raise ValueError('Block contour already contains its head')
        if require_line_styles and r['kind']!='block_arrow' and r.get('line_style') not in ('solid','dashed','dotted','unknown'):
            raise ValueError('Each non-block route needs line_style: solid/dashed/dotted/unknown')
        ss=ids(r.get('sources',[]))
        if not set(ss)<=set(sources) or (require_account and not ss):raise ValueError('Route must reference supplied source labels')
        r['sources']=ss;used.update(ss)
    if require_account:
        ex=ids(raw.get('excluded'));un=ids(raw.get('uncertain'))
        if used.intersection(ex+un) or set(ex).intersection(un):raise ValueError('A source cannot be used and rejected/unresolved')
        if used|set(ex)|set(un)!=set(sources):raise ValueError('Account for every supplied source label')
        raw.update(excluded=ex,uncertain=un)
    return raw


def review(raw,size,source_labels):
    if not isinstance(raw,dict) or raw.get('action') not in ('keep','replace','uncertain'):raise ValueError('action: keep, replace or uncertain')
    if raw['action']=='replace':return graph(raw,size,source_labels,False)
    return raw


def coverage(raw,size,known):
    if not isinstance(raw,dict) or not isinstance(raw.get('regions'),list):raise ValueError('regions list required')
    for r in raw['regions']:
        bbox(r.get('bbox'),size)
        r['targets']=ids(r.get('targets',[]))
        if not set(r['targets'])<=set(known):raise ValueError('Unknown existing target')
        if r.get('task') not in ('missing','repair'):raise ValueError('Region task must be missing or repair')
        if r['task']=='repair' and not r['targets']:raise ValueError('Repair names the targets to replace')
        if r['task']=='missing' and r['targets']:raise ValueError('Use repair to update existing targets')
    return raw


def association(raw,labels):
    if not isinstance(raw,dict) or raw.get('status') not in ('complete','partial','mixed','missing','uncertain'):raise ValueError('Invalid support status')
    out=copy.deepcopy(raw)
    for k in ('geometry','style'):
        out[k]=ids(raw.get(k,[]))
        if not set(out[k])<=set(labels):raise ValueError('Selection must reference displayed labels')
    if raw['status']=='complete' and not out['geometry']:raise ValueError('Complete geometry needs supporting evidence')
    if raw['status']=='missing' and (out['geometry'] or out['style']):raise ValueError('Missing has no selected support')
    if type(raw.get('more',False)) is not bool:raise ValueError('more must be boolean')
    return out
