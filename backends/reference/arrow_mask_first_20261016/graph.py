"""Explicit model point graphs. No snapping, smoothing, mask fitting or head guessing."""
import copy
import numpy as np


def empty():return dict(nodes={},routes=[])


def edges(graph):
    """Shared edges come ONLY from explicit identical node references."""
    table={}
    for route in graph['routes']:
        ns=route['nodes'];pairs=list(zip(ns,ns[1:]))
        if route['kind']=='block_arrow':pairs.append((ns[-1],ns[0]))
        for a,b in pairs:
            key=tuple(sorted((a,b)))
            row=table.setdefault(key,dict(id=f'E{len(table)+1}',nodes=[a,b],routes=[]))
            row['routes'].append(route['id'])
    return list(table.values())


def points(graph,route):return np.asarray([graph['nodes'][n] for n in route['nodes']],float)


def box(graph):
    p=np.asarray([graph['nodes'][n] for r in graph['routes'] for n in r['nodes']],float)
    return [*p.min(axis=0),*p.max(axis=0)] if len(p) else [0,0,1,1]


def from_answer(answer,view,prefix):
    matrix=np.asarray(view['view_to_global']);nodes={}
    for name,x,y in answer['nodes']:
        xy=(matrix@np.array([x,y,1.]))[:2];nodes[prefix+':'+name]=xy.tolist()
    routes=[]
    for r in answer['routes']:
        row=copy.deepcopy(r);row['id']=prefix+':'+r['id']
        row['nodes']=[prefix+':'+n for n in r['nodes']];routes.append(row)
    return dict(nodes=nodes,routes=routes)


def local(graph,view):
    matrix=np.asarray(view['global_to_view']);out=copy.deepcopy(graph)
    out['nodes']={n:(matrix@np.r_[p,1])[:2].tolist() for n,p in graph['nodes'].items()}
    return out


def combine(graphs):
    out=empty()
    for g in graphs:
        for n,p in g['nodes'].items():
            if n in out['nodes'] and out['nodes'][n]!=p:raise ValueError('Conflicting shared node '+n)
            out['nodes'][n]=p
        out['routes']+=copy.deepcopy(g['routes'])
    if len({r['id'] for r in out['routes']})!=len(out['routes']):raise ValueError('Duplicate route ID')
    return out


def components(graph,prefix):
    """Only explicit shared node references define networks; equal coordinates do not."""
    pending=list(graph['routes']);output=[]
    while pending:
        group=[pending.pop(0)];used=set(group[0]['nodes']);changed=True
        while changed:
            changed=False
            for r in list(pending):
                if used.intersection(r['nodes']):
                    group.append(r);used.update(r['nodes']);pending.remove(r);changed=True
        tid=f'{prefix}.{len(output)+1}'
        output.append(dict(id=tid,graph=dict(nodes={n:p for n,p in graph['nodes'].items() if n in used},routes=group),
            sources=sorted({s for r in group for s in r.get('sources',[])}),ancestors=[tid]))
    return output


def brief(graph,view=None):
    g=local(graph,view) if view else graph
    names={n:f'P{i}' for i,n in enumerate(g['nodes'],1)}
    return dict(nodes=[[names[n],round(p[0],1),round(p[1],1)] for n,p in g['nodes'].items()],
        routes=[dict(id=f'E{i}',kind=r['kind'],nodes=[names[n] for n in r['nodes']],heads=r['heads'])
                for i,r in enumerate(g['routes'],1)])


def replace_targets(targets,ids,new):
    """Closed replacement transaction. Unaffected targets and their branches survive."""
    known={t['id'] for t in targets}
    if len(set(ids))!=len(ids) or not set(ids)<=known:raise ValueError('Unknown or multiply consumed target')
    if (known-set(ids)).intersection(t['id'] for t in new):raise ValueError('Replacement ID collision')
    return [t for t in targets if t['id'] not in ids]+new
