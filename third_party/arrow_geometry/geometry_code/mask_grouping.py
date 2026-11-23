"""Whole observation scopes; proximity never decides physical connectivity."""
import numpy as np
from scipy.spatial import cKDTree
import evidence as ev


def group(rows,size,options):
    parent=list(range(len(rows)));relations=[];geometry=[]
    for r in rows:
        support=np.ascontiguousarray(r['_support'].astype('uint8')!=0)
        origin=np.asarray(r['box'][:2]);ys,xs=np.nonzero(support)
        # Observation scopes use raw support proximity, never skeleton topology.
        xy=np.c_[xs,ys]+origin
        geometry.append((cKDTree(xy) if len(xy) else None,xy))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    distance=options['endpoint_gap_at_2048']*max(size)/2048
    for i,a in enumerate(rows):
        for j in range(i+1,len(rows)):
            b=rows[j]
            if ev.gap(a['box'],b['box'])>distance:continue
            overlap=ev.intersect(a,b)/max(1,min(a['area'],b['area']))
            nearest=float('inf')
            for source,dest in [(i,j),(j,i)]:
                tree,points=geometry[dest][0],geometry[source][1]
                if tree is not None and len(points):nearest=min(nearest,float(tree.query(points)[0].min()))
            if overlap>=options['containment'] or (distance>0 and nearest<=distance):
                parent[find(j)]=find(i)
                relations.append(dict(a=a['id'],b=b['id'],containment=round(overlap,4),
                    endpoint_distance=None if not np.isfinite(nearest) else round(nearest,3)))
    scopes={}
    for i,r in enumerate(rows):scopes.setdefault(find(i),[]).append(r)
    return list(scopes.values()),relations
