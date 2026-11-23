"""Compact typed contracts; semantic uncertainty stays separate from format errors."""
import re
import math
import json
from ..common import valid_box
from ..styling.contracts import number,color


def ident(v):
    if not isinstance(v,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',v):raise ValueError('ID must start with a letter and contain only letters/digits/_/-')
    return v


def discovery(raw,size,known=(),layout=False):
    if not isinstance(raw,dict) or not isinstance(raw.get('objects'),list):raise ValueError('objects array required')
    objects=[]
    for i,row in enumerate(raw['objects']):
        if not isinstance(row,dict) or not valid_box(row.get('bbox_2d'),size):raise ValueError(f'objects.{i}.bbox_2d must be a positive box within {size[0]-1},{size[1]-1}')
        text=row.get('description')
        if not isinstance(text,str) or not text.strip():raise ValueError(f'objects.{i}.description required')
        same=row.get('same_as')
        if same is not None and same not in known:raise ValueError('same_as must reference a visible existing object ID or null')
        objects.append({'bbox_2d':row['bbox_2d'],'description':text,'same_as':same})
    regions=[]
    if layout:
        for row in raw.get('regions',[]):
            if not valid_box(row.get('bbox_2d'),size):raise ValueError('Invalid layout region bbox')
            regions.append({'bbox_2d':row['bbox_2d'],'description':str(row.get('description',''))})
    return {'objects':objects,'regions':regions,'uncertain':raw.get('uncertain',[])}


def geometry(raw,size):
    state=raw.get('status','ready')
    if state not in ('ready','needs_context','not_arrow'):raise ValueError('status must be ready/needs_context/not_arrow')
    if state!='ready':
        if not isinstance(raw.get('reason'),str) or not raw['reason'].strip():raise ValueError('non-ready geometry needs reason')
        return {'status':state,'reason':raw['reason'],'points':{},'paths':[]}
    points=raw.get('points');paths=raw.get('paths')
    if not isinstance(points,dict) or not points or not isinstance(paths,list) or not paths:raise ValueError('Nonempty points table and paths required')
    pts={}
    for key,xy in points.items():
        ident(key)
        if not isinstance(xy,list) or len(xy)!=2:raise ValueError(f'points.{key}: [x,y] required')
        q=[number(v,'points.'+key) for v in xy]
        if any(not 0<=q[i]<=size[i]-1 for i in (0,1)):raise ValueError(f'points.{key}: outside current image {size}')
        pts[key]=q
    rows=[];seen=set();used=set()
    for p in paths:
        pid=ident(p.get('id'))
        if pid in seen:raise ValueError('Duplicate path ID '+pid)
        seen.add(pid);segs=[];last=None;first=None
        for index,seg in enumerate(p.get('segments',[])):
            kind=seg.get('kind');ids=seg.get('points')
            if kind not in ('line','curve') or not isinstance(ids,list) or len(ids)<2:raise ValueError(pid+': segment requires kind line/curve and >=2 point IDs')
            if any(k not in pts for k in ids):raise ValueError(pid+': unknown point ID')
            if last is not None and ids[0]!=last:raise ValueError(f'{pid}.segments.{index}: starts at {ids[0]}, but previous segment ends at {last}; share one endpoint ID. A dashed shaft is one continuous underlying centerline, not separate ink dashes.')
            first=ids[0] if first is None else first;last=ids[-1]
            for a,b in zip(ids,ids[1:]):
                if math.dist(pts[a],pts[b])<1e-6:raise ValueError(pid+': degenerate consecutive points')
            used.update(ids);segs.append({'kind':kind,'points':ids})
        if not segs:raise ValueError(pid+': nonempty segments required')
        for name,node in [('start_head',first),('end_head',last)]:
            if type(p.get(name)) is not bool:raise ValueError(pid+': '+name+' must be boolean')
        rows.append({'id':pid,'segments':segs,'start_head':p['start_head'],'end_head':p['end_head']})
    # Headless diagram connectors are valid. Their identity is a visual decision,
    # not a reason to manufacture heads to satisfy a topological schema rule.
    return {'status':'ready','points':{k:pts[k] for k in pts if k in used},'paths':rows,'uncertain':raw.get('uncertain',[])}


def review(raw,ids=()):
    if raw.get('verdict') not in ('accept','revise','uncertain'):raise ValueError('verdict must be accept/revise/uncertain')
    out={'verdict':raw['verdict']}
    if raw['verdict']!='accept':
        for k in ('evidence','suggestion'):
            if not isinstance(raw.get(k),str) or not raw[k].strip():raise ValueError(k+' required for non-accept')
            out[k]=raw[k]
    return out


def geometry_review(raw,geom=None):
    """Validate visual findings before the verdict; preserve the refine contract."""
    if not isinstance(raw,dict):raise ValueError('Geometry review must be a JSON object')
    fields=('off_points','endpoint_issues','missing_sections')
    lists={}
    point_ids=set(geom['points']) if geom is not None else None
    endpoints={p['segments'][0]['points'][0] for p in geom['paths']} | {p['segments'][-1]['points'][-1] for p in geom['paths']} if geom is not None else None
    def text(value,field):
        if not isinstance(value,str) or not value.strip():raise ValueError(field+' must contain nonempty image evidence or an actionable correction')
        return value.strip()
    for field in fields:
        values=raw.get(field)
        if not isinstance(values,list):raise ValueError(field+' is required and must be an array; use [] when no issue is visible')
        rows=[]
        for index,row in enumerate(values):
            prefix=f'{field}.{index}'
            if not isinstance(row,dict):raise ValueError(prefix+' must be an object')
            item={'visible':text(row.get('visible'),prefix+'.visible')}
            if field=='missing_sections':
                item['fix']=text(row.get('fix'),prefix+'.fix')
            else:
                key=text(row.get('id'),prefix+'.id')
                if point_ids is not None and key not in point_ids:
                    raise ValueError(prefix+'.id '+repr(key)+' is not a sample point ID; use one of '+', '.join(sorted(point_ids)))
                if field=='endpoint_issues' and endpoints is not None and key not in endpoints:
                    raise ValueError(prefix+'.id '+repr(key)+' is not a path endpoint; use one of '+', '.join(sorted(endpoints)))
                item={'id':key,**item,'move':text(row.get('move'),prefix+'.move')}
            rows.append(item)
        lists[field]=rows
    reason=text(raw.get('reason'),'reason')
    verdict=raw.get('verdict')
    if verdict not in ('accept','revise','uncertain'):raise ValueError('verdict must be accept/revise/uncertain')
    found=any(lists.values())
    if verdict=='accept' and found:raise ValueError('verdict accept conflicts with reported issues; use revise, or correct findings only if the images support it')
    if verdict=='revise' and not found:raise ValueError('verdict revise requires at least one visible issue in off_points, endpoint_issues or missing_sections')
    evidence='; '.join(f'{field}: {json.dumps(rows,ensure_ascii=False)}' for field,rows in lists.items() if rows) or reason
    changes=[f'{row["id"]}: {row["move"]}' for field in ('off_points','endpoint_issues') for row in lists[field]]
    changes += [row['fix'] for row in lists['missing_sections']]
    suggestion='; '.join(changes) if changes else ('Resolve the source-visibility question before revising geometry: '+reason if verdict=='uncertain' else '')
    return {**lists,'reason':reason,'evidence':evidence,'suggestion':suggestion,'verdict':verdict}


def styles(raw,paths,mode):
    field='styles' if mode=='shaft' else 'heads';rows=raw.get(field)
    if not isinstance(rows,list):raise ValueError(field+' array required')
    expected={p['id']:p for p in paths};out=[];seen=set()
    for row in rows:
        key=row.get('id')
        if key not in expected or key in seen:raise ValueError('Unknown/duplicate style ID '+str(key))
        seen.add(key);r={'id':key}
        if mode=='shaft':
            r.update(stroke=color(row.get('stroke')),width_px=number(row.get('width_px'),'width_px',True),dash=row.get('dash','solid'))
            if r['dash'] not in ('solid','dash','dot','dash_dot'):raise ValueError('Invalid dash pattern')
            values=row.get('dash_pattern_px')
            if values is not None and r['dash']!='solid':
                if not isinstance(values,list) or len(values) not in (2,4,6):raise ValueError(key+': dash_pattern_px must be a JSON ARRAY of 2, 4 or 6 positive numbers, e.g. [12,8], not a string; preserve your estimated lengths')
                r['dash_pattern_px']=[number(v,'dash_pattern_px',True) for v in values]
        else:
            for end in ('start','end'):
                h=row.get(end)
                if not expected[key][end+'_head']:
                    if h is not None:raise ValueError(key+': '+end+' has no observed head; keep null')
                    r[end]=None;continue
                if not isinstance(h,dict) or h.get('type') not in ('triangle','stealth','arrow','diamond','oval'):raise ValueError(key+': invalid '+end+' head type')
                if any(h.get(k) not in ('sm','med','lg') for k in ('width','length')):raise ValueError('Head width/length must be sm/med/lg')
                r[end]={k:h[k] for k in ('type','width','length')}
        out.append(r)
    if seen!=set(expected):raise ValueError('Missing paths: '+','.join(set(expected)-seen))
    return {field:out}
