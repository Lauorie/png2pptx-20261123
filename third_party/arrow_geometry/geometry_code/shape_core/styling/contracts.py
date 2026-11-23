"""Small style contracts; independent rows survive unrelated field failures."""
import math
import re
from collections import Counter
from .geometry import closed


def color(value):
    if not isinstance(value,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',value):
        raise ValueError('must be #RRGGBB or null')
    return value.upper()


def number(value, name, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise ValueError(name+': must be a finite '+('positive ' if positive else '')+'number')
    return float(value)


def fill(value):
    if value is None: return None
    if isinstance(value,str): return color(value)
    if not isinstance(value,dict) or value.get('type')!='linear_gradient':
        raise ValueError('fill: expected null, color or linear_gradient')
    return {'type':'linear_gradient','start_color':color(value.get('start_color')),
            'end_color':color(value.get('end_color')),
            'angle_deg':number(value.get('angle_deg'),'angle_deg')%360}


def style(row, part, candidates=None):
    # Accept the legacy argument for callers, but palette entries are references.
    # Geometry, color syntax and numeric/style constraints still apply.
    if 'fill' not in row or 'stroke' not in row: raise ValueError('fill and stroke are required')
    f = fill(row['fill']); stroke = color(row['stroke']) if row['stroke'] is not None else None
    if f is not None and not closed(part): raise ValueError('fill: open path must remain unfilled')
    width = number(row.get('width_px'),'width_px',True) if stroke else 0.
    dash = row.get('dash','solid') if stroke else 'solid'
    if dash not in ('solid','dash','dot','dash_dot'): raise ValueError('dash: unsupported pattern')
    result = {'fill':f,'stroke':stroke,'width_px':width,'dash':dash}
    pattern = row.get('dash_pattern_px')
    if stroke and dash!='solid' and pattern is not None:
        if not isinstance(pattern,list) or len(pattern) not in (2,4,6):
            raise ValueError('dash_pattern_px: expected 2, 4 or 6 alternating lengths')
        result['dash_pattern_px'] = [number(v,'dash_pattern_px',True) for v in pattern]
    if row.get('issue'): result['issue'] = str(row['issue'])[:2000]
    return result


def to_source_pixels(value, source_to_crop_scale):
    """Normalize one validated model style once; rule styles never pass here."""
    scale=number(source_to_crop_scale,'source_to_crop_scale',True)
    result=dict(value)
    result['width_px']=value['width_px']/scale
    if 'dash_pattern_px' in value:
        result['dash_pattern_px']=[v/scale for v in value['dash_pattern_px']]
    return result


def rows(raw, expected, mode, part_table=None, candidates=None, pair_table=None):
    """Return canonical raw rows for Runtime cache revalidation plus validation audit."""
    field = {'select':'styles','refine':'styles','critic':'checks','relations':'relations'}[mode]
    # Runtime revalidates its canonical cached result. Retain the original rows so
    # precise retry feedback and request identity stay identical on resume.
    values = raw.get('_source_rows',raw.get(field,[])) if isinstance(raw,dict) else []
    if not isinstance(values,list): values=[]
    counts=Counter(v.get('id') for v in values if isinstance(v,dict) and isinstance(v.get('id'),str))
    good={}; errors={}; discarded=[]
    for v in values:
        if not isinstance(v,dict): discarded.append({'error':'row is not an object'}); continue
        ident=v.get('id')
        if not isinstance(ident,str) or ident not in expected:
            discarded.append({'error':'unknown ID','row':v}); continue
        if counts[ident]!=1: errors[ident]='Duplicate ID'; continue
        allowed=({'id','fill','stroke','width_px','dash','dash_pattern_px','issue'} if mode in ('select','refine') else
                 {'id','verdict','source_evidence','rendered_difference','suggestion'} if mode=='critic' else
                 {'id','front','evidence'})
        extras=sorted(set(v)-allowed)
        if isinstance(v.get('fill'),dict):
            extras.extend('fill.'+k for k in sorted(set(v['fill'])-{'type','start_color','end_color','angle_deg'}))
        if extras:discarded.append({'id':ident,'ignored_fields':extras})
        try:
            if mode in ('select','refine'):
                out=style(v,part_table[ident])
            elif mode=='critic':
                verdict=v.get('verdict')
                if verdict not in ('accept','revise','uncertain'): raise ValueError('verdict must be accept/revise/uncertain')
                out={'verdict':verdict}
                if verdict!='accept':
                    for k in ('source_evidence','rendered_difference','suggestion'):
                        if not isinstance(v.get(k),str) or not v[k].strip(): raise ValueError(k+': required for non-accept')
                        out[k]=v[k]
            else:
                front=v.get('front'); pair=pair_table[ident]
                if front is not None and front not in pair: raise ValueError('front must be one of this pair IDs or null')
                if not isinstance(v.get('evidence'),str) or not v['evidence'].strip(): raise ValueError('evidence is required')
                out={'front':front,'evidence':v['evidence']}
            good[ident]={'id':ident,**out}
        except (ValueError,TypeError) as exc: errors[ident]=str(exc)
    for ident in expected:
        if ident not in good and ident not in errors: errors[ident]='Missing row'
    return {field:list(good.values()),'_errors':errors,'_discarded':discarded,
            '_source_rows':values}
