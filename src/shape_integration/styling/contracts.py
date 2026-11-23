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
    # Measured fields are applied by the controller; the model may omit them.
    fixed = set(part.get('_fixed', ()))
    if not stroke: width = 0.
    elif 'width_px' in fixed and row.get('width_px') is None: width = None
    else: width = number(row.get('width_px'),'width_px',True)
    dash = row.get('dash','solid') if stroke else 'solid'
    if dash not in ('solid','dash','dot','dash_dot'):
        if 'dash' not in fixed: raise ValueError('dash: unsupported pattern')
        dash = 'solid'
    result = {'fill':f,'stroke':stroke,'width_px':width,'dash':dash}
    if 'fill_opacity' in row:
        opacity=number(row['fill_opacity'],'fill_opacity')
        if not 0<=opacity<=1 or f is None and opacity!=1:
            raise ValueError('fill_opacity: expected 0..1 on a filled closed face')
        result['fill_opacity']=opacity
    pattern = row.get('dash_pattern_px')
    if stroke and dash!='solid' and pattern is not None and 'dash_pattern_px' not in fixed:
        if not isinstance(pattern,list) or len(pattern) not in (2,4,6):
            raise ValueError('dash_pattern_px: expected 2, 4 or 6 alternating lengths')
        result['dash_pattern_px'] = [number(v,'dash_pattern_px',True) for v in pattern]
    if row.get('issue'): result['issue'] = str(row['issue'])[:2000]
    return result


def to_source_pixels(value, source_to_crop_scale):
    """Normalize one validated model style once; rule styles never pass here."""
    scale=number(source_to_crop_scale,'source_to_crop_scale',True)
    result=dict(value)
    result['width_px']=value['width_px']/scale if value['width_px'] is not None else None
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
        allowed=({'id','fill','fill_opacity','stroke','width_px','dash','dash_pattern_px','issue','behind'} if mode in ('select','refine') else
                 {'id','verdict','source_evidence','rendered_difference','suggestion','affected_fields','expected_line_pattern','expected_stroke_width','behind','in_front_of'} if mode=='critic' else
                 {'id','front','front_paint','evidence'})
        extras=sorted(set(v)-allowed)
        if isinstance(v.get('fill'),dict):
            extras.extend('fill.'+k for k in sorted(set(v['fill'])-{'type','start_color','end_color','angle_deg'}))
        if extras:discarded.append({'id':ident,'ignored_fields':extras})
        try:
            if mode in ('select','refine'):
                out=style(v,part_table[ident])
                if 'behind' in v:
                    behind=v['behind']
                    if not isinstance(behind,list) or any(not isinstance(k,str) or k not in part_table or k==ident for k in behind):
                        raise ValueError('behind: expected other visible target IDs')
                    out['behind']=list(dict.fromkeys(behind))
            elif mode=='critic':
                verdict=v.get('verdict')
                if verdict not in ('accept','revise','uncertain'): raise ValueError('verdict must be accept/revise/uncertain')
                out={'verdict':verdict}
                if verdict!='accept':
                    for k in ('source_evidence','rendered_difference','suggestion'):
                        if not isinstance(v.get(k),str) or not v[k].strip(): raise ValueError(k+': required for non-accept')
                        out[k]=v[k]
                    affected=v.get('affected_fields', [])
                    if not isinstance(affected,list) or any(k not in ('fill','stroke_presence','stroke_color','stroke_width','line_pattern','gradient','occlusion') for k in affected):
                        raise ValueError('affected_fields: unsupported style field')
                    out['affected_fields']=affected
                    if 'expected_line_pattern' in v:
                        pattern=v['expected_line_pattern']
                        if verdict!='revise' or 'line_pattern' not in affected or pattern not in ('solid','dash','dot','dash_dot'):
                            raise ValueError('expected_line_pattern: requires a supported line_pattern revision')
                        out['expected_line_pattern']=pattern
                    if 'expected_stroke_width' in v:
                        direction=v['expected_stroke_width']
                        if verdict!='revise' or 'stroke_width' not in affected or direction not in ('increase','decrease'):
                            raise ValueError('expected_stroke_width: requires a supported stroke_width revision')
                        out['expected_stroke_width']=direction
                # Order is independent of colour/width acceptance. Retain a
                # source-supported relation even when the model omitted the
                # occlusion field; the controller compares it to the render.
                if verdict!='uncertain':
                    for name in ('behind','in_front_of'):
                        order_ids=v.get(name,[])
                        if not isinstance(order_ids,list) or any(not isinstance(k,str) or k not in (part_table or {}) or k==ident for k in order_ids):
                            raise ValueError(name+': expected other labeled context IDs')
                        if order_ids:out[name]=list(dict.fromkeys(order_ids))
                    if set(out.get('behind',[]))&set(out.get('in_front_of',[])):
                        raise ValueError('Contradictory occlusion directions')
                    has_order=bool(out.get('behind') or out.get('in_front_of'))
                    if verdict=='revise' and 'occlusion' in out.get('affected_fields',[]) and not has_order:
                        raise ValueError('occlusion revision requires a source-supported behind or in_front_of relation')
                    if has_order and isinstance(v.get('source_evidence'),str):out['source_evidence']=v['source_evidence']
            else:
                front=v.get('front'); pair=pair_table[ident]
                if front is not None and front not in pair: raise ValueError('front must be one of this pair IDs or null')
                if not isinstance(v.get('evidence'),str) or not v['evidence'].strip(): raise ValueError('evidence is required')
                out={'front':front,'evidence':v['evidence']}
                if 'front_paint' in v:
                    if v['front_paint'] not in ('unfilled','translucent','opaque','unknown'):
                        raise ValueError('front_paint: unsupported paint relation')
                    out['front_paint']=v['front_paint']
            good[ident]={'id':ident,**out}
        except (ValueError,TypeError) as exc: errors[ident]=str(exc)
    for ident in expected:
        if ident not in good and ident not in errors: errors[ident]='Missing row'
    return {field:list(good.values()),'_errors':errors,'_discarded':discarded,
            '_source_rows':values}
