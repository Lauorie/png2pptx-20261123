"""Style-only wire format. Endpoints are stable path IDs, never A labels."""
import copy
import math
from shape_core.styling import contracts as shape_contract
import protocol

SIZES=('sm','med','lg')


def geometry_signature(rows):
    fields=('id','kind','start','via','end','beziers','contours','shape_parts','node_ids','shared_nodes')
    return [{k:copy.deepcopy(r[k]) for k in fields if k in r} for r in rows]


def head(value):
    if value is None:return None
    if not isinstance(value,dict) or value.get('type') not in set(protocol.HEADS)-{'none','unknown'}:
        raise ValueError('Head must be null or a native head type')
    if value.get('width') not in SIZES or value.get('length') not in SIZES:
        raise ValueError('Head width/length must be sm, med or lg')
    return {k:value[k] for k in ('type','width','length')}


def validate_style(value,row):
    if row['kind']=='block_arrow':
        parts={p['id']:p for p in row['shape_parts']}
        raw=value.get('parts')
        if not isinstance(raw,list) or len(raw)!=len(parts) or {p.get('id') for p in raw}!=set(parts):
            raise ValueError('Return one shape style per part ID')
        styles={p['id']:shape_contract.style(p,parts[p['id']]) for p in raw}
        # Legacy keys remain only for existing overview/export consumers.
        first=next(iter(styles.values()));fill=first['fill']
        color=fill if isinstance(fill,str) else fill['start_color'] if fill else first['stroke'] or '#808080'
        return dict(id=row['id'],color=color,width_px=first['width_px'],dash=dict(type='solid',preset='solid'),
                    start=None,end=None,shape_parts=styles,source='glm_style',state='estimated')
    color=shape_contract.color(value.get('color'))
    width=shape_contract.number(value.get('width_px'),'width_px',True)
    dash=value.get('dash_px')
    if not isinstance(dash,list) or len(dash) not in (0,2,4,6):
        raise ValueError('dash_px is [] for solid or 2/4/6 alternating mark/gap lengths')
    dash=[shape_contract.number(x,'dash_px',True) for x in dash]
    if row.get('geometry_heads_reported'):
        from geometry_heads import category
        for end,old in zip(('start','end'),row['heads']):
            new=head(value[end]);new_type=new['type'] if new else 'none'
            if old!='unknown' and category(new_type)!=category(old):
                raise ValueError('Style cannot change geometry endpoint presence/direction/category at '+end)
    return dict(id=row['id'],color=color,width_px=width,
                dash=dict(type='dashed' if dash else 'solid',preset='dash' if dash else 'solid',pattern_px=dash,
                          **(dict(mark_px=dash[0],gap_px=dash[1]) if len(dash)==2 else {})),
                start=head(value['start']),end=head(value['end']),source='glm_style',state='estimated')


def validate(raw,rows):
    table={r['id']:r for r in rows};values=raw.get('styles') if isinstance(raw,dict) else None
    if not isinstance(values,list) or len(values)!=len(table) or {v.get('id') for v in values}!=set(table):
        raise ValueError('styles must contain each supplied path ID exactly once')
    # Return wire values to keep cache validation idempotent.
    for value in values:validate_style(value,table[value['id']])
    return raw


def checks(raw,rows):
    table={r['id']:r for r in rows};values=raw.get('checks') if isinstance(raw,dict) else None
    if not isinstance(values,list) or len(values)!=len(table) or {v.get('id') for v in values}!=set(table):
        raise ValueError('checks must contain each supplied path ID once')
    for v in values:
        if v.get('verdict') not in ('accept','revise','uncertain','geometry_issue'):
            raise ValueError('Unknown style review verdict')
        if v['verdict']!='accept' and (not isinstance(v.get('reason'),str) or not v['reason'].strip()):
            raise ValueError('Explain the visible difference briefly')
        if v.get('geometry_action','keep') not in ('keep','mask'):raise ValueError('geometry_action must be keep or mask')
        if v.get('geometry_action')=='mask' and (table[v['id']]['kind']!='block_arrow' or not table[v['id']].get('mask_shape_parts')):
            raise ValueError('Mask geometry replacement is available only for a block arrow with preserved mask evidence')
    return raw


def wire(row):
    s=row['style']
    if row['kind']=='block_arrow':
        import shape_render
        return dict(id=row['id'],parts=[dict(id=k,**v) for k,v in shape_render.part_styles(row).items()])
    return dict(id=row['id'],color=s.get('color'),width_px=s.get('width_px'),
                dash_px=s.get('dash',{}).get('pattern_px') or
                    ([s['dash']['mark_px'],s['dash']['gap_px']] if s.get('dash',{}).get('mark_px') and s['dash'].get('gap_px') else []),
                start=s.get('start'),end=s.get('end'),
                evidence_status=s.get('state','pixel_estimate'))


def prompt(rows,size,mode='estimate',feedback=None):
    targets=[]
    for r in rows:
        target=dict(id=r['id'],kind=r['kind'],start=r['start'],end=r['end'])
        if r.get('geometry_heads_reported'):target['fixed_endpoint_semantics']=r['heads']
        if r['kind']=='block_arrow':
            from shape_core.styling.geometry import bounds
            target['part_ids']=[p['id'] for p in r['shape_parts']]
            target['parts']=[dict(id=p['id'],bbox=bounds(p)) for p in r['shape_parts']]
        targets.append(target)
    review=mode=='critic'
    return dict(task=('Compare style of the actual rendered PPT with CLEAN; geometry is fixed.' if review else
                      'Estimate visible styles for these fixed paths/shapes from CLEAN.'),size=list(size),targets=targets,
        current_styles=[wire(r) for r in rows],feedback=feedback,
        instructions=[
            'CLEAN is source truth. LOCATORS identifies paths; its colors and labels are annotations. RENDER is the actual PPT on the identical pixel grid. Compare only the named paths, not missing text or unrelated shapes.',
            'All dimensions are pixels in CLEAN. Estimate shaft color, width and dash rhythm on a clean representative portion, excluding the head, junctions and occlusion. A gap between dash marks is not missing geometry. Previous estimates and mask measurements may be wrong.',
            'start/end refer to the supplied physical path endpoints in this order, never to logical flow. null means no head. triangle: filled, flat rear, no notch; stealth: filled, inward rear notch, pointed rear wings; arrow: open V, two constant-width strokes. diamond/oval are available. Native width and length each use sm/med/lg.',
            'Closed block arrows use shape parts with fill (null, #RRGGBB, or linear_gradient), stroke (null or #RRGGBB), width_px, dash and optional dash_pattern_px. Their contour already contains the head: never attach a shaft head.',
            ('Report accept/revise/uncertain/geometry_issue per ID with a short visible reason for non-accept. Do not accept because upstream said correct. Geometry errors are separate; never disguise them by enlarging heads or line widths. For a block arrow only, geometry_action=mask may select the supplied mask contour when it is visibly better than the shape; otherwise keep.' if review else
            'Change style only. Never add/delete/move a path, infer semantic directions, or repair geometry with style. When fixed_endpoint_semantics is supplied, preserve none versus directional head versus round terminal at each ordered end; refine only dimensions or fine directional-head shape. Return the compact requested JSON; no long explanation.')
        ],output_schema=({'checks':'[{id,verdict:accept|revise|uncertain|geometry_issue,reason,geometry_action:keep|mask}]'} if review else
            {'styles':'Line: {id,color:#RRGGBB,width_px,dash_px:[] or alternating mark/gap lengths,start:null or {type,width:sm|med|lg,length:sm|med|lg},end:same}. Block: {id,parts:[{id,fill:null|#RRGGBB|{type:linear_gradient,start_color,end_color,angle_deg},stroke:null|#RRGGBB,width_px,dash:solid|dash|dot|dash_dot,dash_pattern_px:optional array}]}. Return {styles:[...]}.'}))


def provisional(image,mask,row):
    """Retain pixel evidence on FINAL geometry, never an old model shaft sample."""
    import numpy as np
    import style_windows
    clean=copy.deepcopy(row);clean.pop('shaft_sample',None)
    try:
        s=style_windows.recover(np.asarray(image),mask,clean)
        s['state']='pixel_estimate'
        import geometry_heads
        return geometry_heads.provisional(s,row)
    except (ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
        import geometry_heads
        return geometry_heads.provisional(dict(color='#808080',width_px=1.5,dash=dict(type='unknown',preset=None),
                    start=None,end=None,state='geometry_only',measurement_error=str(exc)),row)


def exportable(style):
    """Explicitly visible provisional ink; never silently erase recovered geometry."""
    s=copy.deepcopy(style)
    if not s.get('color'):s['color']='#808080'
    if not s.get('width_px'):s['width_px']=1.5
    if s.get('dash',{}).get('preset') is None and s.get('dash',{}).get('type')=='unknown':
        s.update(color='#808080',state='geometry_only')
        s['dash']=dict(type='solid',preset='solid')
    return s
