"""Three endpoint categories, fixed native med/med, source-pixel width floor."""
import copy,math
import style_contract as legacy

VERSION='style_three_heads_mid_min2.0'
MINIMUM_LINE_WIDTH_PX=2.
MINIMUM_BLOCK_OUTLINE_WIDTH_PX=1.
PROFILES={'three_heads_mid_min1','three_heads_mid_min2'}
HEADS={'none':None,'circle':'oval','arrow':'triangle'}

def category(head):
    if not head or head.get('type') in ('none','unknown'):return 'none'
    return 'circle' if head.get('type')=='oval' else 'arrow'

def endpoint(value):
    if not isinstance(value,str) or value not in HEADS:raise ValueError('Endpoint must be none, circle or arrow')
    return None if HEADS[value] is None else dict(type=HEADS[value],width='med',length='med')

class Contract:
    geometry_signature=staticmethod(legacy.geometry_signature)
    def __init__(self,view=None):
        matrix=(view or {}).get('view_to_global',[[1,0,0],[0,1,0],[0,0,1]])
        self.to_source=math.sqrt(abs(matrix[0][0]*matrix[1][1]))
        if not math.isfinite(self.to_source) or self.to_source<=0:raise ValueError('Invalid style scale')
        self.minimum=MINIMUM_LINE_WIDTH_PX/self.to_source

    def clamp(self,width,minimum_source_px=MINIMUM_LINE_WIDTH_PX):
        if type(width) not in (int,float) or not math.isfinite(width):raise ValueError('width_px must be finite')
        return max(float(width),minimum_source_px/self.to_source)

    def validate_style(self,value,row):
        value=copy.deepcopy(value);audit=[]
        minimum=MINIMUM_BLOCK_OUTLINE_WIDTH_PX if row['kind']=='block_arrow' else MINIMUM_LINE_WIDTH_PX
        if row['kind']=='block_arrow':
            for part in value.get('parts',[]):
                if part.get('stroke') is not None:
                    raw=part.get('width_px');part['width_px']=self.clamp(raw,minimum)
                    audit.append(dict(part=part.get('id'),raw_local_px=raw,effective_local_px=part['width_px'],raw_source_px=raw*self.to_source,effective_source_px=max(minimum,part['width_px']*self.to_source)))
        else:
            raw=value.get('width_px');value['width_px']=self.clamp(raw)
            audit.append(dict(raw_local_px=raw,effective_local_px=value['width_px'],raw_source_px=raw*self.to_source,effective_source_px=max(minimum,value['width_px']*self.to_source)))
            value['start']=endpoint(value.get('start'));value['end']=endpoint(value.get('end'))
        unlocked=copy.deepcopy(row);unlocked.pop('geometry_heads_reported',None)
        result=legacy.validate_style(value,unlocked)
        result['policy']=dict(version=VERSION,minimum_source_width_px=minimum,head_size='med/med',widths=audit)
        return result

    def validate(self,raw,rows):
        values=raw.get('styles') if isinstance(raw,dict) else None;table={r['id']:r for r in rows}
        if not isinstance(values,list) or len(values)!=len(table) or any(not isinstance(v,dict) for v in values) or {v.get('id') for v in values}!=set(table):
            raise ValueError('Return each supplied style ID exactly once')
        for v in values:self.validate_style(v,table[v['id']])
        return raw

    def checks(self,raw,rows):
        result=legacy.checks(raw,rows)
        if any(c.get('geometry_action','keep')!='keep' for c in result['checks']):raise ValueError('Style critic cannot replace geometry')
        return result

    def wire(self,row):
        value=legacy.wire(row)
        if row['kind']!='block_arrow':
            value['start']=category(row['style'].get('start'));value['end']=category(row['style'].get('end'))
            value.pop('evidence_status',None)
        return value

    def enforce_rows(self,rows):
        result=copy.deepcopy(rows)
        for r in result:
            previous=r['style'];validated=self.validate_style(self.wire(r),r)
            r['style']={**previous,**validated,'state':previous.get('state','geometry_preview')}
            if r['kind']!='block_arrow':
                r['heads']=[r['style'][e]['type'] if r['style'][e] else 'none' for e in ('start','end')]
                for e in ('start','end'):
                    r[e+'_head']=r['style'][e] is not None;r[e+'_head_type']=r['style'][e]['type'] if r['style'][e] else None
        return result

    def prompt(self,rows,size,mode='estimate',feedback=None):
        targets=[]
        for r in rows:
            t=dict(id=r['id'],kind=r['kind'],start=r['start'],end=r['end'])
            if r['kind']=='block_arrow':t['part_ids']=[p['id'] for p in r['shape_parts']]
            targets.append(t)
        critic=mode=='critic'
        include_block_rules=not critic or any(r['kind']=='block_arrow' for r in rows)
        return dict(protocol=VERSION,
            task='Check the rendered styles against CLEAN, especially each endpoint category.' if critic else 'Recover styles from CLEAN for these fixed paths/shapes. Verify BOTH endpoint categories.',
            size=list(size),targets=targets,current_styles=[self.wire(r) for r in rows],feedback=feedback,
            instructions=[
                'CLEAN is the source; LOCATORS only identifies paths and physical start/end. Previous head labels may be wrong: correct them from the visible endpoint marks. Path order does not imply direction.',
                'Only three endpoint categories: none (no terminal mark), circle (a distinct round terminal mark), arrow (an ordinary directional arrowhead). Judge the visible mark attached to each specified physical endpoint, separately at both ends. A small filled or outlined round mark terminating the target stroke is circle even when it touches or overlaps the border of a larger diagram node. Its proximity or attachment to a node is not a reason to remove it. A larger diagram node reached by a line without a separate terminal mark, a plain round line cap, or a crossing alone is not a head. Distinguish a separate terminal mark from the surrounding node using CLEAN; preserve real double-headed paths.',
                'Head width and length are fixed to mid (native med/med). Do not estimate, output or critique head dimensions or fine arrowhead subtypes.',
                f'Estimate shaft color, width_px and dash_px in CLEAN pixels on a representative shaft. Backend clamps ordinary line-arrow shaft widths to at least {MINIMUM_LINE_WIDTH_PX:g} ORIGINAL-image pixels; do not request refinement to undo this floor. Head dimensions also must not be compensated by changing shaft width.',
                *([f'Closed block-arrow heads are part of their fixed contours: recover per-part fill/stroke/dash only, never add endpoint markers. Their existing outline rule remains {MINIMUM_BLOCK_OUTLINE_WIDTH_PX:g} ORIGINAL-image pixel minimum for a visible stroke; no stroke stays absent. Ignore acceptable curved-outline differences; do not change geometry or select a mask.'] if include_block_rules else []),
                'Review only clear endpoint category/direction or color/width/dash differences. Accept intentional fixed head sizes and the width floor. Geometry errors are separate; leave paths unchanged. Return compact JSON.' if critic else 'Change style and endpoint category only. Keep all paths, coordinates and shape contours unchanged. Return compact JSON.'
            ],render_policy=dict(head_size='med/med',line_minimum_original_width_px=MINIMUM_LINE_WIDTH_PX,line_minimum_CLEAN_width_px=self.minimum,
                                 **(dict(block_outline_minimum_original_width_px=MINIMUM_BLOCK_OUTLINE_WIDTH_PX,
                                         block_outline_minimum_CLEAN_width_px=MINIMUM_BLOCK_OUTLINE_WIDTH_PX/self.to_source) if include_block_rules else {})),
            output_schema={'checks':'[{id,verdict:accept|revise|uncertain|geometry_issue,reason:short or empty for accept}]'} if critic else
                {'styles':'Line: {id,color:#RRGGBB,width_px,dash_px:[] or 2/4/6 alternating mark/gap lengths,start:none|circle|arrow,end:none|circle|arrow}. Block: {id,parts:[{id,fill:null|#RRGGBB|{type:linear_gradient,start_color,end_color,angle_deg},stroke:null|#RRGGBB,width_px,dash:solid|dash|dot|dash_dot,dash_pattern_px:optional array}]}. Return {styles:[...]}.'})

def merge_style(original,styled):
    """Commit only style/head metadata, retaining exact original geometry floats."""
    if original['id']!=styled['id']:raise ValueError('Style ID mismatch')
    result=copy.deepcopy(original)
    for k in ['style','heads','start_head','end_head','start_head_type','end_head_type','line_style',
              'head_recovery','pre_head_style_review']:
        if k in styled:result[k]=copy.deepcopy(styled[k])
    result['head_source']=styled.get('head_source','style_three_category_review')
    return result
