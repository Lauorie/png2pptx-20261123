"""Small visual tasks sharing the integration Runtime and its global limiter."""
import copy
from ..common import save, read
from ..transport import RequestTimeout
from .contracts import rows, number, to_source_pixels

SYSTEM='Recover styles of existing editable geometry from the supplied source pixels. Geometry is fixed. Return one JSON object only.'
PROTOCOL='shape_integration.style.explicit_order.13'
# Rules shared by every style role. Role-specific rules follow; no role receives
# instructions about fields it cannot output.
COMMON=[
    'overlap_context只提供前后顺序，不决定填充。填充和透明度只由本次CLEAN/ACTUAL对比判断。'
    '白色与透明不能按名称或SAM掩码区分：掩码只提取到黑边也可能是白色不透明面。'
    '白色背景本身不是白色填充的证据。分组边框没有独立填充或遮挡证据时用fill:null。'
    '检查填充是否错误遮住原图可见的节点、线条或图案；实际遮住后方内容的面保留不透明填充。',
    'Match each ID to its contour before reading colour. TARGET DETAILS shows one target per card; '
    'use its CLEAN pixels, not a nearby node or the annotation colour. A triangle and an ellipse are different targets.',
    'A cylinder primitive includes the complete body and top rim seam. Its matching top ellipse '
    'is an independently styled face ABOVE that body; preserve its source-supported '
    'fill and order without adding a second outline where the native body already supplies the rim. '
    'Choose the actual top pigment, including the same pigment as the body for a uniformly coloured '
    'cylinder. Do not accept automatic Office lightening as source evidence or simulate it with a gradient. '
    'Text backing is decided at composition. Do not imitate a label knockout by recoloring a frame, '
    'adding a white face or changing its geometry; infer the underlying stroke through the label.',
    'Read appearance from CLEAN source; artificial target contours, IDs and IMG boxes are identification only. IMG assets and text are not target shape styles.',
    'A closed path can be an unfilled stroke centerline. The contour does not imply solid fill. Do not fill hollow loops or add an outline to a color boundary without evidence.',
    'Opaque white and transparent null differ. Preserve white fill where a foreground face hides '
    'rear edges in CLEAN; keep genuinely hollow loops and transparent frames unfilled.',
    'Infer occluded appearance only from visible portions or matching instances; exclude neighboring contamination.',
    'Fill may be null, #RRGGBB, or {type:linear_gradient,start_color:#RRGGBB,end_color:#RRGGBB,angle_deg:number}. Gradient has exactly two colors; 0 degrees points right, 90 down in the source image. Do not infer a gradient merely because a part depicts 3D.',
    'Filled faces may optionally have fill_opacity in [0,1], default 1. Use translucency only when CLEAN shows rear contours through the face. Choose base fill pigment and opacity together: displayed overlapping pixels are composites, not independent cell colors. Keep the face border separate. Do not invent per-cell gradients or recolor a rear grid to simulate an overlapping transparent face.',
]
ESTIMATOR=[
    'behind 的方向只有一种：A 的 behind:[B] 表示先画 A、再画 B，B 遮住 A。不能把 A 后面的对象填入 behind。'
    'For overlapping parts of the SAME compound, optional behind:[target IDs] records visible back-to-front order. '
    'A drop shadow sits behind its body; stacked faces follow source occlusion. behind may reference any labeled context ID, including other batches, but return styles only for targets. Keep visible internal nodes and their edges in front of their own enclosing fill. behind never means inside. Omit when unsupported; do not reorder from names alone.',
    'Use the complete compound as visual context. Return one semantic style for each ID in targets; context IDs have no output record. Match FILL, separate STROKE presence/color and gradient direction from CLEAN. similar_components are fallible grouping hints: preserve each target\'s own appearance.',
    'Propose fill and separate stroke presence/color. Omit width_px, dash and dash_pattern_px: '
    'the native fitter measures widths/dash lengths, and visual review resolves pattern conflicts. '
    'Preserve source-supported light colours; an unsupported stroke colour may use a repeated local border core.',
    'A small filled dot or notehead is often a filled silhouette with no separate stroke. Do not add an outline with the same color, or use a thick hollow stroke to imitate a filled dot. Preserve a real hollow center when visible.',
    'A non-solid stroke needs visible repeated gaps along the contour in CLEAN. Do not turn a continuous line into dash or dot because of antialiasing, compression or diagonal staircase artifacts.',
    'Use issue only for an unresolved representation or evidence limitation; it does not request another pass.',
]
CRITIC=[
    'If a target cannot be identified in CLEAN, return uncertain. Do not accept or revise the style of a different nearby part.',
    'When aligned SAM RGB/mask views are supplied, inspect them together with CLEAN. Mask holes, fill interiors, '
    'occlusions and crossing lines are not dash gaps. Resolve categorical conflicts from the visible source stroke.',
    'Inspect every ID in targets against the complete compound context. Check shared grid boundaries and repeated borders for inconsistent weight or color. Return no verdict for read-only context IDs; a matching neighbor does not excuse a wrong target.',
    'current_style lists the fill, stroke color and line pattern behind the supplied rendering. Judge correctness only from the source and rendered images.',
    'Accept only an observed source/render match. Revise with visible source evidence, an actual rendered discrepancy, and actionable style advice. If a visible defect cannot be assigned confidently to a target ID, use uncertain for the affected candidate targets and describe its location; uncertainty about ownership is not evidence of a match.',
    'Geometry mismatch and missing text/photos are not fixed by recoloring or changing points. When they block style judgement return uncertain and explain the limitation.',
    'Gradient colors vary over the whole fitted face, not the padded crop. Inspect source trend and rendered trend, including visible portions under occlusion.',
    'Compare thickness, dot diameter, dash lengths, gaps and repeat density along corresponding visible sections of same-scale CLEAN and actual PPT. Describe differences qualitatively; do not state pixel numbers. There is no visibility floor that excuses a visibly heavy stroke or enlarged dot.',
    'List corrected fields in affected_fields: fill, stroke_presence, stroke_color, stroke_width, '
    'line_pattern, gradient, occlusion. For occlusion identify which labeled face must be behind which other face; inspect the complete compound. Review shared_with members independently; numerical widths may differ.',
    '层序依据 CLEAN 中谁遮住谁；A behind B 表示 B 在 A 上面。与 ACTUAL PPT 不同时应标记 revise/occlusion，即使颜色和线宽正确。'
    'For an occlusion revision supply behind:[context IDs] and/or in_front_of:[context IDs] for this target. '
    'These relations are applied directly. Do not change its color to imitate another face covering it. Use uncertain if the order is not observable.',
    'For line_pattern revision state expected_line_pattern as solid, dash, dot or dash_dot from CLEAN. '
    'SAM/RGB visual review may resolve conflicting numerical categories; do not guess dash lengths.',
    'When revising stroke_width, state expected_stroke_width as increase or decrease from CLEAN. '
    'This selects a source-supported numerical width mode; never supply a numerical width.',
]


def check_uncertain_pattern(runtime, folder, images, current, measurement, options):
    """One bounded visual observation of an unresolved or conflicting pattern."""
    p={'protocol':'shape_integration.pattern_observation.1',
       'task':'Inspect the target stroke in CLEAN and the corresponding ACTUAL PPT crop. '
              'Decide its visible source line pattern and whether its rendered weight clearly differs; ignore fill, words and other paths.',
       'current_pattern':current,'numeric_evidence':{
           k:measurement.get(k) for k in ('dash','dash_reason','pattern_sam_matched_samples','pattern_sam_rgb_conflicts')},
        'rules':['The target overlay identifies one fixed path; it is not source evidence. Numeric labels are fallible.',
                'Trace corresponding visible source spans. Repeated real empty gaps are needed '
                'for dash/dot; crossings, occlusion and antialias staircases are not gaps.',
                'Inspect the aligned SAM input RGB and target mask when supplied. SAM may include '
                'fill or miss thin strokes; mask holes alone are not source dash gaps. '
                'Use unknown if the source geometry or ownership prevents a decision.',
                'If the corresponding ACTUAL rendered stroke is clearly too thin or heavy, '
                 'stroke_weight may be increase or decrease; use match when equal or unknown when unclear. '
                'Do not infer weight from a wrong contour or unrelated crossing.',
                'Return one concise JSON; no lengths, coordinates, geometry or other style edits.'],
       'output':{'source_pattern':'solid|dash|dot|dash_dot|unknown','source_evidence':'brief visible evidence',
                 'stroke_weight':'match|increase|decrease|unknown'}}
    def validate(raw):
        if not isinstance(raw,dict) or raw.get('source_pattern') not in ('solid','dash','dot','dash_dot','unknown'):
            raise ValueError('Expected one supported source_pattern')
        if not isinstance(raw.get('source_evidence'),str) or not raw['source_evidence'].strip():
            raise ValueError('Source evidence is required')
        result={k:raw[k] for k in ('source_pattern','source_evidence')}
        weight=raw.get('stroke_weight','unknown')
        if weight not in ('match','increase','decrease','unknown'):raise ValueError('Unsupported stroke_weight')
        result['stroke_weight']=weight
        if weight in ('increase','decrease'):result['expected_stroke_width']=weight
        if 'expected_stroke_width' in raw:
            if raw['expected_stroke_width'] not in ('increase','decrease'):raise ValueError('Expected increase/decrease')
            result['expected_stroke_width']=raw['expected_stroke_width']
        return result
    stage=runtime.for_stage(shape_schema_retries=0,reasoning_effort='low')
    result=stage.call(folder,'pattern_observation',
        'Resolve one target stroke from CLEAN, SAM evidence and actual rendering. Return concise JSON only.',p,images,validate,
        min(options['max_tokens'],8192))
    save(folder/'observation.json',result)
    return result


def prompt(mode, targets, units, *, feedback=None, pairs=None, color_candidates=None):
    p={'protocol':PROTOCOL,'task':mode,'targets':targets,'units':units}
    example_id=targets[0]['id'] if targets else 'P0'
    if mode in ('select','refine'):
        p['task']='Recover FILL and separate STROKE presence/color. Native SAM/CLEAN fitting measures widths and proposes a line pattern; visual review resolves conflicts.'
        p['rules']=COMMON+(ESTIMATOR if mode=='select' else ESTIMATOR[1:])+([
            'color_candidates and gradient_candidates are optional references counted from the source. Choose any valid #RRGGBB or supported two-color gradient that better matches CLEAN; avoid copying background or antialiasing colors merely because they rank highly.' ] if mode=='select' else [
            'previous_style contains semantic fill and stroke color from the supplied rendering. feedback lists critic findings. Change only semantic fields with visible evidence of a problem and return complete semantic styles. Width/pattern feedback also triggers numerical refitting even when semantic colors stay unchanged. Occlusion is applied directly from the critic; do not restate or change behind relations.'])
        if mode=='select':p['color_candidates']=list(color_candidates or [])
        p['output']={'styles':[{'id':example_id,'fill':'#RRGGBB|null|linear_gradient object','stroke':'#RRGGBB|null',
                               'fill_opacity':1,
                               'behind':[],
                               'issue':'optional unresolved limitation'}]}
        if feedback:p['feedback']=feedback
        if mode=='refine':p['output']['styles'][0].pop('behind',None)
    elif mode=='critic':
        p['task']='Check each target\'s current_style against CLEAN source and ACTUAL PPT rendering. Prioritize FILL presence/type/colors and STROKE presence/color/thickness; for non-solid strokes check dash/dot lengths, gaps and repeating rhythm. Only existing target geometry is in scope.'
        p['rules']=COMMON+CRITIC
        p['output']={'checks':[{'id':example_id,'verdict':'accept|revise|uncertain',
                              'source_evidence':'required only if non-accept','rendered_difference':'required only if non-accept',
                              'suggestion':'required only if non-accept; qualitative source-supported correction or limitation',
                              'affected_fields':['stroke_width|line_pattern|stroke_presence|stroke_color|fill|gradient|occlusion'],
                              'expected_line_pattern':'optional, only for line_pattern revision: solid|dash|dot|dash_dot'}]}
        p['output']['checks'][0]['expected_stroke_width']='optional, only for stroke_width revision: increase|decrease'
        p['output']['checks'][0].update(behind=[],in_front_of=[])
    else:
        p['task']='只根据干净原图判断重叠组件的前后顺序。颜色、填充和透明度由后续实际渲染对比统一判断，本步不作这些决定。'
        p['pairs']=pairs;p['rules']=['Do not choose based on list order, size, or semantic label alone. Use null when evidence is insufficient.',
                                    'Infer only the order visible in the source. If fitted contours cannot be reliably matched to the source, report null; do not try to solve or repair their geometry.',
                                    'Each C ID and its same-color outline identify one complete component, including all its internal parts. Do not compare or reorder parts within a component.',
                                    'Return null if the components interleave or no single whole-component order is supported. Do not output coordinates or change geometry.']
        p['output']={'relations':[{'id':'R0','front':'one of this pair component IDs or null','evidence':'原图中可见的遮挡位置；证据不足则说明'}]}
    return p


def simple_paint_review(p, table):
    """A few isolated opaque primitives need no occlusion/topology reasoning."""
    targets=p.get('targets',[])
    if (not 1<=len(targets)<=4 or p.get('overlap_context')!=[] or
            {t['id'] for t in targets}!=set(table)):
        return False
    boxes=[]
    for target in targets:
        style=target.get('current_style',{});box=target.get('source_bbox')
        if (target.get('kind') not in ('rectangle','rounded_rectangle','ellipse') or
                isinstance(style.get('fill'),dict) or style.get('fill_opacity',1)!=1 or
                not box or len(box)!=4):return False
        boxes.append(box)
    return not any(min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
                   for i,a in enumerate(boxes) for b in boxes[i+1:])


def ask(runtime, folder, mode, p, images, expected, table, options, *, candidates=None, pairs=None,source_to_crop_scale=None):
    """One first call plus one exact-row format correction. Timeouts never retry."""
    if mode in ('select','refine'):
        source_to_crop_scale=number(source_to_crop_scale,'source_to_crop_scale',True)
    # Fixed geometry plus measured pigment candidates is a bounded paint
    # selection. Keep the larger reasoning budget for topology/occlusion.
    bounded_critic=mode=='critic' and simple_paint_review(p,table)
    stage=runtime.for_stage(shape_schema_retries=0,
        **({'reasoning_effort':'low'} if mode in ('select','refine') or bounded_critic else {}))
    budget=min(options['max_tokens'],8192) if bounded_critic else options['max_tokens']
    field={'select':'styles','refine':'styles','critic':'checks','relations':'relations'}[mode]
    remaining=list(expected);good={};events=[];error_map={};previous=None
    for attempt in range(options['protocol_retries']+1):
        if not remaining:break
        actual=copy.deepcopy(p)
        if mode=='relations':actual['pairs']=[x for x in p['pairs'] if x['id'] in remaining]
        else:
            # Keep the complete visual/context evidence, but request only rows
            # that still need correction. Otherwise the original EVERY-target
            # rule contradicts the exact-row repair and repeats valid output.
            actual['targets']=[copy.deepcopy(row) for row in p['targets'] if row['id'] in remaining]
            actual['output_ids']=list(remaining)
            actual['output_scope']='Return exactly one record per output_ids entry. units, images, context_ids and read_only_targets supply context only; never add records for their other IDs.'
            if attempt:
                actual['read_only_targets']=[copy.deepcopy(row) for row in p['targets'] if row['id'] not in remaining]
        actual['output'][field][0]['id']=remaining[0]
        if attempt:
            actual['format_correction']={'errors':error_map,'repair_ids':remaining,'previous_response':previous,
                                         'instruction':'Return only these failed/missing IDs. Keep valid rows from the previous response out of this correction. Correct the exact fields; all visual rules still apply.'}
        validator=lambda raw:rows(raw,remaining,mode,table,candidates,pairs)
        try:
            result=stage.call(folder,f'{mode}_{attempt:02d}',SYSTEM,actual,images,validator,budget)
            events.append({'attempt':attempt,'valid':len(result[field]),'errors':result['_errors'],
                           'discarded':result['_discarded']})
            for row in result[field]:good[row['id']]={k:v for k,v in row.items() if k!='id'}
            error_map=result['_errors'];remaining=[x for x in remaining if x not in good]
            previous=result.get('_source_rows',result[field])
        except RequestTimeout as exc:
            error_map={x:'timeout: '+str(exc) for x in remaining};events.append({'attempt':attempt,'timeout':True});break
        except (ValueError,RuntimeError) as exc:
            error_map={x:str(exc) for x in remaining};events.append({'attempt':attempt,'error':str(exc)})
            raw_folder=folder/f'{mode}_{attempt:02d}_00'
            saved=raw_folder/'response_0.json'
            if saved.exists():
                response=read(saved);choice=response.get('choices',[{}])[0]
                if choice.get('finish_reason')=='stop':previous=choice.get('message',{}).get('content','')
                else:break  # A truncated stream is never repaired into a committed style.
            elif not (raw_folder/'result.json').exists():
                # Transport/queue failures have no completed answer to repair.
                # Their existing transport owner already exhausted its budget.
                break
            # An interrupted cached call is evidence, not authorization to resubmit.
            if 'Interrupted prior call' in str(exc):break
    # Runtime caches and retries remain in model/crop units. Only the boundary
    # between this task and controller state converts to original source pixels.
    canonical={k:to_source_pixels(v,source_to_crop_scale) for k,v in good.items()} if mode in ('select','refine') else good
    save(folder/'validation.json',{'mode':mode,'model_valid':good,'valid':canonical,'errors':error_map,'events':events,
                                  'length_conversion':{'model_unit':'main_clean_crop_pixels','stored_unit':'original_source_pixels',
                                                       'source_to_crop_scale':source_to_crop_scale} if mode in ('select','refine') else None})
    return {'valid':canonical,'errors':{k:v for k,v in error_map.items() if k not in good},'events':events}
