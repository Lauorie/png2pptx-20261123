"""Small visual tasks sharing the integration Runtime and its global limiter."""
import copy
from ..common import save, read
from ..transport import RequestTimeout
from .contracts import rows, number, to_source_pixels

SYSTEM='Recover styles of existing editable geometry from the supplied source pixels. Geometry is fixed. Return one JSON object only.'
PROTOCOL='shape_integration.style.10'
RULES=[
    'Read appearance from CLEAN source; artificial target contours, IDs and IMG boxes are identification only. IMG assets and text are not target shape styles.',
    'A closed path can be an unfilled stroke centerline. The contour does not imply solid fill. Do not fill hollow loops or add an outline to a color boundary without evidence.',
    'Use visible source evidence; infer occluded appearance from the same part or clearly matching repeated instances. Separate foreground contamination from the target. Do not alter geometry.',
    'Review the whole compound together. Keep visibly matching repeated parts consistent in fill, stroke, width, dash and gradient; preserve real differences such as distinct face colors. Ignore tiny antialiasing variations. shared_with lists separate components already visually confirmed to share this target style; inspect them all and return one style for the target ID. Expose contradictory source evidence as an issue rather than inventing agreement.',
    'Fill may be null, #RRGGBB, or {type:linear_gradient,start_color:#RRGGBB,end_color:#RRGGBB,angle_deg:number}. Gradient is opaque with exactly two colors; 0 degrees points right, 90 down in the source image. Do not infer a gradient merely because a part depicts 3D.',
]


def prompt(mode, targets, units, *, feedback=None, pairs=None, color_candidates=None):
    p={'protocol':PROTOCOL,'task':mode,'targets':targets,'units':units}
    example_id=targets[0]['id'] if targets else 'P0'
    if mode in ('select','refine'):
        p['task']='Recover each target\'s two primary attributes: FILL (none, solid color, or supported gradient) and STROKE (none, color, width, and line pattern). Match dashed/dotted stroke rhythm as carefully as its color and width.'
        p['rules']=RULES+([
            'color_candidates and gradient_candidates are optional references. Choose any valid #RRGGBB or supported two-color gradient that better matches CLEAN. Candidate absence alone is not an issue; avoid copying background or antialiasing colors merely because they rank highly.' ] if mode=='select' else [
            'previous_style contains the final parameters used for the supplied previous rendering, with lengths in this MAIN CLEAN crop pixels. Review all listed parts together against CLEAN and critic feedback; adjust these parameters only for evidence-supported problems, preserve correct values, and return complete styles for all target IDs.'])
        p['rules'].append('For a non-solid stroke, identify dash, dot or dash_dot and estimate one repeating dash/dot-and-gap cycle in dash_pattern_px: alternating painted and empty lengths, such as [dash,gap] or [dash,gap,dot,gap]. Match the visible rhythm; omit the lengths only when the source does not support estimating them.')
        if mode=='select':p['color_candidates']=list(color_candidates or [])
        p['rules'].append('Omit issue for a normal successful estimate or routine explanation. Use it only for a material problem needing correction or an unresolved representation/evidence limitation.')
        p['rules'].append('Estimate width_px and dash_pattern_px freely, including fractional pixels, in the MAIN CLEAN crop exactly as supplied. Do not convert to the original image. Put numeric lengths only in structured fields; keep issue text qualitative.')
        p['rules'].append('The CLEAN crop may be strongly enlarged: include the continuous lighter antialiased edge transitions as part of the stroke when estimating its full visible width, rather than measuring only the darkest central band.')
        p['output']={'styles':[{'id':example_id,'fill':'#RRGGBB|null|linear_gradient object','stroke':'#RRGGBB|null',
                               'width_px':'positive main-CLEAN-crop pixel width if stroke, otherwise null','dash':'solid|dash|dot|dash_dot',
                               'dash_pattern_px':'for non-solid strokes: one repeating cycle of alternating painted/gap lengths in main CLEAN crop pixels; omit if visually indeterminate',
                               'issue':'optional short material uncertainty or representation limitation'}]}
        if feedback:p['feedback']=feedback
    elif mode=='critic':
        p['task']='Check each target\'s current_style against CLEAN source and ACTUAL PPT rendering. Prioritize FILL presence/type/colors and STROKE presence/color/width; for non-solid strokes check dash/dot lengths, gaps and repeating rhythm. Only existing target geometry is in scope.'
        p['rules']=RULES[:4]+[
            'current_style gives the actual parameters behind the supplied rendering, with width_px and dash_pattern_px in MAIN CLEAN crop pixels. Use it to identify which style fields need correction, and judge their correctness from the source and rendered images. The supplied parameter values are not evidence that the style matches the source.',
            'Accept concise matches. Reject only with visible source evidence, an actual rendered discrepancy, and actionable style advice. Do not invent a defect from object semantics.',
            'Geometry mismatch and missing text/photos are not fixed by recoloring or changing points. When they block style judgement return uncertain and explain the limitation.',
            'Gradient colors vary over the whole fitted face, not the padded crop. Inspect source trend and rendered trend, including visible portions under occlusion.']
        p['rules'].append('Compare thickness, painted lengths, gaps and number of repeats along corresponding visible boundary sections in the same-scale CLEAN and rendered crops; being dashed in both is not sufficient. Give qualitative evidence and advice, without numerical pixel lengths. Check consistency across matching parts and every shared_with component; report contradictory group appearance as uncertain with visible evidence.')
        p['output']={'checks':[{'id':example_id,'verdict':'accept|revise|uncertain',
                              'source_evidence':'required only if non-accept','rendered_difference':'required only if non-accept',
                              'suggestion':'required only if non-accept; concrete source-supported correction or limitation'}]}
    else:
        p['task']='Infer the whole-component front/back order for these partially overlapping component pairs from source boundary/occlusion evidence. Complete containment is already handled outside this task.'
        p['pairs']=pairs;p['rules']=['Do not choose based on list order, size, or semantic label alone. Use null when evidence is insufficient.',
                                    'Infer only the order visible in the source. If fitted contours cannot be reliably matched to the source, report null; do not try to solve or repair their geometry.',
                                    'Each C ID and its same-color outline identify one complete component, including all its internal parts. Do not compare or reorder parts within a component.',
                                    'Return null if the components interleave or no single whole-component order is supported. Do not output coordinates or change geometry.']
        p['output']={'relations':[{'id':'R0','front':'one of this pair component IDs or null','evidence':'short visible source evidence, or why unresolved'}]}
    if mode in ('select','refine','critic') and units.get('minimum_stroke_width_px',0)>0:
        p['rules'].append('The bottom MIN strip is a same-scale reference for units.minimum_stroke_width_px in MAIN CLEAN pixels: every non-null stroke must be at least this wide; it is a lower bound, not the expected width of every target, and does not require adding a stroke to an unoutlined shape.')
        p['rules'].append('Enlarged crops and pale antialiased edges can cause stroke width to be underestimated; include the connected lighter edges and prefer a slightly thicker stroke when uncertain so the shape remains visible in the full figure. This visibility floor may intentionally exceed a very thin source stroke; do not request a width below it, while still correcting clearly excessive thickness.')
    return p


def ask(runtime, folder, mode, p, images, expected, table, options, *, candidates=None, pairs=None,source_to_crop_scale=None):
    """One first call plus one exact-row format correction. Timeouts never retry."""
    if mode in ('select','refine'):
        source_to_crop_scale=number(source_to_crop_scale,'source_to_crop_scale',True)
    stage=runtime.for_stage(shape_schema_retries=0)
    field={'select':'styles','refine':'styles','critic':'checks','relations':'relations'}[mode]
    remaining=list(expected);good={};events=[];error_map={};previous=None
    for attempt in range(options['protocol_retries']+1):
        if not remaining:break
        actual=copy.deepcopy(p)
        if mode=='relations':actual['pairs']=[x for x in p['pairs'] if x['id'] in remaining]
        # Format correction still sees the complete object/group context.
        else:actual['targets']=copy.deepcopy(p['targets'])
        actual['output'][field][0]['id']=remaining[0]
        if attempt:
            actual['format_correction']={'errors':error_map,'repair_ids':remaining,'previous_response':previous,
                                         'instruction':'Return only these failed/missing IDs. Keep valid rows from the previous response out of this correction. Correct the exact fields; all visual rules still apply.'}
        validator=lambda raw:rows(raw,remaining,mode,table,candidates,pairs)
        try:
            result=stage.call(folder,f'{mode}_{attempt:02d}',SYSTEM,actual,images,validator,options['max_tokens'])
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
            # An interrupted cached call is evidence, not authorization to resubmit.
            if 'Interrupted prior call' in str(exc):break
    # Runtime caches and retries remain in model/crop units. Only the boundary
    # between this task and controller state converts to original source pixels.
    canonical={k:to_source_pixels(v,source_to_crop_scale) for k,v in good.items()} if mode in ('select','refine') else good
    save(folder/'validation.json',{'mode':mode,'model_valid':good,'valid':canonical,'errors':error_map,'events':events,
                                  'length_conversion':{'model_unit':'main_clean_crop_pixels','stored_unit':'original_source_pixels',
                                                       'source_to_crop_scale':source_to_crop_scale} if mode in ('select','refine') else None})
    return {'valid':canonical,'errors':{k:v for k,v in error_map.items() if k not in good},'events':events}
