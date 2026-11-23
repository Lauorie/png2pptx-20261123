"""Small stage-specific prompts. No global coordinates or candidate style guesses."""
from . import PROTOCOL

HEAD_GUIDE_LABEL='Native head enum shape legend; displayed size and color are illustrative'


def path_context(geom):
    return [{k:p[k] for k in ('id','start_head','end_head')} for p in geom['paths']]


def path_topology(geom):
    """Identity and point order only; no coordinates or fitting-model explanations."""
    return [{'id':p['id'],
             'segments':[{'kind':s['kind'],'points':list(s['points'])} for s in p['segments']],
             'start_head':p['start_head'],'end_head':p['end_head']} for p in geom['paths']]

SYSTEM='Recover line arrows from the supplied scientific figure. Return only the JSON response described in response_format, with its fields at the top level. No prose, Notes, code fences or second JSON. Do not echo the instructions or wrap the response in output. Read CLEAN pixels; colored annotations identify targets and are not source evidence.'
SCOPE='Recover arrows expressible as a stroked centerline plus native endpoint heads, including short arrows, long feedback routes, bends/curves, double heads and branching networks. Include headless connector lines when they visibly connect diagram components; never invent heads. An open V head is a line arrow even when short or thick after crop enlargement. Exclude broad filled block-arrow polygons with a wide body and shoulders; judge structure, not enlarged pixel width. Exclude text, unheaded plot curves, image texture and ordinary shape borders. Follow visible routes, not connections implied only by diagram semantics.'


def discover(size,known=(),*,layout=False,concerns=()):
    p={'protocol':PROTOCOL,'task':'Locate complete line arrows and connected arrow networks; approximate enclosures suffice. Detailed centerlines and styles will be fitted later.' if layout else 'Locate complete line arrows/networks missing from the annotated existing candidates; approximate enclosures suffice.',
       'scope':SCOPE,'coordinates':f'Current CLEAN image pixel centers: x=0..{size[0]-1}, y=0..{size[1]-1}. bbox_2d is [left,top,right,bottom].',
       'rules':['One bbox encloses the ENTIRE shaft and all heads of one arrow or explicitly connected branching network. Preserve small arrows; do not combine unrelated parallel arrows. A headed endpoint is the visible extreme tip, not the head base.',
                'The numbered boxes are visual hints only. If an arrow is already covered, omit it or set same_as to its existing ID. Return new objects only when distinct. Do not turn a clipped fragment into a complete new arrow; describe the missing context in uncertain.',
                'Descriptions and uncertain are passed to other crops: use brief labels, route and head directions, with NO coordinate numbers or pixel measurements. Put coordinates only in bbox_2d. Use uncertain only for unresolved missing/clipped routes, not a list of excluded graphics.'],
       'existing_ids':list(known),'response_format':{'objects':[{'bbox_2d':[0,0,20,20],'description':'brief coordinate-free identity and route','same_as':None}],'uncertain':[]}}
    if layout:
        p['rules'].append('Also divide the figure into a FEW coarse layout regions that together cover the whole canvas. Keep a simple figure in one region. Do not make one region per arrow; overlaps and cross-region arrows are allowed.')
        p['response_format']['regions']=[{'bbox_2d':[0,0,size[0]-1,size[1]-1],'description':'coarse section'}]
    if concerns:p['reported_missing_context']=list(concerns)
    return p


def geometry(size,description,previous=None,feedback=None):
    p={'protocol':PROTOCOL,'task':'Recover the complete centerline geometry of this target arrow/network. Style will be recovered later.',
       'target':description,'scope':SCOPE,'coordinates':f'Only current CLEAN crop pixel centers, x=0..{size[0]-1}, y=0..{size[1]-1}.',
       'rules':['The candidate description is a tentative identity hint, not proof of an arrow. Verify the route and visible heads in CLEAN. Trace the entire shaft, every bend and branch. Headed endpoints must be at the visible extreme arrow TIPS, not the base of the head. Headless endpoints end where source ink ends. Do not snap to nearby shape centers or erase gaps to shapes.',
                'Use enough points/segments to preserve the visible route. line joins points with straight segments; curve points lie ON the source centerline and will be smoothly interpolated. Use separate line/curve segments at sharp corners. Do not replace a long multi-bend arrow by an endpoint chord. Dashed/dotted arrows use a CONTINUOUS underlying centerline; painted/gap lengths belong to style, not separate disconnected segments.',
                'Use one shared point ID at each real junction. A crossing without a connection is not a junction. Represent shared trunks once, with separate branches. Double-headed paths set both head flags true; headless connectors set both false.',
                'Adjacent segments of one path share their endpoint ID. Describe plausible continuity behind short occlusions when the source route supports it, retaining material uncertainty. If the target extends beyond the crop, return needs_context; use not_arrow only for clear out-of-scope targets.',
                'For refinement, previous_geometry and its point/line overlay describe the same prior revision in this crop. Correct evidence-supported problems and return the COMPLETE network, preserving correct branches.'],
       'response_format':{'status':'ready|needs_context|not_arrow','points':{'v0':[0,0],'v1':[10,10]},
                 'paths':[{'id':'p0','segments':[{'kind':'line|curve','points':['v0','v1']}],'start_head':False,'end_head':True}],
                 'uncertain':[],'reason':'required only for needs_context/not_arrow'}}
    if previous is not None:p['previous_geometry']=previous
    if feedback:p['feedback']=feedback
    return p


def critic(geom,style=None,size=None):
    if style is None:
        return {'protocol':PROTOCOL,
            'task':'Image 1 is CLEAN source. Image 2 overlays fitted points and lines on the same source. '
                   'Check whether the sampled point centers and each stretch between them lie on the target source stroke. '
                   'List visible problems and reasons first; give the verdict last.',
            'paths':path_topology(geom),
            'rules':[
                'Point IDs in paths.segments.points label sample centers. Path IDs such as p0 label entire paths; they are not sample points or control points. Judge the circle centers, not the offset text badges.',
                'Report a point or intervening stretch when it is visibly displaced sideways from the source stroke, including a background gap between fit and source. Identify the point ID, what is visible, and the direction or correction. Source strokes and backgrounds may have any color. Symmetric exposed edges of a thicker source stroke are not displacement.',
                'Check middle points and intervening stretches individually. Correct endpoints do not excuse a curve cutting inside or outside the source arc. For dashed strokes or brief occlusions, use the visible route on both sides; a dash gap alone is not a geometry error.',
                'Headed endpoints belong at the extreme visible tip; headless endpoints end at the source ink or actual junction. Report uncovered target sections/branches as missing. Report fitted branches with no source counterpart for removal; do not move them onto text or shape borders. Check every branch and junction.',
                'Ignore width, color and head style. Do not claim subpixel deviations you cannot see. Use only visible source evidence. Accept only when all three issue lists are empty and the point/line, endpoint and completeness checks have visible support. Use uncertain when source visibility prevents a decision and explain what cannot be seen.'
            ],
            'response_format':{
                'off_points':[{'id':'existing sample point ID','visible':'source versus fit evidence; name the adjacent point too if a stretch is wrong','move':'direction toward source, reshape the stretch, or remove an unsupported branch'}],
                'endpoint_issues':[{'id':'existing endpoint point ID','visible':'source versus fitted endpoint evidence','move':'correction to the source tip, ink end or junction'}],
                'missing_sections':[{'visible':'visible target section or branch not covered by the fit','fix':'where to add or extend the route; no existing ID required'}],
                'reason':'Brief image evidence for the decision; for uncertain say which source section is unreadable. Empty issue lists mean no issue, not skipped checks.',
                'verdict':'accept|revise|uncertain; accept requires all issue lists empty, revise requires at least one issue'
            }}
    p={'protocol':PROTOCOL,'task':'Compare CLEAN with the ACTUAL native PPT render for the target arrow/network. Check shaft color/width/dash rhythm first, then head type, width and length. Missing unrelated shapes/text/photos are outside scope.',
       'rules':['Give one concise visual decision. Accept a visually close reconstruction; reject substantial visible differences with source evidence and an actionable supported parameter change. Current style describes the render, not proof that it matches CLEAN.',
                'Geometry is fixed. Judge shaft color/width/dash and head shape/size without re-deriving endpoints. Read the head interior and arms from CLEAN: two stroked arms with an open notch are arrow, filled triangular interior is triangle, filled head with a small rear notch is stealth. Use the native enum shape legend when supplied; its displayed size/color are not target estimates.',
                'Numeric lengths are pixels of these target crops. Head types are triangle/stealth/arrow/diamond/oval; width and length only sm/med/lg. Native head sizes are not universal multiples of shaft width: judge the ACTUAL render, not an assumed ratio. Do not thicken a correct shaft to compensate for a head-size limit. Keep evidence and suggestions brief.'],
       'response_format':{'verdict':'accept|revise|uncertain','evidence':'visible source versus fitted/rendered difference, required for non-accept','suggestion':'specific correction, required for non-accept'}}
    p.update(paths=path_context(geom),current_style=style,image_size=size)
    return p


def style(mode,geom,current=None,feedback=None,minimum=0,*,size=None):
    p={'protocol':PROTOCOL,'task':('Recover shaft styles for ALL paths together. Heads are estimated next.' if mode=='shaft' else
                                 'Choose native PPT head type and width/length enums at each observed headed endpoint, using the fixed shaft width below.'),
       'paths':path_context(geom),'image_size':size,'units':'All numeric lengths are pixels of the supplied main CLEAN crop; do not convert to global pixels.',
       'rules':['Read colors and widths from CLEAN. Target IDs/contours are identification only. Visibly matching branches should share styles; preserve genuine differences.',
                'Crop enlargement can soften antialiased edges: include the connected lighter stroke edges in its visible thickness, not only the darkest core. Prefer a slightly thicker estimate if uncertain.',
                'When refining, previous_style is the exact prior rendered style in this crop. Preserve correct values and change only source-supported defects. Return complete results for every path.'],
       'response_format':{'styles':[{'id':geom['paths'][0]['id'],'stroke':'#RRGGBB','width_px':2,'dash':'solid|dash|dot|dash_dot','dash_pattern_px':[12,8]}]} if mode=='shaft' else
                {'heads':[{'id':geom['paths'][0]['id'],'start':None,'end':{'type':'triangle|stealth|arrow|diamond|oval','width':'sm|med|lg','length':'sm|med|lg'}}]}}
    if mode=='shaft':
        p['minimum_width_px']=minimum
        p['rules'].append('Estimate width and dash painted/gap lengths from CLEAN; example numbers in response_format are placeholders. dash_pattern_px is an optional JSON ARRAY of 2, 4 or 6 positive numbers alternating painted/gap lengths, never a string; omit it for solid. The minimum is only a visibility floor, not a suggested target width.')
    else:
        p['rules']=['Choose visually close enums; the actual-render critic will adjust substantial size errors. Do not fit head corner coordinates or derive renderer formulas. Read head structure in CLEAN: an open V of two stroked arms is arrow; a filled triangle is triangle; a filled head with a small rear notch is stealth. Compare with the native enum shape legend.',
                    'The legend identifies enum shapes only; its displayed pixel size and color are not target estimates. Choose width and length independently from sm/med/lg; native sizes are not universal multiples of shaft width. During refinement, heads visible in the ACTUAL render have exact enums in shaft_style: use their sizes as reference, while CLEAN decides type. Do not recompute endpoints or change shaft styles.',
                    'The target annotation labels headed terminals as path_id.start / path_id.end. False head flags require null. Return every path. Feedback and previous head choices are provisional; use CLEAN to decide and preserve genuine differences between heads.']
    if current is not None:p['previous_style' if mode=='shaft' and feedback else 'shaft_style']=current
    if feedback:p['feedback']=feedback
    return p
