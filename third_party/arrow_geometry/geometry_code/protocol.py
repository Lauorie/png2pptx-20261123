"""One semantic decision over measured geometry; free coordinates only on explicit repair."""
import copy
import contracts as ct
HEADS={'none':'no arrowhead at this endpoint; a shaft end or junction is not a head',
       'triangle':'filled triangle with a straight rear edge; no inward notch or projecting rear wings',
       'stealth':'filled swallowtail with an inward V-shaped rear notch and two pointed rear wings',
       'arrow':'open V formed by two constant-width strokes; the interior is unfilled, even when the strokes are thick',
       'diamond':'diamond-shaped head','oval':'round or elliptical head','unknown':'endpoint is unclear; do not default to triangle'}
SYSTEM='Use the original image as evidence. Mask and geometry overlays are fallible measurements. Return only the requested JSON.'
SCOPE=('Reconstruct visible diagram linework and arrows, headed or headless: connectors, internal wiring, dense networks, graph edges, '
       'axes, baselines, separators, brackets, dimensions, callout leader lines and legend lines. '
       'Exclude the outlines of closed NON-ARROW shapes: rectangles, rounded rectangles, circles, ellipses, polygons, and closed frames or containers, whether solid or dashed. '
       'These belong to shape reconstruction. Do not reconstruct their individual sides or nominate their borders as missing lines, even when a crop or mask contains only part of the outline. '
       'A grid made of adjacent closed cells is ONE composite shape: exclude both its outer border and internal shared cell edges; do not split it into independent lines or nominate those edges as missing. '
       'Use whole-figure context to distinguish shape components from independent lines. Connectors between distinct objects remain eligible, including dense networks; a connector forming a loop is not automatically a shape outline. '
       'Being inside a module, being dense, or lacking an arrowhead is NOT a reason to exclude a line. Broad block arrows remain eligible as closed arrow shapes with fill and stroke. '
       'Exclude text/glyph strokes and photographic texture. Do not duplicate an object explicitly supplied as already reconstructed; a box or a SAM mask alone does not prove coverage. '
       'Follow the visible strokes and attachments; never assume all-to-all connections or extend wires through opaque objects. Crossings alone are not junctions. '
       'SAM candidates are fallible evidence, not proof of eligibility.')
OCCLUSION=('Infer an occluded continuation only when visible stroke portions and their attachments support the same connector. '
           'Two lines ending on one object may be separate; matching color, alignment or a familiar architecture is insufficient. '
           'Preserve a blank-background gap unless visible dash rhythm or other direct evidence supports continuation. '
           'Crossings are not junctions. If connection identity remains ambiguous, leave it unresolved instead of inventing a hidden trunk.')

LINE_STYLES={'solid':'continuous stroke','dashed':'separate elongated marks','dotted':'separate dots','unknown':'not enough visible evidence'}

# Shared by first-pass geometry, omission repair and geometry review. A head is
# an endpoint attribute; extra drawable paths represent actual connector shafts.
ARROW_ENDPOINT_BOUNDARIES=(
    'Collinear or touching shafts can be separate arrows. A visible interior arrowhead is a semantic endpoint: '
    'keep successive arrows as separate targets/paths, each with its own tail and tip; do not run one path through that head. '
    'A genuine double-headed connector with heads only at its two ends may remain one path.')
_CENTERLINE_CORE=(
    'For ordinary line arrows, recover the shaft centerline and its physical endpoints. '
    'The actual tail and visible arrowhead tip have priority. When one visible head establishes direction, '
    'order the complete shaft from tail to tip and end at the tip, not at the head base or a wing. '
    'Do not trace arrowhead edges, V-shaped wings, triangle outlines or head fills as paths or segments. '
    'Arrowheads are endpoint attributes, not additional connector strokes. ')
CENTERLINE_SEMANTICS=(
    _CENTERLINE_CORE+
    'This rule concerns ordinary shaft-plus-head arrows; a broad block_arrow retains its full closed contour. '
    + ARROW_ENDPOINT_BOUNDARIES)
LINEWORK_CENTERLINE_SEMANTICS=(
    'This request uses the linework representation selected upstream: shaft centerlines with endpoint head attributes. '
    'Preserve that representation while fitting or reviewing the geometry; do not reclassify the target as a filled shape '
    'or replace a shaft with its perimeter. Apparent thickness, fill and magnification do not change this contract. '
    + _CENTERLINE_CORE + ARROW_ENDPOINT_BOUNDARIES +
    ' A visibly looping connector remains valid linework; closure alone is not evidence of a shape perimeter.')
MULTIPATH_SEMANTICS=(
    'Multiple paths represent real connector shafts: shared trunks and branches, dense or neural-network wiring, '
    'distinct lines, or genuinely disconnected visible portions. They do not represent the body and wings of one arrowhead. '
    'An unbranched ordinary arrow uses one shaft path unless source evidence requires a real disconnection. '
    'Keep a shared trunk once and use shared junction points where supported; do not invent connections across gaps.')

CURVE_SAMPLING=(
    'For an ordinary smooth curve, prefer 3 to 5 ordered points ON its centerline, INCLUDING both endpoints; '
    'these are not Bezier control points. Locate the actual start and tip first, then choose the few intermediate '
    'points that describe the visible bow and bend direction. Do not densely sample the stroke. '
    'Use the same sparse representation during refinement: move, replace or remove points, and check the entire '
    'visible curve as well as its endpoints. A genuinely more complex curve may use additional meaningful points '
    'or curve segments when needed; retain sharp corners as joined line segments and branches as separate paths.')


def line_styles(raw,labels):
    values=raw.get('line_styles') if isinstance(raw,dict) else None
    if not isinstance(values,dict) or set(values)!=set(labels) or any(v not in LINE_STYLES for v in values.values()):
        raise ValueError('line_styles must declare each selected line ID exactly once using solid/dashed/dotted/unknown')
    return values


def organize(labels,draft,masks,description='',correspondences=None):
    return dict(task='Confirm identities and endpoint head types for ONLY the named global targets, using the measured local SAM geometry.',
        target_sources=labels,target_description=description,measured_geometry=draft,local_masks=masks,head_types=HEADS,line_style_types=LINE_STYLES,
        measured_correspondences=correspondences or {},
        instructions=[SCOPE,ARROW_ENDPOINT_BOUNDARIES,
            'CONTEXT marks global targets. CLEAN is the original crop. E paths and N endpoints are measured from local SAM; colors are not ground truth. Ordinary targets with usable geometry take those paths directly. Do not re-estimate coordinates.',
            'G and M name ink masks, never the nearby text or boxes. Use measured_correspondences to locate G in local CLEAN coordinates and identify its M/E evidence; do not infer ownership from where a label happens to be printed. Overlap scores are fallible retrieval hints, not mandatory merges. All visualization colors are arbitrary.',
            'G sources are coarse discovery hints, not confirmed physical arrows. Base target grouping on visible E routes and CLEAN. Multiple overlapping G masks may all support the same target; a fragmented mask may support several targets. Do not try to explain every difference between SAM masks or the architecture of the whole figure.',
            'Group E paths into independent arrows or a truly shared network. Merge duplicate/fractured source evidence; a mixed G source may support several targets. Only list N in junctions for a true common connection, not an ordinary crossing. A shared path may be referenced by multiple targets but will be drawn once.',
            'Give the head type at every endpoint N used by selected E paths, including none at internal junctions. For a large filled block arrow use filled_masks instead of centerlines; the complete mask outline will be used.',
            'For every selected E path give line_styles[E]: solid, dashed, dotted or unknown from CLEAN. Exclude the head from this judgment; occlusion alone does not make a line dashed. Do not measure widths or dash lengths. Shared networks may contain different styles on different E paths.',
            'Neighbors introduced by the crop are context, not new targets. Each G is assigned to a target, excluded, or uncertain. If local geometry is missing, wrongly connected, or incomplete, keep its identity with repair set to a short explanation; otherwise repair=null. Additional masks can be inspected in that repair step.'],
        output_schema={'targets':'list of {sources:[G IDs], paths:[E IDs], filled_masks:[M IDs], repair:null or short reason}',
            'heads':'object mapping used N IDs to head types','line_styles':'object mapping selected E IDs to solid/dashed/dotted/unknown',
            'junctions':'list of true shared N IDs','excluded':'list of G IDs','uncertain':'list of G IDs'})


def decision(raw,labels,draft,mids):
    if not isinstance(raw,dict) or not isinstance(raw.get('targets'),list):raise ValueError('targets list required')
    a=copy.deepcopy(raw);used=set();paths={r['id']:r for r in draft['paths']};nodes=set(draft['nodes']);required=set()
    for t in a['targets']:
        t['sources']=ct.ids(t.get('sources'));t['paths']=ct.ids(t.get('paths',[]));t['filled_masks']=ct.ids(t.get('filled_masks',[]))
        if not t['sources'] or not set(t['sources'])<=set(labels):raise ValueError('Target must name its known source identities')
        if not set(t['paths'])<=set(paths) or not set(t['filled_masks'])<=set(mids):raise ValueError('Unknown measured path or mask')
        if t.get('repair') is not None and not isinstance(t['repair'],str):raise ValueError('repair is null or a brief reason')
        if not (t['paths'] or t['filled_masks'] or t.get('repair')):raise ValueError('Target needs mask geometry or explicit repair')
        if not t.get('repair'):required.update(n for e in t['paths'] for n in paths[e]['nodes'])
        used.update(t['sources'])
    heads=a.get('heads');a['junctions']=ct.ids(a.get('junctions',[]))
    if not isinstance(heads,dict) or not set(heads)<=nodes or any(h not in HEADS for h in heads.values()):raise ValueError('Known N IDs and head types required')
    if not required<=set(heads):raise ValueError('Declare heads for used N endpoints: '+','.join(sorted(required-set(heads))))
    if not set(a['junctions'])<=nodes:raise ValueError('Unknown shared node')
    line_styles(a,{p for t in a['targets'] if not t.get('repair') for p in t['paths']})
    a['excluded']=ct.ids(a.get('excluded'));a['uncertain']=ct.ids(a.get('uncertain'))
    ex,un=set(a['excluded']),set(a['uncertain'])
    if used&ex or used&un or ex&un or used|ex|un!=set(labels):raise ValueError('Account for each G in targets, excluded or uncertain')
    return a


def repair(target,draft,size,labels):
    return dict(task='Repair ONLY this confirmed target. Prefer the expanded local mask geometry when usable; estimate coordinates only if the masks cannot represent its route.',
        target=target,available_sources=labels,size=size,measured_geometry=draft,head_types=HEADS,line_style_types=LINE_STYLES,
        instructions=[SCOPE,CENTERLINE_SEMANTICS,MULTIPATH_SEMANTICS,
            'Do not recover unrelated neighboring arrows. This is the same target; the crop did not create new identities.',
            'mode=mask selects E paths or filled M masks using the decision schema. For a shared network explicitly keep common junction nodes. No coordinates are needed in mask mode.',
            'Also classify the visible stroke in CLEAN: mask mode supplies line_styles for each selected E; points mode supplies line_style on each non-block route. Use solid/dashed/dotted/unknown. Head ink and occlusion gaps do not define a dash pattern. Do not estimate style dimensions.',
            'Only when mask geometry is unusable, mode=points supplies nodes [[id,x,y]] and routes [{id,kind,nodes:[ordered IDs],heads:[first,last],sources:[G IDs]}]. Coordinates are local CLEAN pixels, one decimal. Real junctions share IDs; crossings do not. A block_arrow uses an ordered closed contour of at least 3 vertices and heads none/none. Curves use on-curve points, not directional descriptions.'],
        output_schema={'mode':'mask or points','targets':'mask mode: list of {sources:[G IDs],paths:[E IDs],filled_masks:[M IDs],repair:null}',
            'heads':'mask mode: N to type','line_styles':'mask mode: selected E to solid/dashed/dotted/unknown',
            'junctions':'mask mode: shared N IDs','excluded':'G IDs','uncertain':'G IDs',
            'nodes':'points mode: list of [id,x,y]','routes':'points mode: route records including line_style on each non-block route'})


def coverage(size,recovered=None,*,input_mode='two_images'):
    if input_mode=='two_images':
        # The tested global task only nominates regions. Numeric ownership and
        # mask evidence stay with the subsequent local reconstruction request.
        return dict(task='Propose candidate regions for LOCAL inspection of possibly missing lines. This is region discovery, not geometry reconstruction.',
            size=list(size),instructions=[
                SCOPE+' A broad filled arrow is a filled object, not a thick shaft with a marker.',
                'ORIGINAL is the intact source. RECOVERED shows actual recovered paths in blue over a faded gray source, including known heads, dashes and filled contours. Gray source ink is not reconstruction. Thin blue shafts are thickened only for visibility. Use this visible coverage directly; do not infer route ownership or reconstruct each numbered path before proposing a region. Ignore minor geometric/style differences.',
                'Look for visible strokes or portions not represented by known paths. Return a roughly enclosing bbox and a short visual locator for each suspected omission. Several nearby missing portions can share one inspection region. Local inspection will accept, reject, split or enlarge it; neither precise endpoints, branch counts, connectivity nor an optimal partition is required now.',
                'Do not trace whole networks or reason about the diagram function. Ignore minor geometric/style differences. It is acceptable to propose a plausible incomplete area without proving its complete route. Return regions=[] when no candidate is visible. Return JSON only.'],
            output_schema=dict(regions='[{bbox:[left,top,right,bottom],description:short visual locator}]'))
    if input_mode!='boxes_and_inventory':
        raise ValueError('Unknown coverage_input_mode: '+str(input_mode))
    return dict(task='Locate missing arrows/lines or missing portions of connections. Return repair regions, not geometry.',size=size,
        already_reconstructed=recovered,
        instructions=[SCOPE,OCCLUSION,
            'ORIGINAL is intact. BOXES only locate known targets; a large box does not prove all its branches are recovered. ACTUAL GEOMETRY shows recovered paths in deep blue over a faded gray source, including known heads, dashes and filled contours. Thin shafts are thickened only for visibility; gray source ink is not reconstruction. Blue results are not ground truth.',
            'Compare source lines with actual paths, including short returns, stubs and branches inside existing boxes. Do not assume another group covers them. Ignore minor style/position errors.',
            'Return a tight bbox around each related missing portion and its attachment points, with a short identifying description. The next request enlarges this region and sees all existing neighboring paths. Do not guess branch counts or coordinates in the description. Return [] only if no missing in-scope connection is visible.'],
        output_schema={'regions':'list of {bbox:[left,top,right,bottom],description:short identifying phrase}'})


def regions(a,size):
    if not isinstance(a,dict) or not isinstance(a.get('regions'),list):raise ValueError('regions list required')
    for r in a['regions']:
        ct.bbox(r.get('bbox'),size)
        if not isinstance(r.get('description'),str):raise ValueError('Omission description required')
    return a
