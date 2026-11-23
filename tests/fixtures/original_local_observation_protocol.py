"""Model sees only local observations, identity descriptions and two aligned images."""
import json
import re
import numpy as np
from jsonschema import Draft202012Validator

SYSTEM = 'Observe the requested figure object carefully. Return only JSON matching the supplied schema. Report observed contour points in the supplied crop pixel coordinates. Treat image annotations and target descriptions as fallible identity aids.'

def obj(properties, required=None):
    return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}

def schema(w,h):
    ident={'type':'string','minLength':1}
    def ids(n=None, minimum=2):
        d={'type':'array','items':ident,'minItems':n or minimum}
        if n:d['maxItems']=n
        return d
    part=[]
    for kind,n in [('ellipse',8),('rounded_rectangle',16),('rectangle',4)]:
        part.append(obj({'id':ident,'kind':{'const':kind},'point_ids':ids(n)}))
    segment={'oneOf':[obj({'kind':{'const':'line'},'point_ids':ids(2)}),obj({'kind':{'const':'curve'},'point_ids':ids(minimum=3)})]}
    part.append(obj({'id':ident,'kind':{'const':'path'},'closed':{'type':'boolean'},'segments':{'type':'array','items':segment,'minItems':1}}))
    return obj({'object_id':ident,'status':{'enum':['ok','unsupported']},
        'points':{'type':'array','items':obj({'id':ident,'xy':{'type':'array','prefixItems':[{'type':'number','minimum':0,'maximum':w-1},{'type':'number','minimum':0,'maximum':h-1}],'items':False,'minItems':2,'maxItems':2}})},
        'parts':{'type':'array','items':{'oneOf':part}},'reason':{'type':'string'},'uncertain':{'type':'array','items':{'type':'string'}}})

# Descriptions may hide numeric geometry in prose. Keep semantic fragments only.
NUMBER = r'-?\d+(?:\.\d+)?'
COORDINATE = re.compile(r'(?i)(?:~|≈)?\b(?:global\s+)?[xy](?:[-\s]+(?:range|span|bounds?|coordinate))?\s*(?:(?:from|to|is|at|=|~|≈|:)\s*)?(?:~|≈)?'+NUMBER+r'(?:\s*(?:-|–|—|to)\s*(?:~|≈)?'+NUMBER+r')?')
GEOMETRY_CLAUSE = re.compile(r'(?i)(?:bbox|bounding\s*box|\bcoordinates?\b|\bpixels?\b|\b\d+(?:\.\d+)?\s*(?:px|times|[x×]\b)|\b(?:upscal\w*|rescal\w*|resize\w*|resolution|scal(?:e|ing))\b|\b[xy]\s*[=~≈]|[\[(]\s*-?\d+(?:\.\d+)?\s*[,，]|\b(?:width|height|radius|radii|diameter)\s*(?:of|is|=|:|~)?\s*\d|\b(?:image\s+boundary|boundary\s+at)\s*\d)')

# Free multi-digit numbers in description prose are withheld; alphanumeric labels
# such as CodeLlama 34B, 31k, S0017 and row/column IDs are identity, not coordinates.
EXTENT_NUMBER = re.compile(r'(?<![A-Za-z0-9_.])\d{2,}(?:\.\d+)?(?![A-Za-z0-9_.])|\b\d{2,}\s*[x×]\s*\d+\b|\b\d+\s*[x×]\s*\d{2,}\b')

def sanitize(text):
    pieces=re.split(r';|(?<=[.!?])\s+',str(text))
    kept=[];removed=[]
    for piece in pieces:
        original=piece.strip()
        # First remove coordinate-bearing parentheticals without losing the object identity.
        piece=re.sub(r'\([^()]*\)|\[[^\[\]]*\]',lambda m:'' if COORDINATE.search(m[0]) or GEOMETRY_CLAUSE.search(m[0]) or EXTENT_NUMBER.search(m[0]) else m[0],piece)
        piece=COORDINATE.sub('',piece)
        piece=re.sub(r'\s+',' ',piece).strip(' ,;')
        piece=re.sub(r'\s+([.,])',r'\1',piece)
        # Numeric extent-only clauses convey no identity once their coordinates are gone.
        useless=re.fullmatch(r'(?i)(?:node extents span|their proposed boxes span|extents span|span|at)[\s,.-]*',piece)
        if GEOMETRY_CLAUSE.search(piece) or EXTENT_NUMBER.search(piece) or useless or not piece:
            removed.append(original)
        else:
            if piece!=original:removed.append(original)
            kept.append(piece)
    return '; '.join(kept),removed

def make_prompt(job, descriptions):
    w,h=job['frame']['actual_crop_input_size_px']
    return {'version':'crop_observed_sparse_v1','target':{'id':job['object_id'],'label':sanitize(job['label'])[0],
        'proposed_shape':job['proposed_type'],'identity_descriptions':descriptions},
        'images':['Image 1 is the clean crop: observe geometry here.','Image 2 is the identical crop with a magenta target box and ID, used only to locate the intended object. The annotation is not a contour; do not snap points to it.'],
        'local_coordinates':{'width_px':w,'height_px':h,'origin':'top-left pixel center','x_direction':'right','y_direction':'down','x_range':[0,w-1],'y_range':[0,h-1],'precision':'At most two decimal places; integer pixels are fine.'},
        'task':'Observe only the requested object and describe its geometry using a shared point table and parts. Output crop-local observed points. Descriptions select the object, not its exact geometry.',
        'constraints':[
            'Place points on the centerline of a visible stroke; if there is no stroke, use the fill boundary. Ignore text, shadows, annotation ink and unrelated objects. A dashed outline follows its underlying contour.',
            'Observe points independently on the clean image. Do not compute evenly spaced template points from a guessed center, radius or annotation box. Do not regularize observed points to make a perfect primitive; the input can be distorted.',
            'Circle or ellipse: use kind ellipse and exactly 8 contour points in clockwise order: topmost, one observed upper-right arc point, rightmost, lower-right arc point, bottommost, lower-left arc point, leftmost, upper-left arc point. Extremes are with respect to crop x/y even when tilted.',
            'Rounded rectangle: exactly 16 contour points clockwise: top straight midpoint, top-right arc start, top-right arc middle, top-right arc end, right straight midpoint, bottom-right arc start, bottom-right arc middle, bottom-right arc end, bottom straight midpoint, bottom-left arc start, bottom-left arc middle, bottom-left arc end, left straight midpoint, top-left arc start, top-left arc middle, top-left arc end. Start/end mean where that corner meets the adjacent straight span. For a rotated object follow the same local side order clockwise. The backend fits one common radius for all four corners; report actual observations even if distortion makes them deviate.',
            'Rectangle: 4 observed corners clockwise starting at its top-left. Do not report virtual sharp corners of a rounded rectangle.',
            'Other polygons, lines, curves or compound objects: preserve every structural part needed for this target. Use path parts with line segments (two endpoint IDs) or curve segments (at least three ordered observed knots). Adjacent segments share endpoint IDs. A closed path must end at the starting ID. Use only the points needed to capture structure; do not force other shapes into 8 or 16 points.',
            'Ellipse and rounded-rectangle parts inside compounds must also obey the 8/16-point rules. Reuse IDs at shared junctions; do not duplicate unrelated objects. Proposed shape is a hint, not permission to invent invisible geometry.',
            'Return no centers, radii, axes, bounding boxes, transform formulas or coordinates outside the crop. The backend computes all fitted parameters and coordinate conversions.',
            'If the intended contour cannot be observed well enough, return status unsupported with empty points and parts and explain briefly. Do not invent hidden points. Otherwise use status ok, and list any uncertain observations.'
        ],'output_schema':schema(w,h)}

def validate_raw(raw, job, prompt):
    Draft202012Validator(prompt['output_schema']).validate(raw)
    if raw['object_id']!=job['object_id']:raise ValueError('Wrong object ID')
    points={p['id']:p['xy'] for p in raw['points']}
    if len(points)!=len(raw['points']):raise ValueError('Duplicate point IDs')
    if raw['status']=='unsupported':
        if points or raw['parts']:raise ValueError('Unsupported must be empty')
        return
    if not points or not raw['parts']:raise ValueError('Empty ok response')
    for p in points.values():
        if not np.isfinite(p).all() or np.max(np.abs(np.array(p)*100-np.round(np.array(p)*100)))>1e-6:raise ValueError('Nonfinite or excessive coordinate decimals')
    refs=set();part_ids=set()
    for part in raw['parts']:
        if part['id'] in part_ids:raise ValueError('Duplicate part IDs')
        part_ids.add(part['id'])
        if part['kind']=='path':
            seg=part['segments'];ids=[p for s in seg for p in s['point_ids']]
            if any(a['point_ids'][-1]!=b['point_ids'][0] for a,b in zip(seg,seg[1:])):raise ValueError('Disconnected path')
            if part['closed'] and seg[-1]['point_ids'][-1]!=seg[0]['point_ids'][0]:raise ValueError('Unclosed path')
        else:
            ids=part['point_ids']
            if len(set(ids))!=len(ids):raise ValueError('Repeated primitive point ID')
        if any(p not in points for p in ids):raise ValueError('Missing point reference')
        refs.update(ids)
        xy=np.array([points[p] for p in ids]);unique=np.unique(xy,axis=0)
        if len(unique)<2:raise ValueError('Degenerate part')
        if part['kind']!='path' and (len(unique)!=len(ids) or np.linalg.matrix_rank(xy-xy.mean(0),tol=1e-6)<2):raise ValueError('Degenerate primitive')
    if refs!=set(points):raise ValueError('Unused points')
