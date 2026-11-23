"""Shared observation semantics for block shape generation and refinement."""
import copy
from shape_core.fitting.contracts import model_observation_schema

VERSION = 'block_shape_observed_knots.1'
SAMPLING = {
    'rectangle': 'Exactly 4 observed corners clockwise, starting at top-left. Do not use imaginary sharp corners for a rounded shape.',
    'ellipse': 'Exactly 8 observed contour points clockwise: topmost, upper-right arc, rightmost, lower-right arc, bottommost, lower-left arc, leftmost, upper-left arc. Extremes are relative to image x/y even when tilted.',
    'rounded_rectangle': 'Exactly 16 observed contour points clockwise: top straight midpoint, top-right arc start/middle/end, right straight midpoint, bottom-right arc start/middle/end, bottom straight midpoint, bottom-left arc start/middle/end, left straight midpoint, top-left arc start/middle/end. The backend fits one common corner radius.',
    'path': 'A line references its two endpoints. A curve references at least 3 ordered points ON the observed contour, including its start and end. The backend interpolates THROUGH EVERY point: these are NOT Bezier control points or handles. Usually 3-5 points suffice per smooth arc; use more or split arcs if necessary. Keep corners, head tips, notches and changes of width. Adjacent segments share endpoint IDs. A closed path ends at its first ID. The whole closed shape is not limited to 3-5 points.'}
CONTOUR_RULES = [
    'Read visible geometry from CLEAN. Sample stroke centerlines when an outline is stroked; otherwise sample the fill boundary. Preserve observed curvature and changes of thickness; do not impose a constant-width band, guessed radius or template.',
    'The box, description and mask identify the target but can be inaccurate. Reconstruct only the requested block arrow and its visible intrinsic parts, including supported holes/concavities; exclude neighbor boxes, text and annotation ink. Do not infer hidden segments.',
    'Geometry only: color, gradient, fill opacity, stroke width and dash style are deferred. Shading or a pale fill alone does not create a hole or an extra part.',
    'Return local CLEAN pixel centers, x right and y down. Preserve shared point IDs and actual closed boundaries. Return compact JSON; omit reason for an ordinary successful observation.'
]

def observation_schema(size):
    schema=copy.deepcopy(model_observation_schema(size))
    schema['properties']['points']['description']='Observed contour samples in CLEAN pixels, never Bezier handles.'
    for part in schema['properties']['parts']['items']['oneOf']:
        if part['properties']['kind'].get('const')=='path':
            for segment in part['properties']['segments']['items']['oneOf']:
                if segment['properties']['kind'].get('const')=='curve':
                    segment['properties']['point_ids']['description']='Ordered ON-CURVE observed knots, including endpoints. The interpolating path passes through every referenced point. NOT control points.'
    return schema

def make_prompt(size,box,row,feedback=()):
    q=dict(protocol=VERSION,task='Recover only the located block arrow as closed shape geometry from CLEAN.',
        size=list(size),bbox=box,instructions=list(CONTOUR_RULES),sampling=copy.deepcopy(SAMPLING),
        curve_example={'segment':{'kind':'curve','point_ids':['start','middle','end']},
                       'meaning':'All three points lie on the source contour. The fitted arc must pass through middle; middle is not a control handle.'},
        output_schema=observation_schema(size))
    if row.get('identity_locator'):q['target_description']=row['identity_locator']
    if feedback:
        q['revision_feedback']=list(feedback)
        q['revision_policy']='Reobserve the complete target from CLEAN using the SAME contour-point protocol. Apply only image-supported feedback and preserve correct features.'
    return q
