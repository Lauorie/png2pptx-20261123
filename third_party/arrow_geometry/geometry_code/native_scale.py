"""Uniform slide scaling to avoid native marker minimum-size distortion."""
import math

EMU_PER_POINT = 12700
POINTS_PER_INCH = 72
MAX_SLIDE_INCHES = 56
MIN_SLIDE_INCHES = 1
# LibreOffice oox/source/drawingml/lineproperties.cxx uses max(nLineWidth,70)
# in 1/100 mm before applying marker size factors. Confirmed by the render probe.
LO_BASELINE_FLOOR_POINTS = 70 * 72 / 2540
SOURCE = 'https://github.com/LibreOffice/core/blob/master/oox/source/drawingml/lineproperties.cxx'


def plan(size, rows, styles, minimum_head_points=4.0, safety=1.01, head_floor_tolerance=0.0):
    if len(rows) != len(styles) or len(size) != 2 or any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Valid canvas and one style per path required')
    width = min(13.333333, 55 * size[0] / size[1])
    height = width * size[1] / size[0]
    initial_pt_per_px = width * 72 / size[0]
    # A narrow crop can fit the physical slide limits after uniform enlargement.
    # Apply this lower bound together with the head floor, before judging safety.
    # Pixel geometry and aspect ratio stay unchanged, including the 1080x81
    # boundary where the nominal 13.333333-inch width rounded height below 1.
    canvas_factor = max(1.0, MIN_SLIDE_INCHES / min(width, height))
    factor = canvas_factor
    evidence = []
    if minimum_head_points is not None:
        if not math.isfinite(minimum_head_points) or minimum_head_points <= 0 or safety < 1:
            raise ValueError('Positive head floor and safety >= 1 required')
        for row, style in zip(rows, styles):
            if row.get('kind') == 'block_arrow':
                continue
            stroke = style.get('width_px') or row.get('width_estimate') or 2.0
            if not math.isfinite(stroke) or stroke <= 0:
                raise ValueError('Finite positive stroke width required')
            for end in ('start', 'end'):
                head = style.get(end)
                if not head or head.get('type') in ('none', 'unknown'):
                    continue
                factors = {'sm': 2.5, 'med': 3.5, 'lg': 5.5} if head['type'] == 'arrow' else {'sm': 2., 'med': 3., 'lg': 5.}
                length = factors[head.get('length', 'med')]
                breadth = factors[head.get('width', 'med')]
                needed_pt = max(minimum_head_points / min(length, breadth), LO_BASELINE_FLOOR_POINTS) * safety
                required = needed_pt / (stroke * initial_pt_per_px)
                factor = max(factor, required)
                evidence.append(dict(id=row['id'], end=end, type=head['type'], stroke_px=stroke,
                    baseline_stroke_pt=stroke * initial_pt_per_px, required_stroke_pt=needed_pt,
                    width_factor=breadth, length_factor=length))
    required_inches = [width * factor, height * factor]
    allowed_factor = min(MAX_SLIDE_INCHES / width, MAX_SLIDE_INCHES / height)
    safe = factor <= allowed_factor
    requested_inches=list(required_inches)
    floor_inflation=0.0
    if not safe and evidence and head_floor_tolerance>0 and canvas_factor<=allowed_factor:
        # At 2048px, a true 1px stroke just exceeds LO's floor at 56 inches.
        # Keep source stroke width and native med/med unchanged; optionally allow
        # only a bounded renderer discrepancy instead of failing every preview.
        floor_inflation=max(max(1.,LO_BASELINE_FLOOR_POINTS/(e['stroke_px']*initial_pt_per_px*allowed_factor))-1. for e in evidence)
        minimum_ok=all(e['stroke_px']*initial_pt_per_px*allowed_factor*min(e['width_factor'],e['length_factor'])>=minimum_head_points for e in evidence)
        if floor_inflation<=head_floor_tolerance and minimum_ok:
            factor=allowed_factor;required_inches=[width*factor,height*factor];safe=True
    return dict(base_inches=[width, height], factor=factor, required_inches=required_inches,
        slide_emu=[math.ceil(v * 914400) for v in required_inches],
        minimum_head_points=minimum_head_points, renderer_baseline_floor_points=LO_BASELINE_FLOOR_POINTS,
        safety=safety, maximum_slide_inches=MAX_SLIDE_INCHES, allowed_factor=allowed_factor,
        safe=safe, headed_endpoints=evidence, uniform_geometry_and_stroke_scaling=True,
        minimum_slide_inches=MIN_SLIDE_INCHES, canvas_minimum_scale_factor=canvas_factor,
        output_pixel_size=list(size), renderer_source=SOURCE,
        requested_inches_before_limit=requested_inches,head_floor_tolerance=head_floor_tolerance,
        permitted_renderer_head_inflation=floor_inflation if safe and requested_inches!=required_inches else 0.)
