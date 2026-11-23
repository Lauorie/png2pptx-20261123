"""Bound physical PPT stroke widths without changing the thinner measurements."""
import math

SLIDE_WIDTH_PT = 13.333333 * 72


def limit(style, source_width, options):
    maximum = options.get('maximum_stroke_width_pt')
    if maximum is None or style.get('stroke') is None:
        return style, None
    maximum_px = maximum * source_width / SLIDE_WIDTH_PT
    before = style['width_px']
    if before <= maximum_px:
        return style, None
    return {**style, 'width_px': maximum_px}, {
        'before_width_px': before, 'after_width_px': maximum_px,
        'before_width_pt': before * SLIDE_WIDTH_PT / source_width,
        'maximum_width_pt': maximum, 'policy': 'upper_bound_only.1'}


def validate(options):
    maximum = options.get('maximum_stroke_width_pt')
    if maximum is not None and (type(maximum) not in (int, float)
                                or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('maximum_stroke_width_pt must be positive finite points or null')
