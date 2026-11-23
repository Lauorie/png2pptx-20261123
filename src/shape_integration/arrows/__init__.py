"""Thin native-line arrows: discovery, centerlines, shaft style, then heads."""
PROTOCOL='shape_integration.arrows.4'


def options(config):
    out={'workers':128,'render_workers':2,'max_tokens':131072,'protocol_retries':1,
         'geometry_refines':1,'style_refines':1,'context_expansions':1,
         'crop_padding':.12,'crop_min_padding_px':12,'crop_scale':3.,
         'crop_min_long_edge':1080,'crop_max_long_edge':2048,
         'minimum_stroke_width_px':.8,'duplicate_tolerance_px':1.5}
    out.update(config.get('arrows',{}))
    for key in ('workers','render_workers','max_tokens','crop_min_long_edge','crop_max_long_edge'):
        if type(out[key]) is not int or out[key]<1:raise ValueError('Invalid arrows.'+key)
    for key in ('protocol_retries','geometry_refines','style_refines','context_expansions'):
        if type(out[key]) is not int or out[key]<0:raise ValueError('Invalid arrows.'+key)
    if out['crop_min_long_edge']>out['crop_max_long_edge']:raise ValueError('Invalid arrow crop range')
    return out
