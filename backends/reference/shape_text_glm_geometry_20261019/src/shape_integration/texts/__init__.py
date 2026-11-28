"""Line-level text recognition, deterministic sizing, and native PPT export."""
import math
PROTOCOL = 'shape_integration.texts.pixel_verified_baselines.17'
FONTS = ('Times New Roman', 'Arial', 'Calibri', 'Courier New')


def options(config):
    out = dict(workers=128, render_workers=2, max_tokens=131072,
               protocol_retries=1, local_enabled=True, max_regions=8,
               global_long_edge=2048, crop_max_long_edge=2048, crop_scale=3.,
               crop_padding=.04, crop_min_padding_px=8.,
               default_font=FONTS[0], native_font_metrics=True,
               font_fit_fraction=.98, render_timeout_seconds=180,
               duplicate_iou=.65, color_recovery=True, native_equations=True,formula_layout_enabled=True,
               geometry_enabled=True, geometry_max_targets=6, geometry_reasoning_effort='low',
               content_review_enabled=True, glyph_registration_enabled=False, source_font_selection=True,
               pixel_reflow_enabled=True, inventory_audit_enabled=True,
               inventory_recall_max_regions=4, crop_recovery_enabled=True)
    out.update(config.get('texts', {}))
    for key in ('workers', 'render_workers', 'max_tokens', 'global_long_edge', 'crop_max_long_edge', 'max_regions', 'geometry_max_targets', 'inventory_recall_max_regions'):
        if type(out[key]) is not int or out[key] < 1:
            raise ValueError('Invalid texts.' + key)
    if type(out['protocol_retries']) is not int or out['protocol_retries'] < 0:
        raise ValueError('Invalid texts.protocol_retries')
    for key in ('crop_scale', 'crop_padding', 'crop_min_padding_px', 'font_fit_fraction', 'duplicate_iou', 'render_timeout_seconds'):
        if type(out[key]) not in (int, float) or not math.isfinite(out[key]) or out[key] < 0:
            raise ValueError('Invalid texts.' + key)
    for key in ('local_enabled', 'native_font_metrics', 'color_recovery', 'native_equations', 'formula_layout_enabled', 'geometry_enabled', 'content_review_enabled', 'glyph_registration_enabled', 'source_font_selection', 'pixel_reflow_enabled', 'inventory_audit_enabled', 'crop_recovery_enabled'):
        if type(out[key]) is not bool:
            raise ValueError('Invalid texts.' + key)
    if out['default_font'] not in FONTS or not 0 < out['font_fit_fraction'] <= 1:
        raise ValueError('Invalid text font/fit policy')
    if not 0 < out['duplicate_iou'] <= 1 or out['crop_scale'] <= 0 or out['render_timeout_seconds'] <= 0:
        raise ValueError('Invalid text sizing/dedup policy')
    if out['geometry_reasoning_effort'] not in ('low','medium','high','max'):
        raise ValueError('Invalid geometry reasoning effort')
    if out['inventory_recall_max_regions'] > 8:
        raise ValueError('texts.inventory_recall_max_regions is bounded to 8')
    return out
