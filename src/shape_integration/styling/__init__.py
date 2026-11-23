"""Fixed-geometry style recovery. Model acceptance is not visual ground truth."""
import math

DEFAULTS = {
    'workers': 16, 'case_workers': 2, 'pixel_workers': 2, 'render_workers': 2,
    'pixel_calibration_enabled': False,
    'color_roles_enabled': False,
    'conditional_pattern_critic': False,
    'critic_enabled': True,
    'sam_visual_pattern': False,
    'minimum_stroke_width_px': 0.,
    'maximum_stroke_width_pt': None,
    'max_tokens': 32768, 'protocol_retries': 1, 'refine_rounds': 1,
    'component_grouping_enabled': True, 'color_candidate_limit': 15,
    'relation_batch_pairs': 12,
    'max_target_parts': 24,
    'crop_padding': .08, 'crop_min_padding_px': 8.,
    'crop_scale': 3., 'crop_min_long_edge': 1080, 'crop_max_long_edge': 2048,
    'geometry_tolerance_px': .2, 'minimum_overlap_area_px': 1.,
    'render_timeout_seconds': 180,
    'live_render_interval_seconds': 5.,
    'gradient_min_samples': 80, 'gradient_min_span': .45,
    'gradient_min_color_difference': 18., 'gradient_max_error': 14.,
    'gradient_error_ratio': .65,
}


def options(config):
    result = {**DEFAULTS, **config.get('styling', {})}
    result['image_ownership_stage'] = config.get('image_ownership_stage', 'prefit')
    from .width_limit import validate
    validate(result)
    # Old run configs remain readable, but never re-enable part batching.
    result.pop('selection_batch_parts', None)
    result.pop('review_batch_parts', None)
    for k in ('workers', 'case_workers', 'pixel_workers', 'render_workers', 'max_tokens',
              'relation_batch_pairs','max_target_parts'):
        if type(result[k]) is not int or result[k] < 1:
            raise ValueError('Invalid styling option: '+k)
    if type(result['pixel_calibration_enabled']) is not bool:
        raise ValueError('Invalid styling option: pixel_calibration_enabled must be a boolean')
    interval=result['live_render_interval_seconds']
    if type(interval) not in (int,float) or not math.isfinite(interval) or interval<0:
        raise ValueError('live_render_interval_seconds must be finite and nonnegative')
    minimum=result['minimum_stroke_width_px']
    if type(minimum) not in (int,float) or not math.isfinite(minimum) or minimum<0:
        raise ValueError('minimum_stroke_width_px must be a finite nonnegative original-pixel length')
    if type(result['component_grouping_enabled']) is not bool:
        raise ValueError('Invalid styling option: component_grouping_enabled must be a boolean')
    if type(result['conditional_pattern_critic']) is not bool:
        raise ValueError('Invalid styling option: conditional_pattern_critic must be a boolean')
    if type(result['critic_enabled']) is not bool:
        raise ValueError('Invalid styling option: critic_enabled must be a boolean')
    if type(result['sam_visual_pattern']) is not bool:
        raise ValueError('Invalid styling option: sam_visual_pattern must be a boolean')
    if type(result['color_candidate_limit']) is not int or not 1 <= result['color_candidate_limit'] <= 15:
        raise ValueError('color_candidate_limit must be an integer from 1 to 15')
    if result['refine_rounds'] not in (0, 1) or result['protocol_retries'] not in (0, 1):
        raise ValueError('Style recovery supports at most one correction and one format retry')
    return result
