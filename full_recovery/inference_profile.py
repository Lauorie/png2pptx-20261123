"""Explicit, reversible inference profiles for newly prepared runs only."""
import copy


PROFILES = {
    'full': {'geometry': True, 'style': True, 'pattern': True},
    # Source observations, numerical guards, coverage, overlap/ownership,
    # text recall and the actual-PPT style critic remain enabled.
    'lean': {'geometry': False, 'style': True, 'pattern': False},
}


def apply(settings, shape, text, *, model, effort=None, profile='full'):
    if profile not in PROFILES:
        raise ValueError('Unknown critic profile: '+profile)
    if effort not in (None, 'low', 'medium', 'high'):
        raise ValueError('Invalid forced reasoning effort')
    settings = copy.deepcopy(settings)
    shape, text = copy.deepcopy(shape), copy.deepcopy(text)
    settings.update(api_model=model, glm_api_model=model, glm_model='api:'+model,
                    critic_profile=profile, critic_switches=PROFILES[profile].copy())
    shape['model'] = text['model'] = model
    switches = PROFILES[profile]
    shape.setdefault('fitting', {})['check_enabled'] = switches['geometry']
    shape.setdefault('styling', {}).update(critic_enabled=switches['style'],
        conditional_pattern_critic=switches['pattern'])
    if effort:
        settings['api_force_reasoning_effort'] = effort
        for config in (shape, text):
            config['reasoning_effort'] = effort
            config['force_reasoning_effort'] = effort
        shape['fitting']['check_reasoning_effort'] = effort
        text.setdefault('texts', {})['geometry_reasoning_effort'] = effort
    return settings, shape, text
