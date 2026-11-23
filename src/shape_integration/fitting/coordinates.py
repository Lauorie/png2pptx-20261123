"""Model coordinate units; all fitting, rendering and exports use pixel centers."""
import copy

NORM1000_CONVENTION = (
    'Return every xy as [x,y] in normalized 0-1000 coordinates over the entire input image, '
    'with (0,0) at the top-left pixel center and (1000,1000) at the bottom-right pixel center.')
NORM1000_BOX_CONVENTION = (
    'Return every bbox_2d as [x_min,y_min,x_max,y_max] in normalized 0-1000 coordinates over the entire input image, '
    'with (0,0) at the top-left pixel center and (1000,1000) at the bottom-right pixel center.')


def coordinate_mode(options):
    mode = options.get('coordinate_mode', 'pixel')
    if mode not in ('pixel', 'norm1000'):
        raise ValueError('Unknown model coordinate mode: '+str(mode))
    return mode


def model_size(size, mode):
    coordinate_mode({'coordinate_mode': mode})
    return (1001, 1001) if mode == 'norm1000' else tuple(size)


def to_pixels(values, size, mode):
    coordinate_mode({'coordinate_mode': mode})
    return [v * (size[i % 2] - 1) / 1000 for i, v in enumerate(values)] if mode == 'norm1000' else list(values)


def observation_to_pixels(observation, size, mode):
    """Convert after Runtime.call, keeping its cached result in model units."""
    result = copy.deepcopy(observation)
    for point in result['points']:
        point['xy'] = to_pixels(point['xy'], size, mode)
    return result
