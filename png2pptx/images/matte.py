"""Local chroma-to-alpha, including enclosed holes; never infer a foreground class."""
import numpy as np
from PIL import Image
from scipy import ndimage


def choose_key(image):
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32).reshape(-1, 3)
    # All source pixels, including tiny attachments, influence the choice.
    candidates = [(255, 0, 255), (0, 255, 0), (0, 255, 255)]
    scores = [float(np.percentile(np.linalg.norm(pixels - c, axis=1), 0.1)) for c in candidates]
    key = candidates[int(np.argmax(scores))]
    return list(key), {'minimum_color_separation_p001': max(scores),
                       'source_near_key_fraction': float((np.linalg.norm(pixels-key, axis=1) < 80).mean())}


def chroma_alpha(image, key_rgb, source_image=None):
    """Keep white foreground; key globally, then unmix only a narrow boundary band.

    This is color matting of a generated background, not an object segmentation model.
    A model can still remove details or fail to paint a hole; quality remains unverified.
    """
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    native_alpha = np.asarray(image.convert('RGBA').getchannel('A'), dtype=np.float32) / 255
    key = np.array(key_rgb, dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    # Diffusion output is seldom the exact requested RGB. Accept only a saturated
    # border cluster of the requested hue, never an arbitrary white/gray fallback.
    chroma = border - border.min(axis=1, keepdims=True)
    chroma /= np.maximum(chroma.max(axis=1, keepdims=True), 1)
    hue_distance = np.linalg.norm(chroma-key/255, axis=1)
    near_border = border[(hue_distance < .7) & (np.ptp(border, axis=1) > 100) &
                         (np.linalg.norm(border-key, axis=1) < 200)]
    if len(near_border) > len(border) * .1:
        # The very last pixel row can be a different tint; estimate the dominant
        # keyed field once its presence at the boundary has been established.
        pixels = rgb.reshape(-1, 3)
        field_chroma = pixels-pixels.min(axis=1, keepdims=True)
        field_chroma /= np.maximum(field_chroma.max(axis=1, keepdims=True), 1)
        field = pixels[(np.linalg.norm(field_chroma-key/255, axis=1) < .7) &
                       (np.ptp(pixels, axis=1) > 100) & (np.linalg.norm(pixels-key, axis=1) < 200)]
        key = np.median(field, axis=0)
    collision = 0.
    if source_image is not None:
        source_pixels = np.asarray(source_image.convert('RGB'), dtype=np.float32)
        collision = float((np.linalg.norm(source_pixels-key, axis=2) < 60).mean())
        if collision > .001:
            return image.convert('RGBA'), {'requested_key': key_rgb, 'effective_key': key.tolist(),
                'transparent_fraction': float((native_alpha==0).mean()), 'source_near_effective_key_fraction': collision,
                'quality_flags': ['source_color_close_to_actual_background'], 'method': 'chroma_key_refused_color_collision'}
    distance = np.linalg.norm(rgb-key, axis=2)
    clear = distance <= 35
    alpha = np.ones(distance.shape, dtype=np.float32)
    alpha[clear] = 0
    band = ndimage.binary_dilation(clear, iterations=2) & ~clear & (distance < 220)
    # Nearest non-edge foreground provides a local estimate for antialiased mixtures.
    sure = (distance > 100) & ~ndimage.binary_dilation(clear, iterations=2)
    if sure.any() and band.any():
        indices = ndimage.distance_transform_edt(~sure, return_distances=False, return_indices=True)
        foreground = rgb[indices[0], indices[1]]
        delta = foreground-key
        estimate = np.sum((rgb-key)*delta, axis=2) / np.maximum(np.sum(delta*delta, axis=2), 1)
        alpha[band] = np.clip(estimate[band], 0, 1)
    soft = (alpha > 0) & (alpha < 1)
    rgb[soft] = np.clip((rgb[soft] - (1-alpha[soft, None])*key) / alpha[soft, None], 0, 255)
    alpha *= native_alpha
    rgb[alpha == 0] = 0
    result = Image.fromarray(np.dstack([rgb, alpha*255]).round().astype('uint8'), 'RGBA')
    fraction = float((alpha == 0).mean())
    flags = []
    if fraction < .03: flags.append('background_key_not_found')
    if fraction > .995: flags.append('nearly_empty_asset')
    return result, {'requested_key': key_rgb, 'effective_key': key.tolist(),
                    'transparent_fraction': fraction, 'quality_flags': flags, 'source_near_effective_key_fraction': collision,
                    'method': 'global_chroma_key_with_local_edge_unmixing'}


def checkerboard(image):
    rgba = image.convert('RGBA')
    yy, xx = np.indices((image.height, image.width))
    gray = np.where((xx//16 + yy//16) % 2, 216, 245).astype('uint8')
    bg = Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).convert('RGBA')
    return Image.alpha_composite(bg, rgba).convert('RGB')
