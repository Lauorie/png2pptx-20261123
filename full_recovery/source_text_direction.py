"""Resolve a reading-direction reversal from complete native glyph ink.

Registration is read-only: the source box and typography are never fitted to
the probe. Only a strongly supported half-turn can change the text angle.
"""
from copy import deepcopy
from pathlib import Path

PROTOCOL = 'complete_source_ink_reading_direction.1'


def observe(row, source):
    import cv2
    import numpy as np
    from PIL import Image

    metric = row.get('layout', {}).get('measurement', {})
    template_path = metric.get('ink_template')
    if (row.get('kind', 'text') == 'formula' or row.get('layout_mode', 'single_line') != 'single_line'
            or row.get('visual_line_count', 1) != 1 or not template_path
            or not Path(template_path).is_file()):
        return None
    color = row.get('color')
    if not isinstance(color, str) or len(color) != 7:
        return None
    pigment = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
    bounds = np.asarray(row['bbox_2d'], float); span = bounds[2:] - bounds[:2]
    if np.any(span <= 0): return None
    margin = np.maximum(3, np.minimum(12, span * .2))
    lo = np.maximum(0, np.floor(bounds[:2] - margin)).astype(int)
    hi = np.minimum(source.size, np.ceil(bounds[2:] + margin)).astype(int)
    if np.any(hi <= lo): return None
    rgb = np.asarray(source.crop((*lo, *hi)).convert('RGB'), float)
    border = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3),
                             rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    background = np.median(border, axis=0); delta = background - pigment
    if float(delta @ delta) < 1: return None
    ink = np.clip(((background-rgb)*delta).sum(2)/float(delta@delta), 0, 1).astype('float32')
    with Image.open(template_path) as opened:
        alpha = opened.convert('RGBA').getchannel('A')
        region = alpha.getbbox()
        if region is None: return None
        alpha = alpha.crop(region)
    angle = float(row.get('rotation_deg', 0)); reversed_angle = (angle + 360) % 360 - 180
    alternatives = []
    for candidate in (angle, reversed_angle):
        glyph = alpha.rotate(-candidate, expand=True)
        base = min(span[0]/glyph.width, span[1]/glyph.height); best = None
        for scale in np.linspace(.8, 1.2, 17):
            size = tuple(max(1, round(s*base*scale)) for s in glyph.size)
            if size[0] > rgb.shape[1] or size[1] > rgb.shape[0]: continue
            template = np.asarray(glyph.resize(size, Image.Resampling.LANCZOS), 'float32')/255
            # Whole-window mass prevents a small fragment from winning by
            # matching one letter while ignoring the rest of the occurrence.
            dot = cv2.matchTemplate(ink, template, cv2.TM_CCORR)
            score = 2*dot/(float((ink*ink).sum()) + float((template*template).sum()) + 1e-9)
            _, value, _, offset = cv2.minMaxLoc(score)
            if best is None or value > best['source_ink_dice']:
                best = {'angle': candidate, 'source_ink_dice': float(value),
                        'probe_scale': float(scale), 'probe_offset': list(offset)}
        if best is None: return None
        alternatives.append(best)
    change = (alternatives[1]['source_ink_dice'] >= .8
              and alternatives[1]['source_ink_dice'] - alternatives[0]['source_ink_dice'] >= .1)
    return {'text_id': row['id'], 'action': 'reverse_reading_direction' if change else 'preserve',
            'source_probe_box': [*lo.tolist(), *hi.tolist()], 'alternatives': alternatives,
            'source_box_and_font_size_unchanged': True, 'probe_position_not_applied': True}


def correct(rows, source):
    from shape_integration.texts.layout import fit
    result = []; receipts = []
    for original in rows:
        receipt = observe(original, source)
        if receipt is not None: receipts.append(receipt)
        if receipt is None or receipt['action'] != 'reverse_reading_direction':
            result.append(original); continue
        row = deepcopy(original); measurement = deepcopy(row['layout']['measurement'])
        row['rotation_deg'] = receipt['alternatives'][1]['angle']
        if measurement.get('rotated_ink_bounds'):
            x0, y0, x1, y1 = measurement['rotated_ink_bounds']
            measurement['rotated_ink_bounds'] = [-x1, -y1, -x0, -y0]
        fitted = fit(row, measurement, row['layout']['font_fit_fraction'])
        if abs(fitted['font_size_px'] - original['font_size_px']) > 1e-6:
            raise ValueError('A reading-direction correction changed font size')
        if fitted['bbox_2d'] != original['bbox_2d'] or fitted['text'] != original['text']:
            raise ValueError('A reading-direction correction changed source content or geometry')
        fitted['source_direction_evidence'] = receipt
        result.append(fitted)
    return result, receipts
