"""Same-call source details for the existing rotated-formula view.

The original CLEAN view remains the coordinate and orientation authority.
These source crops help read glyphs, orientation and script placement; they never supply
new geometry, choose a formula, or add a model stage.
"""
from pathlib import Path
import math

from PIL import Image, ImageDraw


def images(units, clean_path, folder):
    from shape_integration.common import save
    folder = Path(folder)
    rows = []
    with Image.open(clean_path) as opened:
        clean = opened.convert('RGB')
        for unit in units:
            predicted = unit.get('rotation_deg', 0)
            if (unit.get('kind') != 'formula' or not isinstance(predicted, (int, float))
                    or not math.isfinite(predicted) or abs(predicted) < 1):
                continue
            angle = 0  # Never present an old predicted rotation as source evidence.
            box = unit.get('locked_bbox_2d', [])
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            margin = max(2, .1*min(box[2]-box[0], box[3]-box[1]))
            region = (max(0, math.floor(box[0]-margin)), max(0, math.floor(box[1]-margin)),
                      min(clean.width, math.ceil(box[2]+margin+1)), min(clean.height, math.ceil(box[3]+margin+1)))
            if region[2]<=region[0] or region[3]<=region[1]:continue
            crop = clean.crop(region)
            scale = min(3., 1024/crop.width, 256/crop.height)
            crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))), Image.Resampling.LANCZOS)
            rows.append((unit['id'], angle, region, crop))
    if not rows:
        return []
    sheet = Image.new('RGB', (1040, sum(crop.height+32 for _, _, _, crop in rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    top = 0
    for ident, angle, _, crop in rows:
        draw.text((8, top+4), f'{ident}: enlarged source pixels, original orientation; coordinates remain in CLEAN', fill='black')
        sheet.paste(crop, (8, top+26))
        top += crop.height+32
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'source_reading_details.png'
    sheet.save(path)
    save(folder/'source_reading_details.json', {'views': [
        {'id': ident, 'display_rotation_deg': angle, 'source_crop_half_open': list(region)}
        for ident, angle, region, _ in rows], 'extra_model_calls': 0,
        'geometry_coordinates': 'original CLEAN only', 'transcription_authority': 'source pixels'})
    return [('Source glyph details in original orientation; distinguish baseline rotation, upright stacking and scripts. Coordinates remain in CLEAN.', path)]
