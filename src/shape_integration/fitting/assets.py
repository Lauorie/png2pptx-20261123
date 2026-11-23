"""Show the same registered image footprints to observation and review."""
import shutil

from PIL import Image, ImageDraw, ImageFont

from ..common import read, save, sha
from .frames import FitFrame
from .views import FONT

ASSET_COLOR = '#008eb8'
IMAGE_OWNERSHIP = (
    'Cyan IMAGE footprints mark registered raster content; do not reconstruct the same content again as shape. '
    'A footprint owns only that content, '
    'not every overlapping object. Independent enclosing frames and their exposed contours or bands '
    'remain shape-owned. Other geometry follows the ordinary shape/arrow ownership rules.')


def annotate_assets(canvas, frame, assets):
    """Draw footprint hints without treating their clipped edges as contours."""
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, max(12, round(max(canvas.size)/85)))
    shown = []
    for asset in assets:
        box = asset['bbox_2d']
        local = frame.to_local(box[:2]) + frame.to_local(box[2:])
        if local[2] < 0 or local[3] < 0 or local[0] > canvas.width-1 or local[1] > canvas.height-1:
            continue
        clipped = [max(0., min(value, canvas.size[i % 2]-1)) for i, value in enumerate(local)]
        draw.rectangle(clipped, outline=ASSET_COLOR, width=3)
        label = 'IMAGE '+asset['id']
        text_box = draw.textbbox((0, 0), label, font=font)
        width, height = text_box[2]-text_box[0]+6, text_box[3]-text_box[1]+6
        x = max(0., min(clipped[0], canvas.width-width))
        y = max(0., min(clipped[1]-height-2, canvas.height-height))
        draw.rectangle((x, y, x+width, y+height), fill='white')
        draw.text((x+3, y+3-text_box[1]), label, fill=ASSET_COLOR, font=font)
        shown.append({'id': asset['id'], 'label': label, 'bbox_original': box,
                      'bbox_local': local, 'clipped_bbox_local': clipped})
    return shown


def render_asset_source(folder):
    source = folder/'clean.png'; target = folder/'critic_source.png'
    registry = folder.parent/'image_assets.json'
    assets = read(registry) if registry.exists() else []
    frame = FitFrame.from_metadata(read(folder/'frame.json'))
    with Image.open(source) as image:
        canvas = image.convert('RGB')
    shown = annotate_assets(canvas, frame, assets)
    if shown:
        canvas.save(target)
    else:
        shutil.copy2(source, target)
    save(folder/'critic_assets.json', {'coordinate_frame': 'local_pixel_centers',
         'image_size': list(canvas.size), 'assets': shown,
         'source_sha256': sha(source), 'render_sha256': sha(target)})
