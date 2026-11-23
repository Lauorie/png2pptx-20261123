"""Review observed points and fitted parts on the exact frozen clean crop."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .paths import sample_commands

PALETTE = ['#d10070', '#007abb', '#00865d', '#bd6300', '#7138cc']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def inspection_style(size, sizing):
    """Line width and point radius grow linearly from 3 to 6 crop pixels."""
    low = sizing.get('crop_min_long_edge', 1080)
    high = sizing.get('crop_max_long_edge', 2048)
    edge = max(size)
    fraction = (edge-low)/(high-low) if high > low else float(edge >= high)
    pixels = 3 + 3*max(0., min(1., fraction))
    # Pillow strokes use whole pixels; keep the continuous value for auditing.
    drawn = int(pixels + .5)
    return {'version': 'shape_integration.critic_visual.1',
            'crop_min_long_edge': low, 'crop_max_long_edge': high,
            'image_long_edge_px': edge, 'interpolated_size_px': pixels,
            'line_width_px': drawn, 'point_radius_px': drawn,
            'point_marker': 'hollow', 'point_outline_width_px': 2}


def comparison(clean_path, raw, result, output_path):
    clean = Image.open(clean_path).convert('RGB')
    size = clean.size
    clean.thumbnail((540, 540), Image.Resampling.LANCZOS)
    sx, sy = clean.width/size[0], clean.height/size[1]
    def xy(point):
        return ((point[0]+.5)*sx-.5, (point[1]+.5)*sy-.5)
    observed = clean.copy(); overlay = clean.copy()
    geometry = Image.new('RGB', clean.size, 'white')
    table = {p['id']: p['xy'] for p in raw.get('points', [])}
    font = ImageFont.truetype(FONT, 11)
    for i, part in enumerate(raw.get('parts', [])):
        color = PALETTE[i % len(PALETTE)]
        refs = part.get('point_ids', [p for s in part.get('segments', []) for p in s['point_ids']])
        coords = [xy(table[pid]) for pid in dict.fromkeys(refs) if pid in table]
        draw = ImageDraw.Draw(observed)
        for x, y in coords:
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        if coords:
            draw.text(coords[0], part['id'], fill=color, font=font)
    for i, part in enumerate((result or {}).get('parts', [])):
        if part['status'] != 'ok':
            continue
        coords = [xy(p) for p in sample_commands(part['commands'], 256)]
        for image in (overlay, geometry):
            draw = ImageDraw.Draw(image)
            draw.line(coords, fill=PALETTE[i % len(PALETTE)], width=2)
    images = [clean, observed, overlay, geometry]
    board = Image.new('RGB', (clean.width*4, clean.height+32), 'white')
    draw = ImageDraw.Draw(board)
    for i, (im, title) in enumerate(zip(images, ['Clean', 'Observed parts', 'Fit overlay', 'Geometry'])):
        board.paste(im, (i*clean.width, 32))
        draw.text((i*clean.width+4, 8), title, font=font, fill='black')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)
