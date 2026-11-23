"""Review observed points and fitted parts on the exact frozen clean crop."""
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont

from .paths import command_strokes

PALETTE = ['#d10070', '#007abb', '#00865d', '#bd6300', '#7138cc']
REVIEW_PALETTE = PALETTE + ['#92462f', '#008c99', '#657300', '#9d4b98', '#4c5361',
                            '#2846a0', '#a34039']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def repeated_details(local, job, folder):
    """Bounded, aligned detail views for the existing geometry review.

    A network can also be a long repeated layout. Connections must not replace
    node-layout evidence. Each image has its own readable size; no tall contact
    sheet is downscaled back to an unreadable overview.
    """
    from PIL import ImageOps
    from ..common import save
    from .frames import FitFrame
    frame=FitFrame.from_metadata(job['frame'])
    b=job['candidate']['bbox_2d']
    box=frame.to_local(b[:2])+frame.to_local(b[2:])
    for part in local['parts']:
        if part['status']!='ok':continue
        p=part['bbox_2d']
        box=[min(box[0],p[0]),min(box[1],p[1]),max(box[2],p[2]),max(box[3],p[3])]
    pad=max(8,.04*min(box[2]-box[0],box[3]-box[1]))
    w,h=frame.input_size
    box=[max(0,math.floor(box[0]-pad)),max(0,math.floor(box[1]-pad)),
         min(w,math.ceil(box[2]+pad)+1),min(h,math.ceil(box[3]+pad)+1)]
    ext=[box[2]-box[0],box[3]-box[1]];axis=int(ext[1]>ext[0])
    count=min(4,max(1,math.ceil(ext[axis]/max(1,ext[1-axis])/1.5)))
    if count==1:return []
    images=[];records=[]
    with Image.open(folder/'clean.png') as clean:
        # The full white-background fit remains a separate input. Local
        # overlays make small displacements visible without asking the critic
        # to mentally register two long repeated columns.
        overlay=clean.convert('RGB');ink=ImageDraw.Draw(overlay)
        for n,part in enumerate(local['parts']):
            if part['status']!='ok':continue
            paths=list(command_strokes(part['commands'],64))
            colour=REVIEW_PALETTE[n%len(REVIEW_PALETTE)]
            for path in paths:
                if len(path)>1:ink.line([tuple(p) for p in path],fill=colour,width=1)
            if paths and len(paths[0]):
                ink.text(tuple(paths[0][0]),f'F{n+1}',fill=colour,stroke_width=1,stroke_fill='white')
        for i in range(count):
            region=box.copy()
            region[axis]=max(box[axis],math.floor(box[axis]+(i-.08)*ext[axis]/count))
            region[axis+2]=min(box[axis+2],math.ceil(box[axis]+(i+1.08)*ext[axis]/count))
            sheet=Image.new('RGB',(1000,560),'white');draw=ImageDraw.Draw(sheet)
            for col,(label,im) in enumerate([('CLEAN',clean),('PREDICTED OUTLINES on source',overlay)]):
                draw.text((col*500+5,5),label,fill='black')
                tile=ImageOps.contain(im.convert('RGB').crop(region),(490,530),Image.Resampling.LANCZOS)
                sheet.paste(tile,(col*500+(500-tile.width)//2,25+(530-tile.height)//2))
            path=folder/f'repeated_detail_{i:02d}.png';sheet.save(path)
            images.append((f'Layout detail {i+1}: CLEAN / colored prediction overlay; source pixels are not recovered geometry; crop {region}',path))
            records.append({'crop_input_px':region,'image':path.name})
    save(folder/'repeated_details.json',{'views':records,'extra_model_calls':0})
    return images


def geometry_only(local,size,output,width=3,*,labels=True):
    """A prediction-only view must never inherit source fills or require point metadata."""
    image=Image.new('RGB',tuple(size),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(FONT,max(12,image.width//90))
    for i,part in enumerate(local['parts']):
        if part['status']!='ok':continue
        paths=command_strokes(part['commands'],64);color=REVIEW_PALETTE[i%len(REVIEW_PALETTE)]
        points=[]
        for path in paths:
            coords=[tuple(p) for p in path];points.extend(coords)
            if len(coords)>1:draw.line(coords,fill=color,width=width)
        if points and labels:
            x=sum(p[0] for p in points)/len(points);y=sum(p[1] for p in points)/len(points)
            draw.text((max(0,min(x,image.width-45)),max(0,min(y,image.height-20))),
                      f'F{i+1}',fill=color,font=font,stroke_width=1,stroke_fill='white')
    image.save(output)


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
        for stroke in command_strokes(part['commands'], 256):
            coords = [xy(p) for p in stroke]
            if len(coords) > 1:
                for image in (overlay, geometry):
                    ImageDraw.Draw(image).line(coords, fill=PALETTE[i % len(PALETTE)], width=2)
    images = [clean, observed, overlay, geometry]
    board = Image.new('RGB', (clean.width*4, clean.height+32), 'white')
    draw = ImageDraw.Draw(board)
    for i, (im, title) in enumerate(zip(images, ['Clean', 'Observed parts', 'Fit overlay', 'Geometry'])):
        board.paste(im, (i*clean.width, 32))
        draw.text((i*clean.width+4, 8), title, font=font, fill='black')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)
