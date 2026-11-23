"""Visible candidate callouts; annotation padding never changes recovery bounds."""
import math

import numpy as np
from PIL import ImageDraw, ImageFont

ORANGE = '#d85b00'
PURPLE = '#a000c0'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def display_bbox(bbox, size, context=False):
    x0, y0, x1, y1 = map(float, bbox)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        raise ValueError('Candidate bbox must have finite positive extent')
    # Pad thin targets in display pixels too: a fraction alone is invisible.
    minimum = 24. if context else 30.
    pad = 5. if context else 7.
    bounds = []
    for lo, hi, side in ((x0, x1, size[0]), (y0, y1, size[1])):
        extra = min(14., max(pad, (hi-lo)*.05))
        width = min(float(side-1), max(minimum, hi-lo+2*extra))
        left = min(max(0., (lo+hi-width)/2), max(0., side-1-width))
        bounds.append((left, left+width))
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]


def annotate(image, bbox, *, context=False, crop_box=None):
    """Return a marked copy and audit metadata, leaving pixels/coords untouched upstream."""
    out = image.convert('RGB').copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    if crop_box is not None:
        b = [max(0., min(float(v), (w-1 if i%2 == 0 else h-1))) for i, v in enumerate(crop_box)]
        draw.rectangle(b, outline=PURPLE, width=2)
    box = display_bbox(bbox, out.size, context)
    x0, y0, x1, y1 = box
    width = max(2, min(4, round(max(out.size)/480)))
    leg = min(17., (x1-x0)/3, (y1-y0)/3)
    for x, y, sx, sy in ((x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)):
        pts = [(x+sx*leg,y),(x,y),(x,y+sy*leg)]
        draw.line(pts, fill='white', width=width+2)
        draw.line(pts, fill=ORANGE, width=width)
    font_size = max(18, min(26, round(max(out.size)/55)))
    font = ImageFont.truetype(FONT, font_size)
    text_box = draw.textbbox((0,0), 'Q', font=font)
    tw, th = text_box[2]-text_box[0]+14, text_box[3]-text_box[1]+10
    gap = 18 if context else 10
    candidates = [(x1+gap,y0-th-gap),(x0-tw-gap,y0-th-gap),
                  (x1+gap,y1+gap),(x0-tw-gap,y1+gap),
                  ((x0+x1-tw)/2,y0-th-gap),((x0+x1-tw)/2,y1+gap),
                  (x1+gap,(y0+y1-th)/2),(x0-tw-gap,(y0+y1-th)/2),
                  (3,3),(w-tw-3,3),(3,h-th-3),(w-tw-3,h-th-3)]
    gray = np.asarray(image.convert('L'))
    def score(point):
        x = int(max(0, min(w-tw-1, point[0])))
        y = int(max(0, min(h-th-1, point[1])))
        rect = [x,y,x+tw,y+th]
        # Place the short label away from target ink, preferring nearby white space.
        area = max(1,tw*th)
        overlap = max(0,min(rect[2],x1)-max(x,x0))*max(0,min(rect[3],y1)-max(y,y0))/area
        ink = float(np.mean(gray[y:y+th,x:x+tw] < 210))
        dx = max(x0-rect[2], rect[0]-x1, 0)
        dy = max(y0-rect[3], rect[1]-y1, 0)
        return overlap*20 + ink*2 + math.hypot(dx,dy)/max(w,h), rect
    _, label = min((score(p) for p in candidates), key=lambda v:v[0])
    # Local crops need no extra stroke that could be mistaken for a connection.
    # Keep the leader only in the whole-image context to locate tiny candidates.
    leader = []
    if context:
        cx, cy = (label[0]+label[2])/2, (label[1]+label[3])/2
        tip = min(((x0,y0),(x1,y0),(x0,y1),(x1,y1)),key=lambda p:math.hypot(p[0]-cx,p[1]-cy))
        start = (max(label[0],min(label[2],tip[0])),max(label[1],min(label[3],tip[1])))
        draw.line([start,tip], fill='white', width=width+2)
        draw.line([start,tip], fill=ORANGE, width=max(2,width-1))
        leader = [list(start), list(tip)]
    draw.rounded_rectangle(label, radius=4, fill='white', outline=ORANGE, width=2)
    draw.text((label[0]+7-text_box[0],label[1]+5-text_box[1]),'Q',font=font,fill=ORANGE)
    return out, dict(id='Q', candidate_bbox=list(bbox), display_bbox=box,
                     label_bbox=label, leader=leader, image_size=[w,h],
                     annotation_only=True, crop_box=crop_box)
