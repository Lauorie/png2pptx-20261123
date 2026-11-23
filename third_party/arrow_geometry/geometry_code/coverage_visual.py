"""Recovered geometry in blue on faded source; display changes never enter rows."""
import copy
import io
import xml.etree.ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import visual
from lib import fit, viz

BLUE = '#004cdb'
DESCRIPTION = ('Blue shows recovered paths, including their known heads, dashes and filled contours, '
               'on a faded gray source. Thin shafts are thickened for visibility only; coordinates, endpoints '
               'and gaps are unchanged. Gray source strokes are not reconstruction. Use ORIGINAL for exact appearance; '
               'blue results can still be wrong.')


def project(rows, view):
    matrix = np.asarray(view['global_to_view'], float)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2,:2]))))
    def xy(points):
        p=np.asarray(points,float);shape=p.shape
        return (np.c_[p.reshape(-1,2),np.ones(p.size//2)] @ matrix.T)[:,:2].reshape(shape).tolist()
    result=[]
    for source in rows:
        r=fit.transform(source,view,to_global=False);p=xy(fit.points(source))
        r.update(start=p[0],via=p[1:-1],end=p[-1])
        if 'points' in r:r['points']=p
        # fit.transform already projected curves, contours and shape commands.
        s=r.setdefault('style',{})
        width=source.get('style',{}).get('width_px') or source.get('width_estimate')
        s['width_px']=float(width)*scale if width and np.isfinite(width) and width>0 else 2.
        s['color']=BLUE
        dash=s.setdefault('dash',{'preset':{'dashed':'dash','dotted':'sysDot'}.get(r.get('line_style'),'solid')})
        result.append(r)
    return result


def render(crop, rows, view, path, labels=True):
    projected=project(rows,view)
    # Reuse the same shaft, dash, marker and filled-contour renderer as results.
    # colored=True also makes geometry with pending appearance visible as a guide.
    root=ET.fromstring(viz.svg(crop.size,projected,colored=True))
    minimum=max(2.,max(crop.size)/850.)
    for element in root.iter():
        for attr in ['fill','stroke']:
            if element.get(attr) not in (None,'none'):element.set(attr,BLUE)
        # Only widen the visible shaft. Invisible marker carriers retain their
        # original width, so known head sizes and their attachment do not change.
        if element.tag.endswith('path') and element.get('id') is not None and element.get('stroke')==BLUE:
            element.set('stroke-width',str(max(minimum,float(element.get('stroke-width','0')))))
    svg=ET.tostring(root,encoding='unicode')
    layer=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGBA')
    gray=ImageOps.grayscale(crop).convert('RGB')
    image=Image.blend(gray,Image.new('RGB',crop.size,'white'),.62).convert('RGBA')
    image.alpha_composite(layer);image=image.convert('RGB')
    if labels:
        draw=ImageDraw.Draw(image);rect=(0,0,image.width,image.height);seen=set()
        for r in projected:
            label=r.get('short_id',r.get('label',r['id']))
            if label in seen:continue
            visible=[];p=fit.points(r)
            for a,b in zip(p,p[1:]):
                ok,a,b=cv2.clipLine(rect,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
                if ok:visible.extend([a,b])
            if visible:visual.label(draw,visible[len(visible)//2],label,BLUE,image);seen.add(label)
    image.save(path)
