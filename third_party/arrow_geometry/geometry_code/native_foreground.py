"""Remove leading PDF page backgrounds from actual native PPT vector output."""
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from svgpathtools import Line, parse_path
from transport.common import write_json


def matrix(value):
    if not value:
        return np.eye(3)
    match = re.fullmatch(r'\s*matrix\(([^)]+)\)\s*', value)
    if not match:
        raise ValueError('Unsupported native SVG transform')
    a, b, c, d, e, f = map(float, re.split(r'[\s,]+', match[1].strip()))
    return np.array([[a, c, e], [b, d, f], [0, 0, 1.]])


def full_white_page(element, transform, bounds):
    if element.tag.rsplit('}', 1)[-1] != 'path':
        return False
    if element.get('fill', '').lower() not in ('white', '#fff', '#ffffff'):
        return False
    if element.get('stroke', 'none') != 'none':
        return False
    path = parse_path(element.get('d', ''))
    if not path or not path.isclosed() or not all(isinstance(s, Line) for s in path):
        return False
    points = np.array([[s.start.real, s.start.imag, 1.] for s in path])
    points = (points @ transform.T)[:, :2]
    lo, hi = points.min(0), points.max(0)
    tolerance = max(.1, max(bounds[2]-bounds[0], bounds[3]-bounds[1])*1e-5)
    if not np.allclose(np.r_[lo, hi], bounds, atol=tolerance, rtol=0):
        return False
    area = abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
               np.dot(points[:, 1], np.roll(points[:, 0], -1)))/2
    rectangle = float(np.prod(hi-lo))
    return abs(area-rectangle) <= max(.01, rectangle*1e-6)


def strip_page_background(svg):
    root = ET.fromstring(svg)
    x, y, width, height = map(float, root.attrib['viewBox'].split())
    bounds = np.array([x, y, x+width, y+height])
    paint = []

    def visit(parent, transform):
        for element in parent:
            tag = element.tag.rsplit('}', 1)[-1]
            if tag in ('defs', 'clipPath', 'mask', 'metadata', 'title', 'desc'):
                continue
            total = transform @ matrix(element.get('transform'))
            if tag in ('g', 'a', 'svg'):
                visit(element, total)
            else:
                paint.append((parent, element, total))

    visit(root, np.eye(3))
    removed = 0
    for parent, element, transform in paint:
        if not full_white_page(element, transform, bounds):
            break
        parent.remove(element)
        removed += 1
    return ET.tostring(root, encoding='utf-8'), removed


def render_layer(svg_path, size, folder):
    """Use the exported native vectors, preserving head shape/color/size/fills."""
    data, removed = strip_page_background(Path(svg_path).read_bytes())
    png = cairosvg.svg2png(bytestring=data, output_width=size[0], output_height=size[1])
    layer = Image.open(io.BytesIO(png)).convert('RGBA')
    alpha = np.asarray(layer)[:, :, 3]
    if not (alpha == 0).any():
        raise ValueError('Native preview still has no transparent source context')
    folder = Path(folder)
    (folder/'foreground.svg').write_bytes(data)
    layer.save(folder/'foreground.png')
    write_json(folder/'foreground.json', dict(removed_leading_page_backgrounds=removed,
        native_vector_source=str(svg_path), output_pixel_size=list(size),
        transparent_fraction=float((alpha == 0).mean()), geometry_and_heads_from_native_ppt=True))
    return layer
