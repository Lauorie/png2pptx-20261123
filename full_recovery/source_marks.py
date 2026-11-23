"""Recover isolated original cross marks as editable native silhouettes.

Four deep concavities, quarter-turn symmetry and a blank native neighbourhood
are required. Arrow bodies, heads, junction fragments and misplaced existing
crosses do not authorize new marks. No sample IDs or semantic word guesses.
"""
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon


def recover(source, native, candidates):
    src = np.asarray(source.convert('RGB')); restored = np.asarray(native.convert('RGB'))
    result = []
    for row in candidates:
        x0, y0, x1, y1 = row['bbox_2d']; w, h = x1-x0+1, y1-y0+1
        if min(w, h) < 7 or not .7 <= w/h <= 1.4:
            continue
        pigment = np.asarray([int(row['color'][i:i+2], 16) for i in (1, 3, 5)])
        tile = src[y0:y1+1, x0:x1+1].astype(float)
        mask = (np.linalg.norm(tile-pigment, axis=2)<75).astype('uint8')
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or hierarchy is None or any(r[3]>=0 for r in hierarchy[0]):
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < .2*w*h:
            continue
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull)<4:
            continue
        try:
            defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            continue
        if defects is None or sum(d[0, 3]/256 >= .08*min(w,h) for d in defects) != 4:
            continue
        side = max(w,h)+4; square = np.zeros((side,side), 'uint8')
        ox, oy = (side-w)//2, (side-h)//2; square[oy:oy+h, ox:ox+w] = mask
        rotated = cv2.warpAffine(square, cv2.getRotationMatrix2D(((side-1)/2,(side-1)/2),90,1),
                                 (side,side), flags=cv2.INTER_NEAREST)
        symmetry = float(((square>0)&(rotated>0)).sum()/max(1,((square>0)|(rotated>0)).sum()))
        if symmetry < .7:
            continue
        pad = max(4, round(.4*max(w,h)))
        xa, ya = max(0,x0-pad), max(0,y0-pad); xb, yb = min(source.width,x1+pad+1), min(source.height,y1+pad+1)
        matches = (np.linalg.norm(restored[ya:yb, xa:xb].astype(float)-pigment, axis=2)<55).astype('float32')
        nearby = cv2.matchTemplate(matches, mask.astype('float32'), cv2.TM_CCORR)
        nearby_coverage = float(nearby.max()/max(1,mask.sum()))
        at_source = matches[y0-ya:y0-ya+h, x0-xa:x0-xa+w]
        if nearby_coverage > .6 or np.mean(at_source[mask>0]) > .15:
            continue
        approximation = cv2.approxPolyDP(contour,max(.4,.015*min(w,h)),True).reshape(-1,2)
        poly = Polygon(approximation)
        if not poly.is_valid:
            continue
        shape = poly.buffer(.5,join_style=2).simplify(.2,preserve_topology=True)
        if shape.geom_type != 'Polygon':
            continue
        points = np.asarray(shape.exterior.coords)[:-1]+[x0,y0]
        result.append({'id':'SOURCE_MARK_%04d'%(len(result)+1), 'kind':'isolated_cross',
            'bbox_2d':row['bbox_2d'], 'points':points.tolist(), 'color':row['color'],
            'evidence':{'source_ink_pixels':int(mask.sum()),'four_concavities':True,
                        'quarter_turn_iou':round(symmetry,4),'native_nearby_coverage':round(nearby_coverage,4)}})
    return result


def append(path, marks, size):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
    from pptx.dml.color import RGBColor
    from pptx.oxml.xmlchemy import OxmlElement
    from shape_integration.fitting.scene import custom_geometry
    if not marks:
        return
    prs = Presentation(str(path)); slide = prs.slides[0]; scale = prs.slide_width/size[0]
    for row in marks:
        points = np.asarray(row['points']); lo, hi = points.min(0), points.max(0)
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,round((lo[0]+.5)*scale),
            round((lo[1]+.5)*scale),max(1,round((hi[0]-lo[0])*scale)),max(1,round((hi[1]-lo[1])*scale)))
        shape.name = row['id']
        custom_geometry(shape,[['M',*points[0]],*[['L',*p] for p in points[1:]],['Z']],[*lo,*hi])
        shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor.from_string(row['color'][1:]); shape.line.fill.background()
        shape._element.spPr.append(OxmlElement('a:effectLst'))
        for style in shape._element.xpath('./p:style'):
            shape._element.remove(style)
    target = Path(path); temp = target.with_name(target.stem+'.source_marks.pptx'); prs.save(temp); temp.replace(target)
