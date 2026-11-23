"""Assemble immutable object revisions in original coordinates; export geometry."""
import copy
import json
import math
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from ..common import read, save, sha
from .frames import FitFrame
from .paths import sample_commands
from .views import PALETTE


def native_parameters(part, frame):
    """Exact affine primitive transport; never average unequal x/y scales."""
    kind = part['kind']; p = part.get('parameters_local', {})
    if kind not in ('rectangle', 'rounded_rectangle', 'ellipse'):
        return None
    f = FitFrame.from_metadata(frame)
    sx, sy = [(f.crop_bounds[i+2]-f.crop_bounds[i])/f.input_size[i]*f.original_size[i]/f.global_size[i] for i in (0, 1)]
    a = p['angle_radians']; rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    axes = np.diag([sx, sy]) @ rotation @ np.diag([p['width'], p['height']])
    center = f.to_original(p['center'])
    if kind == 'ellipse':
        u, lengths, _ = np.linalg.svd(axes)
        return dict(center=center, width=float(lengths[0]), height=float(lengths[1]),
                    angle_radians=math.atan2(u[1, 0], u[0, 0]))
    width, height = np.linalg.norm(axes, axis=0)
    if abs(float(axes[:, 0] @ axes[:, 1])) > 1e-10*width*height:
        return None
    if kind == 'rounded_rectangle' and not math.isclose(sx, sy, rel_tol=1e-10):
        return None  # Affine corner arcs are no longer circular.
    return dict(center=center, width=float(width), height=float(height),
                angle_radians=math.atan2(axes[1, 0], axes[0, 0]), radius=p.get('radius', 0)*sx)


def assemble(folder, item, candidates, records, upstream, coverage):
    scene = {'schema_version': 'shape_integration.scene.1', 'case_id': item['id'],
             'coordinate_frame': 'original_pixel_centers', 'source_size': item['source_size'],
             'source_sha256': item['source_sha256'], 'upstream': upstream,
             'objects': {}, 'points': {}, 'parts': {}, 'paths': {}, 'unresolved': [],
             'coverage': coverage, 'visual_accuracy_verified': False}
    for candidate in candidates:
        oid = candidate['id']; record = records.get(oid, {'state': 'pending', 'selected_revision': None, 'history': []})
        obj = {'id': oid, 'candidate': candidate, **copy.deepcopy(record), 'part_keys': [], 'point_keys': []}
        scene['objects'][oid] = obj
        if record['state'] != 'accepted_by_model':
            scene['unresolved'].append({'object_id': oid, 'state': record['state'], 'reason': record.get('reason', record.get('error', ''))})
        revision = record.get('selected_revision')
        if not revision:
            continue
        fit_path = folder/'objects'/oid/revision/'fit_original.json'
        fit = read(fit_path)
        if fit['object_id'] != oid or fit['coordinate_frame'] != 'original_pixel_centers':
            raise ValueError('Invalid assembled fit identity/frame')
        obj.update(fit_path=str(fit_path.relative_to(folder)), fit_sha256=sha(fit_path),
                   frame=fit['frame'], fit_status=fit['fit_status'], observation_status=fit['observation_status'],
                   completeness_status=fit['completeness_status'], shared_point_diagnostics=fit['shared_point_diagnostics'])
        for point in fit['points']:
            key = json.dumps(point['key'], ensure_ascii=False)
            scene['points'][key] = point; obj['point_keys'].append(key)
        for part in fit['parts']:
            part = copy.deepcopy(part); key = json.dumps(part['key'], ensure_ascii=False)
            part['point_keys'] = [json.dumps([*part['key'][:2], pid], ensure_ascii=False) for pid in part['observed_point_ids']]
            if any(pid not in scene['points'] for pid in part['point_keys']):
                raise ValueError('Scene point ownership reference missing')
            if part['status'] == 'ok':
                part['native_parameters_original'] = native_parameters(part, fit['frame'])
                scene['paths'][key] = {'key': part['key'], 'commands': part['commands'], 'object_id': oid}
            scene['parts'][key] = part; obj['part_keys'].append(key)
    scene['counts'] = {'candidates': len(candidates), 'terminal_objects': sum(r['state'] != 'pending' for r in records.values()),
                       'accepted_by_model': sum(r['state'] == 'accepted_by_model' for r in records.values()),
                       'successful_parts': sum(p['status'] == 'ok' for p in scene['parts'].values()),
                       'failed_parts': sum(p['status'] != 'ok' for p in scene['parts'].values()),
                       'unresolved_objects': len(scene['unresolved'])}
    save(folder/'scene.json', scene)
    return scene


def exportable(scene, accepted_only=False):
    """Keep the selected usable geometry regardless of the review outcome."""
    for index, obj in enumerate(scene['objects'].values()):
        if accepted_only and obj['state'] != 'accepted_by_model':
            continue
        parts = [scene['parts'][key] for key in obj['part_keys'] if scene['parts'][key]['status'] == 'ok']
        yield obj, parts, PALETTE[index % len(PALETTE)]


def svg(scene, accepted_only=False):
    w, h = scene['source_size']
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="-0.5 -0.5 {w} {h}">',
             '<rect x="-0.5" y="-0.5" width="100%" height="100%" fill="white"/>']
    for obj, parts, color in exportable(scene, accepted_only):
        lines.append('<g id='+json.dumps(obj['id'])+' data-state='+json.dumps(obj['state'])+'>')
        for p in parts:
            path = ' '.join(c[0]+' '+' '.join(format(v, '.12g') for v in c[1:]) for c in p['commands'])
            ident = escape(obj['id']+'/'+p['id'], {'"': '&quot;'})
            lines.append(f'<path id="{ident}" d="{path}" fill="none" stroke="{color}" stroke-width="1.2"/>')
        lines.append('</g>')
    return '\n'.join(lines+['</svg>'])


def custom_geometry(shape, commands, bounds):
    bx, by, ex, ey = bounds; unit = 10000
    geom = OxmlElement('a:custGeom')
    for name in ['avLst', 'gdLst', 'ahLst', 'cxnLst']:
        geom.append(OxmlElement('a:'+name))
    rect = OxmlElement('a:rect')
    for k, v in {'l': '0', 't': '0', 'r': 'r', 'b': 'b'}.items():
        rect.set(k, v)
    geom.append(rect); paths = OxmlElement('a:pathLst'); path = OxmlElement('a:path')
    path.set('w', str(max(1, round((ex-bx)*unit)))); path.set('h', str(max(1, round((ey-by)*unit))))
    path.set('fill', 'norm' if any(c[0] == 'Z' for c in commands) else 'none')
    for cmd in commands:
        node = OxmlElement('a:'+{'M': 'moveTo', 'L': 'lnTo', 'C': 'cubicBezTo', 'Z': 'close'}[cmd[0]])
        for i in range(1, len(cmd), 2):
            pt = OxmlElement('a:pt'); pt.set('x', str(round((cmd[i]-bx)*unit))); pt.set('y', str(round((cmd[i+1]-by)*unit))); node.append(pt)
        path.append(node)
    paths.append(path); geom.append(paths)
    sppr = shape._element.spPr
    sppr.replace(sppr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom'), geom)


def write_pptx(scene, path, accepted_only=False, *, style_writer=None):
    prs = Presentation(); w, h = scene['source_size']; scale = Inches(13.333333)/w
    prs.slide_width, prs.slide_height = round(w*scale), round(h*scale)
    slide = prs.slides.add_slide(prs.slide_layouts[6]); counts = {'native': 0, 'custom': 0, 'groups': 0}
    for obj, parts, color in exportable(scene, accepted_only):
        target = slide.shapes
        if len(parts) > 1:
            group = slide.shapes.add_group_shape(); group.name = obj['id']; target = group.shapes; counts['groups'] += 1
        for part in parts:
            p = part['native_parameters_original']; kind = part['kind']
            if p:
                cx, cy = p['center']; width, height = p['width'], p['height']
                typ = {'ellipse': MSO_AUTO_SHAPE_TYPE.OVAL, 'rectangle': MSO_AUTO_SHAPE_TYPE.RECTANGLE,
                       'rounded_rectangle': MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE}[kind]
                shape = target.add_shape(typ, round((cx-width/2+.5)*scale), round((cy-height/2+.5)*scale),
                                         max(1, round(width*scale)), max(1, round(height*scale)))
                shape.rotation = math.degrees(p['angle_radians']) % 360
                if kind == 'rounded_rectangle':
                    shape.adjustments[0] = p['radius']/min(width, height)
                counts['native'] += 1
            else:
                points = [(c[i], c[i+1]) for c in part['commands'] for i in range(1, len(c), 2)]
                xs, ys = zip(*points); bx, by = min(xs), min(ys); ex, ey = max(max(xs), bx+1e-3), max(max(ys), by+1e-3)
                shape = target.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, round((bx+.5)*scale), round((by+.5)*scale),
                                         max(1, round((ex-bx)*scale)), max(1, round((ey-by)*scale)))
                custom_geometry(shape, part['commands'], (bx, by, ex, ey)); counts['custom'] += 1
            shape.name = obj['id']+'/'+part['id']+' ['+obj['state']+']'; shape.fill.background()
            shape.line.color.rgb = RGBColor.from_string(color[1:]); shape.line.width = Pt(1.2*13.333333*72/w)
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            style = shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
            if style is not None:
                shape._element.remove(style)
            if style_writer is not None:
                style_writer(shape, part, scale)
    prs.save(path)
    return counts


def export(scene, folder, source):
    import cairosvg
    with Image.open(source) as im:
        view = im.convert('RGB')
    draw = ImageDraw.Draw(view)
    for obj, parts, color in exportable(scene):
        for part in parts:
            pts = [tuple(p) for p in sample_commands(part['commands'], 128)]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
    view.save(folder/'fit_overlay.png')
    counts = {}
    for name, accepted in [('geometry_final', False), ('geometry_accepted', True)]:
        content = svg(scene, accepted); (folder/(name+'.svg')).write_text(content)
        cairosvg.svg2png(bytestring=content.encode(), write_to=str(folder/(name+'.png')))
        counts[name] = write_pptx(scene, folder/(name+'.pptx'), accepted)
    # Preserve existing diagnostic links without rendering the same scene twice.
    for suffix in ['svg', 'png', 'pptx']:
        shutil.copyfile(folder/('geometry_final.'+suffix), folder/('geometry_diagnostic.'+suffix))
    counts['geometry_diagnostic'] = dict(counts['geometry_final'])
    save(folder/'export.json', {'counts': counts, 'primary_output': 'geometry_final',
        'selection_policy': 'All successful parts from each selected_revision, regardless of object state; review states are preserved.',
        'source_styles_restored': False,
        'powerpoint_ui_editing_verified': False,
        'ellipse_path_note': 'SVG uses the fitted cubic ellipse approximation; native PPT uses the analytic affine ellipse. Cubic approximation radial error is approximately 0.0273 percent before affine transport.'})
    return counts
