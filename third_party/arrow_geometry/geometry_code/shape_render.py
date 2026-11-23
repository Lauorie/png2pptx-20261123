"""Arrow adapter for the shape pipeline's closed paths and fill/stroke contract.

DrawingML styling follows shape_integration/styling/export.py::apply_style.
Geometry/validation code is vendored unchanged in shape_core, with source hashes.
"""
from html import escape
from pptx.oxml.xmlchemy import OxmlElement
from pptx.dml.color import RGBColor
from shape_core.styling.geometry import gradient_line
from shape_core.fitting.paths import sample_commands


def mask_parts(row):
    contours=row.get('contours',[[row['start'],*row.get('via',[]),row['end']]])
    cmds=[]
    for pts in contours:
        cmds += [['M',*pts[0]],*[['L',*p] for p in pts[1:]],['Z']]
    return [dict(id='outline',commands=cmds,closed=True)]


def part_styles(row):
    supplied=row['style'].get('shape_parts',{})
    return {p['id']:supplied.get(p['id'],dict(fill=row['style']['color'],stroke=None,width_px=0,dash='solid'))
            for p in row.get('shape_parts') or mask_parts(row)}


def pattern(s):
    if not s.get('stroke') or s.get('dash','solid')=='solid':return []
    w=s['width_px']
    return s.get('dash_pattern_px') or {'dash':[4*w,3*w],'dot':[w,3*w],'dash_dot':[4*w,3*w,w,3*w]}[s['dash']]


def native_parts(slide,row,scale):
    styles=part_styles(row)
    for index,part in enumerate(row.get('shape_parts') or mask_parts(row)):
        commands=part['commands'];sample=sample_commands(commands)
        xy=[tuple(round((v+.5)*scale) for v in p) for p in sample]
        builder=slide.shapes.build_freeform(*xy[0],scale=1.)
        builder.add_line_segments(xy[1:],close=True)
        shape=builder.convert_to_shape();shape.name=row['id'] if index==0 else row['id']+'/'+part['id']
        path=shape._element.spPr.custGeom.pathLst[0]
        for child in list(path):path.remove(child)
        for command in commands:
            tag={'M':'moveTo','L':'lnTo','C':'cubicBezTo','Z':'close'}[command[0]]
            node=OxmlElement('a:'+tag)
            for j in range(1,len(command),2):
                pt=OxmlElement('a:pt')
                pt.set('x',str(round((command[j]+.5)*scale)-shape.left))
                pt.set('y',str(round((command[j+1]+.5)*scale)-shape.top));node.append(pt)
            path.append(node)
        s=styles[part['id']];fill=s['fill']
        if fill is None:shape.fill.background()
        elif isinstance(fill,str):
            shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(fill[1:])
        else:
            shape.fill.gradient();gf=shape._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}gradFill')
            for node in list(gf):gf.remove(node)
            gf.set('rotWithShape','1');stops=OxmlElement('a:gsLst')
            for pos,color in [(0,fill['start_color']),(100000,fill['end_color'])]:
                stop=OxmlElement('a:gs');stop.set('pos',str(pos));rgb=OxmlElement('a:srgbClr');rgb.set('val',color[1:]);stop.append(rgb);stops.append(stop)
            gf.append(stops);line=OxmlElement('a:lin');line.set('ang',str(round(fill['angle_deg']%360*60000)));line.set('scaled','0');gf.append(line);gf.append(OxmlElement('a:tileRect'))
        if s['stroke'] is None:shape.line.fill.background()
        else:
            shape.line.color.rgb=RGBColor.from_string(s['stroke'][1:]);shape.line.width=max(1,round(s['width_px']*scale))
            ln=shape._element.spPr.get_or_add_ln()
            for child in list(ln):
                if child.tag.rsplit('}',1)[-1] in ('prstDash','custDash'):ln.remove(child)
            seq=pattern(s);dash=OxmlElement('a:custDash' if seq else 'a:prstDash')
            if not seq:dash.set('val','solid')
            for a,b in zip(seq[::2],seq[1::2]):
                seg=OxmlElement('a:ds');seg.set('d',str(max(1,round(a/s['width_px']*100000))));seg.set('sp',str(max(1,round(b/s['width_px']*100000))));dash.append(seg)
            ln.append(dash);ln.set('cap','rnd' if s['dash']=='dot' else 'flat')
        for node in list(shape._element):
            if node.tag.rsplit('}',1)[-1]=='style':shape._element.remove(node)
        shape._element.spPr.append(OxmlElement('a:effectLst'))


def svg_parts(row,color_override=None,prefix='shape'):
    output=[];styles=part_styles(row)
    for i,p in enumerate(row.get('shape_parts') or mask_parts(row)):
        s=styles[p['id']];fill=color_override or s['fill'] or 'none'
        if isinstance(fill,dict):
            a,b=gradient_line(p,fill['angle_deg']);ident=prefix+str(i)
            output.append(f'<defs><linearGradient id="{ident}" gradientUnits="userSpaceOnUse" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"><stop offset="0" stop-color="{fill["start_color"]}"/><stop offset="1" stop-color="{fill["end_color"]}"/></linearGradient></defs>')
            fill='url(#'+ident+')'
        d=' '.join(c[0]+' '+' '.join(str(v) for v in c[1:]) for c in p['commands'])
        seq=pattern(s);dash='' if not seq else ' stroke-dasharray="'+','.join(map(str,seq))+'"'
        stroke=(color_override if s['stroke'] else None) or s['stroke'] or 'none'
        output.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="{stroke}" stroke-width="{s["width_px"]}"{dash}/>')
    return ''.join(output)
