"""Native editable styles plus an actual PPT render, with invariant geometry audit."""
import copy
import math
import shutil
import threading
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import fitz
from lxml import etree
from PIL import Image
from pptx.oxml.xmlchemy import OxmlElement
from pptx.dml.color import RGBColor
from ..common import save, read, sha, digest, compare
from ..resource_limits import bounded
from ..fitting.scene import write_pptx
from .geometry import part_key, gradient_line, fill_frame

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
PDF_CROP_PROTOCOL='direct_pdf_crop.1'
# PyMuPDF document operations must not run concurrently in the case threads.
_PDF_LOCK=threading.Lock()


def pattern(style):
    if not style.get('stroke') or style['dash']=='solid':return None
    w=style['width_px']
    return style.get('dash_pattern_px') or {'dash':[4*w,3*w],'dot':[w,3*w],'dash_dot':[4*w,3*w,w,3*w]}[style['dash']]


def apply_style(shape, part, scale, style):
    f=style['fill']
    if f is None:
        if hasattr(shape,'fill'):shape.fill.background()
    elif isinstance(f,str):
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(f[1:])
    else:
        shape.fill.gradient()
        gf=shape._element.spPr.find('a:gradFill',NS)
        for node in list(gf): gf.remove(node)
        gf.set('rotWithShape','1');gslist=OxmlElement('a:gsLst')
        for position,color in [(0,f['start_color']),(100000,f['end_color'])]:
            stop=OxmlElement('a:gs');stop.set('pos',str(position))
            rgb=OxmlElement('a:srgbClr');rgb.set('val',color[1:]);stop.append(rgb);gslist.append(stop)
        gf.append(gslist);linear=OxmlElement('a:lin')
        rotation=math.degrees(fill_frame(part)[3])
        linear.set('ang',str(round(((f['angle_deg']-rotation)%360)*60000)))
        linear.set('scaled','0');gf.append(linear)
        tile=OxmlElement('a:tileRect');gf.append(tile)
    if style['stroke'] is None:
        shape.line.fill.background();return
    shape.line.color.rgb=RGBColor.from_string(style['stroke'][1:]);shape.line.width=max(1,round(style['width_px']*scale))
    line=shape._element.spPr.find('a:ln',NS)
    for node in list(line):
        if etree.QName(node).localname in ('prstDash','custDash'):line.remove(node)
    dash=pattern(style)
    if dash:
        node=OxmlElement('a:custDash')
        for a,b in zip(dash[::2],dash[1::2]):
            seg=OxmlElement('a:ds');seg.set('d',str(max(1,round(a/style['width_px']*100000))))
            seg.set('sp',str(max(1,round(b/style['width_px']*100000))));node.append(seg)
    else:
        node=OxmlElement('a:prstDash');node.set('val','solid')
    line.append(node);line.set('cap','rnd' if style['dash']=='dot' else 'flat')


def ordered_scene(scene, relations):
    out=copy.deepcopy(scene)
    order=relations['object_order']+[k for k in scene['objects'] if k not in relations['object_order']]
    out['objects']={k:out['objects'][k] for k in order}
    for k,ids in relations['part_orders'].items():
        obj=out['objects'][k];obj['part_keys']=ids+[p for p in obj['part_keys'] if p not in ids]
    return out


def svg(scene, styles, relations):
    w,h=scene['source_size'];defs=[];paths=[]
    for index,key in enumerate(relations['flat_order']):
        p=scene['parts'][key];s=styles[key];f=s['fill']
        fill_value=f or 'none'
        if isinstance(f,dict):
            a,b=gradient_line(p,f['angle_deg']);ident=f'gradient_{index}'
            defs.append(f'<linearGradient id="{ident}" gradientUnits="userSpaceOnUse" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"><stop offset="0" stop-color="{f["start_color"]}"/><stop offset="1" stop-color="{f["end_color"]}"/></linearGradient>')
            fill_value='url(#'+ident+')'
        d=' '.join(c[0]+' '+' '.join(format(v,'.12g') for v in c[1:]) for c in p['commands'])
        dash=pattern(s);extra='' if not dash else ' stroke-dasharray="'+','.join(map(str,dash))+'"'
        cap='round' if s['dash']=='dot' else 'butt'
        paths.append(f'<path id="{escape(p["object_id"]+"/"+p["id"], {chr(34):"&quot;"})}" d="{d}" fill="{fill_value}" stroke="{s["stroke"] or "none"}" stroke-width="{s["width_px"]}" stroke-linecap="{cap}"{extra}/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="-.5 -.5 {w} {h}"><defs>'+''.join(defs)+'</defs><rect x="-.5" y="-.5" width="100%" height="100%" fill="white"/>'+''.join(paths)+'</svg>'


def geometry_signature(path):
    with zipfile.ZipFile(path) as archive: root=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    rows={}
    def walk(node,parent='slide'):
        for child in node:
            name=etree.QName(child).localname
            if name=='grpSp':
                ident=child.find('p:nvGrpSpPr/p:cNvPr',NS).get('name');key=parent+'/'+ident
                rows[key]={'parent':parent,'group_xfrm':etree.tostring(child.find('p:grpSpPr/a:xfrm',NS)).decode()}
                walk(child,key)
            elif name=='sp':
                ident=child.find('p:nvSpPr/p:cNvPr',NS).get('name');sp=child.find('p:spPr',NS)
                geometry=sp.find('a:prstGeom',NS)
                if geometry is None:geometry=sp.find('a:custGeom',NS)
                rows[parent+'/'+ident]={'parent':parent,'xfrm':etree.tostring(sp.find('a:xfrm',NS)).decode(),
                                       'geometry':etree.tostring(geometry).decode()}
    walk(root.find('p:cSld/p:spTree',NS))
    return rows


@bounded('render')
def render_native(pptx, output, size, timeout=180):
    from ..office_render import convert_pdf
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=convert_pdf(pptx,output,timeout)
    pdf=result['pdf']
    if result['returncode'] or result['timed_out'] or not result['pdf_published']:
        raise RuntimeError('Native PPT render failed: '+result['log'][-1000:])
    with _PDF_LOCK, fitz.open(pdf) as doc:
        if len(doc)!=1:raise ValueError('Expected a single-slide PPT render')
        page=doc[0];pix=page.get_pixmap(matrix=fitz.Matrix(size[0]/page.rect.width,size[1]/page.rect.height),alpha=False)
        image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        if image.size!=tuple(size):image=image.resize(size,Image.Resampling.LANCZOS)
        image.save(output.parent/'native.png')
    shutil.copy2(pdf,output.parent/'native.pdf')
    return output.parent/'native.png'


def render_pdf_crop(pdf, source_size, box, size, scale, offset):
    """Rasterize the actual PPT PDF once, on the CLEAN crop's pixel grid.

    Map the full PDF page to original-image coordinates, then apply exactly the
    same crop, isotropic scale and fractional padding as the source crop. A
    crop-sized PDF canvas clips the vector page before rasterization, avoiding
    both whole-page high-resolution buffers and enlarged low-resolution PNGs.
    """
    pdf=Path(pdf)
    with _PDF_LOCK, fitz.open(pdf) as source, fitz.open() as local:
        if not source.is_pdf:raise ValueError('Expected a native PPT PDF, not a raster image')
        if len(source)!=1:raise ValueError('Expected a single-slide PPT PDF')
        page=local.new_page(width=size[0],height=size[1])
        x0=-box[0]*scale+offset[0];y0=-box[1]*scale+offset[1]
        destination=fitz.Rect(x0,y0,x0+source_size[0]*scale,y0+source_size[1]*scale)
        if source[0].get_contents():
            page.show_pdf_page(destination,source,0,keep_proportion=False)
        pix=page.get_pixmap(matrix=fitz.Matrix(1,1),colorspace=fitz.csRGB,alpha=False)
        image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        if image.size!=tuple(size):raise ValueError('PDF crop does not match the CLEAN pixel grid')
        metadata={'method':PDF_CROP_PROTOCOL,'path':str(pdf.resolve()),'sha256':sha(pdf),
                  'page_index':0,'page_size_pt':list(source[0].rect),'page_rect_display':list(destination)}
    return image,metadata


def export(scene, styles, relations, folder, source, options, render_slots, *, native=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity={'scene':digest(scene),'styles':digest(styles),'relations':digest(relations),'source':sha(source),'native':native,
              'pdf_crop_protocol':PDF_CROP_PROTOCOL}
    if (folder/'export.json').exists():
        previous=read(folder/'export.json')
        if previous.get('identity')!=identity:raise ValueError('Rendering input changed; use a new revision')
        for name,expected in previous['files'].items():
            if sha(folder/name)!=expected:raise ValueError('Saved style rendering changed: '+name)
        return previous
    keys=set(relations['flat_order'])
    if not keys<=set(styles):raise ValueError('Missing style for exportable geometry')
    content=svg(scene,styles,relations);(folder/'styled.svg').write_text(content)
    import cairosvg  # Text-only export does not need the SVG rasterizer.
    cairosvg.svg2png(bytestring=content.encode(),write_to=str(folder/'styled.png'))
    write_pptx(scene,folder/'geometry_reference.pptx')
    counts=write_pptx(ordered_scene(scene,relations),folder/'styled.pptx',
                      style_writer=lambda shape,p,scale:apply_style(shape,p,scale,styles[part_key(p)]))
    before=geometry_signature(folder/'geometry_reference.pptx');after=geometry_signature(folder/'styled.pptx')
    if before!=after:raise ValueError('Style export changed geometry or group membership')
    if native:
        with render_slots:render_native(folder/'styled.pptx',folder/'native_render',scene['source_size'],options['render_timeout_seconds'])
        compare([source,folder/'native.png'],['Source','Actual PPT render (shapes only)'],folder/'comparison.png',width_cap=1600)
    result={'identity':identity,'geometry_unchanged':True,'counts':counts,'gradient_parts':sorted(k for k in keys if isinstance(styles[k]['fill'],dict)),
            'native_rendered':native,'powerpoint_ui_editing_verified':False,
            'pptx_sha256':sha(folder/'styled.pptx'),'visual_accuracy_verified':False}
    result['files']={name:sha(folder/name) for name in ['styled.svg','styled.png','styled.pptx']+(['native.pdf','native.png','comparison.png'] if native else [])}
    save(folder/'export.json',result);return result
