"""Small native PPT writer/renderer and a measured med/med head legend."""
import json,subprocess,tempfile,threading
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement
from style import pending

KINDS=['triangle','stealth','arrow','diamond','oval']
from render_sync import PDF_LOCK


def write(size,rows,styles,path):
    # Local target crops may be tall. Preserve their aspect ratio while keeping
    # both physical slide dimensions inside PowerPoint's 1..56 inch limits.
    width_inches=min(13.333333,55*size[0]/size[1])
    if width_inches<1 or width_inches*size[1]/size[0]<1:
        raise ValueError('Crop aspect ratio exceeds native slide limits; use the full figure canvas')
    prs=Presentation();prs.slide_width=Inches(width_inches);prs.slide_height=round(prs.slide_width*size[1]/size[0])
    slide=prs.slides.add_slide(prs.slide_layouts[6]);scale=prs.slide_width/size[0]
    for row,style in zip(rows,styles):
        from style_contract import exportable
        style=exportable(style)
        points=[row['start'],*row.get('via',[]),row['end']]
        if row.get('kind')=='block_arrow':
            from shape_render import native_parts
            native_parts(slide,{**row,'style':style},scale)
            continue
        if len(points)==2:
            xy=[round((v+.5)*scale) for p in points for v in p]
            shape=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,*xy)
        else:
            xy=[tuple(round((v+.5)*scale) for v in p) for p in points]
            builder=slide.shapes.build_freeform(*xy[0],scale=1.0)
            builder.add_line_segments(xy[1:],close=False)
            shape=builder.convert_to_shape();shape.fill.background()
            if row.get('beziers'):
                # Preserve a genuinely smooth editable cubic path in OOXML.
                geom_path=shape._element.spPr.custGeom.pathLst[0]
                for node in list(geom_path):geom_path.remove(node)
                def point(node,p):
                    el=OxmlElement('a:pt');el.set('x',str(round((p[0]+.5)*scale)-shape.left));el.set('y',str(round((p[1]+.5)*scale)-shape.top));node.append(el)
                move=OxmlElement('a:moveTo');point(move,row['beziers'][0][0]);geom_path.append(move)
                for curve in row['beziers']:
                    node=OxmlElement('a:cubicBezTo')
                    for p in curve[1:]:point(node,p)
                    geom_path.append(node)
        shape.name=row['id'];shape.line.width=round((style['width_px'] or row.get('width_estimate') or 2)*scale)
        shape.line.color.rgb=RGBColor.from_string(style['color'][1:])
        ln=shape._element.spPr.get_or_add_ln()
        for node in list(ln):
            if node.tag.rsplit('}',1)[-1] in ('prstDash','custDash','headEnd','tailEnd','round','bevel','miter'):ln.remove(node)
        pattern=style['dash'];width=style['width_px'] or row.get('width_estimate') or 2
        sequence=pattern.get('pattern_px') or ([pattern['mark_px'],pattern['gap_px']] if pattern.get('mark_px') and pattern.get('gap_px') else [])
        if sequence:
            # DrawingML has no dash-offset attribute. Keep one measured cycle;
            # zero-gap phase emulation is rendered inconsistently by LO.
            dash=OxmlElement('a:custDash')
            for mark,gap in zip(sequence[::2],sequence[1::2]):
                segment=OxmlElement('a:ds')
                segment.set('d',str(max(1,round(mark/width*100000))))
                segment.set('sp',str(max(1,round(gap/width*100000))))
                dash.append(segment)
        else:
            dash=OxmlElement('a:prstDash');dash.set('val',pattern.get('preset','solid'))
        ln.append(dash)
        join=OxmlElement('a:miter');join.set('lim','400000');ln.append(join)
        for end,tag in [('start','headEnd'),('end','tailEnd')]:
            el=OxmlElement('a:'+tag);head=style.get(end)
            el.set('type',head['type'] if head else 'none')
            if head:el.set('w',head.get('width','med'));el.set('len',head.get('length','med'))
            ln.append(el)
        for node in list(shape._element):
            if node.tag.rsplit('}',1)[-1]=='style':shape._element.remove(node)
        shape._element.spPr.append(OxmlElement('a:effectLst'))
    prs.save(path)
    Path(path).with_suffix('.export.json').write_text(json.dumps(dict(
        geometry_paths=len(rows),style_pending=[r['id'] for r,s in zip(rows,styles) if pending(dict(style=s))],
        pending_policy='Visible gray provisional geometry; pending styles are not silently dropped',
        native_shapes=len(slide.shapes)),indent=2))


def render(pptx,folder,size):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arrow-style-lo-') as profile:
        command=['libreoffice','-env:UserInstallation='+Path(profile).as_uri(),'--headless','--convert-to','pdf','--outdir',str(folder.resolve()),str(Path(pptx).resolve())]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
    (folder/'renderer.log').write_text(proc.stdout+proc.stderr)
    pdf=folder/(Path(pptx).stem+'.pdf')
    if proc.returncode or not pdf.exists():raise RuntimeError('Native PPT render failed; see '+str(folder/'renderer.log'))
    with PDF_LOCK, fitz.open(pdf) as document:
        page=document[0];pix=page.get_pixmap(matrix=fitz.Matrix(size[0]/page.rect.width,size[1]/page.rect.height),alpha=True)
        # pix.pil_image() contains premultiplied channels; PNG decoding restores
        # straight RGBA for correct overlap compositing.
        import io
        rgba=Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGBA')
        if rgba.size!=tuple(size):rgba=rgba.resize(size,Image.Resampling.LANCZOS)
        rgba.save(folder/'styled_rgba.png')
        image=Image.new('RGBA',tuple(size),'white');image.alpha_composite(rgba);image=image.convert('RGB')
        image.save(folder/'styled.png')
        (folder/'styled.svg').write_text(page.get_svg_image())
    return image


def legend(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    size=[840,1200];rows=[];styles=[]
    for i,kind in enumerate(KINDS):
        y=120+i*240
        rows.append(dict(id=kind,start=[90,y],via=[],end=[630,y]))
        styles.append(dict(color='#000000',width_px=12,dash=dict(preset='solid'),start=None,end=dict(type=kind)))
    write(size,rows,styles,folder/'legend.pptx');image=render(folder/'legend.pptx',folder,size)
    manifest=dict(reference_width_px=12,types=KINDS,tip=[630,120],row_spacing=240,width='med',length='med',image_size=size)
    (folder/'legend.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return np.asarray(image),manifest


def case_templates(folder,image_width,rows,styles):
    """Native mid sizes depend on physical stroke width, not a fixed px ratio."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    refs=[];ref_styles=[];positions={};y=0;columns=max(1,min(5,image_width//200))
    cell=image_width/columns
    for row,style in zip(rows,styles):
        if not (row.get('start_head') or row.get('end_head')):continue
        width=style['width_px'] or row.get('width_estimate') or 2
        pitch=max(80,10*width);positions[row['id']]={}
        for i,kind in enumerate(KINDS):
            x=(i%columns+.65)*cell;yy=y+(i//columns+.5)*pitch
            refs.append(dict(id=row['id']+'_'+kind,start=[x-cell*.45,yy],via=[],end=[x,yy]))
            ref_styles.append(dict(color='#000000',width_px=width,dash=dict(preset='solid'),start=None,end=dict(type=kind)))
            positions[row['id']][kind]=[x,yy]
        y+=((len(KINDS)+columns-1)//columns)*pitch
    if not refs:return None
    # PowerPoint slide dimensions must be at least one inch.
    size=[image_width,max(int(np.ceil(y)),int(np.ceil(image_width/13.333333)))]
    write(size,refs,ref_styles,folder/'templates.pptx');im=render(folder/'templates.pptx',folder,size)
    manifest=dict(types=KINDS,positions=positions,image_size=size,head_size='med/med')
    (folder/'templates.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return np.asarray(im).mean(axis=-1)/255,manifest
