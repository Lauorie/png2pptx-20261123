"""Editable one-line text frames; source pixels and all base objects stay intact."""
from pathlib import Path
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from ..common import save, sha, compare
from ..styling.export import render_native
from ..resource_limits import bounded


def disable_auto_bullets(paragraph):
    """Literal source markers belong to runs, never inherited PPT list formatting."""
    properties=paragraph._p.get_or_add_pPr()
    for child in list(properties):
        if etree.QName(child).localname in ('buNone','buAutoNum','buChar','buBlip'):
            properties.remove(child)
    marker=OxmlElement('a:buNone')
    # DrawingML puts the bullet choice before tabs/default run properties.
    following=next((child for child in properties if
                    etree.QName(child).localname in ('tabLst','defRPr','extLst')),None)
    if following is None:properties.append(marker)
    else:properties.insert(properties.index(following),marker)
    properties.set('marL','0');properties.set('indent','0')


def write(rows, size, path, base=None):
    prs=Presentation(str(base)) if base else Presentation()
    if base:
        if len(prs.slides)!=1 or abs(prs.slide_width/prs.slide_height-size[0]/size[1])>1e-4:
            raise ValueError('Text base must be a single slide with the source aspect ratio')
        slide=prs.slides[0]
    else:
        prs.slide_width=round(Inches(13.333333));prs.slide_height=round(prs.slide_width*size[1]/size[0])
        slide=prs.slides.add_slide(prs.slide_layouts[6])
    before=[etree.tostring(s._element) for s in slide.shapes]
    scale=prs.slide_width/size[0];records=[]
    for row in rows:
        x,y,w,h=row['frame_px']
        multiline_fallback=bool(row.get('multiline_render_fallback') and row.get('kind')!='formula')
        if multiline_fallback:
            x,y,x1,y1=row['bbox_2d'];w=x1-x+1;h=y1-y+1
        shape=slide.shapes.add_textbox(round(x*scale),round(y*scale),max(1,round(w*scale)),max(1,round(h*scale)))
        shape.name='TXT '+row['id'];shape.rotation=row['rotation_deg']%360
        from .background import paint
        paint(shape,row)
        tf=shape.text_frame;tf.clear();tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        insets=[0.,0.,0.,0.] if multiline_fallback else row.get('text_insets_px',[0.,0.,0.,0.])
        tf.margin_left,tf.margin_top,tf.margin_right,tf.margin_bottom=[round(v*scale) for v in insets]
        tf.word_wrap=multiline_fallback;tf.auto_size=MSO_AUTO_SIZE.NONE;tf.vertical_anchor=MSO_ANCHOR.TOP
        para=tf.paragraphs[0];para.alignment=PP_ALIGN.LEFT;para.space_before=para.space_after=Pt(0);para.line_spacing=1.
        disable_auto_bullets(para)
        # Explicit paragraph font prevents a 18pt default from altering tiny-script baselines.
        # Font sizes use EMU here; one point is 12700 EMU, not one EMU.
        font_size_px=row['font_size_px']
        drawing_runs=[row['runs']]
        stacked=bool(row.get('layout_mode')=='stacked_vertical' and row.get('kind')!='formula')
        if stacked:
            from .layout_integrity import stacked_runs
            drawing_runs=stacked_runs(row)
        elif multiline_fallback:
            # No invented line bboxes after a failed retry. Use a native wrapped
            # frame at the original bounds, explicitly marked approximate.
            from .layout import pillow_metrics, projected
            from .multiline_retry import fallback_lines
            from copy import deepcopy
            drawing_runs=fallback_lines(row);metrics=[]
            for line in drawing_runs:
                probe=deepcopy(row);probe['runs']=line
                metrics.append(pillow_metrics(probe))
            block_width=max(m['width'] for m in metrics)
            block_height=max(m['height'] for m in metrics)+24*1.2*(len(metrics)-1)
            projected_width,projected_height=projected(block_width,block_height,row['rotation_deg'])
            font_size_px=.95*24*min(w/max(1,projected_width),h/max(1,projected_height))
        font_size_emu=max(Pt(1),round(font_size_px*scale))
        para.font.name=row['font'];para.font.size=font_size_emu
        if stacked:para.line_spacing=Emu(round(font_size_emu*1.2))
        if row.get('kind')=='formula':
            from .equations import append_equation, with_fallback
            append_equation(shape,row,scale)
            with_fallback(slide,shape,row,scale)
        rendered_runs=[] if row.get('kind')=='formula' else [
            (index,r) for index,line in enumerate(drawing_runs) for r in line]
        previous_line=0
        for line_index,r in rendered_runs:
            if line_index!=previous_line:
                para.add_line_break();previous_line=line_index
            run=para.add_run();run.text=r['text'];run.font.name=row['font']
            # Office applies its own script-size reduction for nonzero baseline.
            # Reducing sz as well would shrink superscripts/subscripts twice.
            run.font.size=font_size_emu
            run.font.bold=r.get('bold',row['bold']);run.font.italic=r.get('italic',row['italic'])
            run.font.color.rgb=RGBColor.from_string(r.get('color',row['color'])[1:])
            run.font.underline=r.get('underline',row.get('underline',False))
            rp=run._r.get_or_add_rPr()
            if r['script']!='normal':rp.set('baseline',str({'sub':-25000,'super':30000}[r['script']]))
            for tag in ('a:ea','a:cs'):
                node=OxmlElement(tag);node.set('typeface',row['font']);rp.append(node)
        records.append({'id':row['id'],'font_size_source_px':font_size_px,
                        'background_fill':row.get('background_fill'),
                        'background_evidence':row.get('background_evidence',''),
                        'font_size_pt':font_size_emu/12700,'font':row['font'],
                        'local_render_font':row['resolved_font'],
                        'representation':'native_equation' if row.get('kind')=='formula' else 'native_text',
                        'multiline_render_fallback':multiline_fallback,
                        'layout_mode':row.get('layout_mode','single_line'),
                        'approximate_render_line_count':len(drawing_runs) if multiline_fallback else None,
                        'run_colors':[r.get('color',row['color']) for r in row['runs']]})
    if before!=[etree.tostring(s._element) for s in list(slide.shapes)[:len(before)]]:
        raise ValueError('Text append modified a base object')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.partial.pptx');prs.save(temporary);temporary.replace(path)
    return {'textboxes':len(rows),'native_equations':sum(r.get('kind')=='formula' for r in rows),
            'base_objects':len(before),'base_objects_unchanged':True,'unit_mapping':records}


def retained_images(base, size):
    """Read actual retained pictures, including axis-aligned nested group transforms."""
    if not base:return []
    prs=Presentation(str(base));sx=size[0]/prs.slide_width;sy=size[1]/prs.slide_height;rows=[]
    def walk(shapes, offset=(0.,0.), scale=(1.,1.)):
        for sh in shapes:
            if abs(sh.rotation%360)>1e-6:continue  # no incorrect axis-aligned exclusion for rotated assets
            if sh.shape_type==6:
                t=sh._element.grpSpPr.xfrm
                if t is None or not t.chExt.cx or not t.chExt.cy:continue
                newscale=(scale[0]*t.ext.cx/t.chExt.cx,scale[1]*t.ext.cy/t.chExt.cy)
                newoff=(offset[0]+scale[0]*t.off.x-newscale[0]*t.chOff.x,
                        offset[1]+scale[1]*t.off.y-newscale[1]*t.chOff.y)
                walk(sh.shapes,newoff,newscale)
            elif sh.shape_type==13:
                x=(offset[0]+scale[0]*sh.left)*sx;y=(offset[1]+scale[1]*sh.top)*sy
                w=scale[0]*sh.width*sx;h=scale[1]*sh.height*sy
                rows.append({'id':f'IMG{len(rows)+1:03d}','bbox_2d':[x,y,x+w,y+h],'name':sh.name})
    walk(prs.slides[0].shapes)
    return rows


@bounded('cpu')
def final(rows, source, folder, slots, opt, base=None):
    from .equations import render_preview
    from .background_selection import select
    rows[:]=select(rows,source,folder/'background_selection',slots,opt,base)
    with Image.open(source) as im:size=list(im.size)
    counts=write(rows,size,folder/'text.pptx')
    with slots:render_preview(folder/'text.pptx',folder/'native_render',size,opt['render_timeout_seconds'])
    compare([source,folder/'native.png'],['Source','Editable text'],folder/'comparison.png',width_cap=1400)
    if base:
        counts['combined']=write(rows,size,folder/'combined.pptx',base)
        with slots:render_preview(folder/'combined.pptx',folder/'combined/native_render',size,opt['render_timeout_seconds'])
        compare([source,folder/'combined/native.png'],['Source','Shapes + arrows + text'],folder/'combined_comparison.png',width_cap=1400)
    save(folder/'export.json',{'counts':counts,'source_sha256':sha(source),'base_sha256':sha(base) if base else None,
                              'text_pptx_sha256':sha(folder/'text.pptx'),'visual_accuracy_verified':False})
    return counts
