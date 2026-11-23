"""Editable one-line text frames; source pixels and all base objects stay intact."""
from pathlib import Path
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from ..common import save, sha, compare
from ..styling.export import render_native


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
        shape=slide.shapes.add_textbox(round(x*scale),round(y*scale),max(1,round(w*scale)),max(1,round(h*scale)))
        shape.name='TXT '+row['id'];shape.rotation=row['rotation_deg']%360
        tf=shape.text_frame;tf.clear();tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        insets=row.get('text_insets_px',[0.,0.,0.,0.])
        tf.margin_left,tf.margin_top,tf.margin_right,tf.margin_bottom=[round(v*scale) for v in insets]
        tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE;tf.vertical_anchor=MSO_ANCHOR.TOP
        para=tf.paragraphs[0];para.alignment=PP_ALIGN.LEFT;para.space_before=para.space_after=Pt(0);para.line_spacing=1.
        # Explicit paragraph font prevents a 18pt default from altering tiny-script baselines.
        para.font.name=row['font'];para.font.size=max(1,round(row['font_size_px']*scale))
        for r in row['runs']:
            run=para.add_run();run.text=r['text'];run.font.name=row['font']
            # Office applies its own script-size reduction for nonzero baseline.
            # Reducing sz as well would shrink superscripts/subscripts twice.
            run.font.size=max(1,round(row['font_size_px']*scale))
            run.font.bold=row['bold'];run.font.italic=row['italic'];run.font.color.rgb=RGBColor.from_string(row['color'][1:])
            rp=run._r.get_or_add_rPr()
            if r['script']!='normal':rp.set('baseline',str({'sub':-25000,'super':30000}[r['script']]))
            for tag in ('a:ea','a:cs'):
                node=OxmlElement(tag);node.set('typeface',row['font']);rp.append(node)
        records.append({'id':row['id'],'font_size_source_px':row['font_size_px'],
                        'font_size_pt':row['font_size_px']*scale/12700,'font':row['font'],
                        'local_render_font':row['resolved_font']})
    if before!=[etree.tostring(s._element) for s in list(slide.shapes)[:len(before)]]:
        raise ValueError('Text append modified a base object')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.partial.pptx');prs.save(temporary);temporary.replace(path)
    return {'textboxes':len(rows),'base_objects':len(before),'base_objects_unchanged':True,'unit_mapping':records}


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


def final(rows, source, folder, slots, opt, base=None):
    with Image.open(source) as im:size=list(im.size)
    counts=write(rows,size,folder/'text.pptx')
    with slots:render_native(folder/'text.pptx',folder/'native_render',size,opt['render_timeout_seconds'])
    compare([source,folder/'native.png'],['Source','Editable text'],folder/'comparison.png',width_cap=1400)
    if base:
        counts['combined']=write(rows,size,folder/'combined.pptx',base)
        with slots:render_native(folder/'combined.pptx',folder/'combined/native_render',size,opt['render_timeout_seconds'])
        compare([source,folder/'combined/native.png'],['Source','Shapes + arrows + text'],folder/'combined_comparison.png',width_cap=1400)
    save(folder/'export.json',{'counts':counts,'source_sha256':sha(source),'base_sha256':sha(base) if base else None,
                              'text_pptx_sha256':sha(folder/'text.pptx'),'visual_accuracy_verified':False})
    return counts
