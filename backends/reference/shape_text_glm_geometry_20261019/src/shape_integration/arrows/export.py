"""Editable open paths with native endpoint properties and actual PPT rendering."""
import copy
import json
import zipfile
from pathlib import Path
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR,MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement
from ..common import save,sha,compare
from ..fitting.scene import custom_geometry
from ..styling.export import apply_style,render_native,render_pdf_crop,NS
from .geometry import commands


def write(objects,size,path,base=None):
    prs=Presentation(str(base)) if base else Presentation()
    if base:
        if len(prs.slides)!=1:raise ValueError('Combined arrow export requires one-slide base PPT')
        if abs(prs.slide_width/prs.slide_height-size[0]/size[1])>1e-4:raise ValueError('Base PPT aspect ratio mismatch')
        slide=prs.slides[0]
    else:
        prs.slide_width=round(Inches(13.333333));prs.slide_height=round(prs.slide_width*size[1]/size[0])
        slide=prs.slides.add_slide(prs.slide_layouts[6])
    scale=prs.slide_width/size[0];counts={'connectors':0,'freeforms':0,'heads':0,'networks':len(objects)}
    for obj in objects:
        geom=obj['geometry'];target=slide.shapes
        if len(geom['paths'])>1:
            group=slide.shapes.add_group_shape();group.name='arrow/'+obj['id'];target=group.shapes
        for p in geom['paths']:
            cmd=commands(geom,p);key=p['id'];s=obj['styles'][key]
            if len(cmd)==2 and cmd[1][0]=='L':
                coords=[round((v+.5)*scale) for c in cmd for v in c[1:]]
                shape=target.add_connector(MSO_CONNECTOR.STRAIGHT,*coords);counts['connectors']+=1
            else:
                pts=[(c[i],c[i+1]) for c in cmd for i in range(1,len(c),2)];xs,ys=zip(*pts)
                b=[min(xs),min(ys),max(xs),max(ys)];b[2]=max(b[2],b[0]+1e-3);b[3]=max(b[3],b[1]+1e-3)
                shape=target.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,round((b[0]+.5)*scale),round((b[1]+.5)*scale),max(1,round((b[2]-b[0])*scale)),max(1,round((b[3]-b[1])*scale)))
                custom_geometry(shape,cmd,b);counts['freeforms']+=1
            # The style exporter uses original-image pixels for lengths.
            apply_style(shape,{'commands':cmd},scale,{'fill':None,**s})
            ln=shape._element.spPr.find('a:ln',NS)
            for old in list(ln):
                if etree.QName(old).localname in ('headEnd','tailEnd'):ln.remove(old)
            for end,tag in [('start','headEnd'),('end','tailEnd')]:
                h=s.get(end);el=OxmlElement('a:'+tag);el.set('type',h['type'] if h else 'none')
                if h:el.set('w',h['width']);el.set('len',h['length']);counts['heads']+=1
                ln.append(el)
            shape.name='arrow/'+obj['id']+'/'+key
            for old in list(shape._element):
                if etree.QName(old).localname=='style':shape._element.remove(old)
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            for el in shape._element.xpath('.//p:cNvPr'):
                el.set('descr',json.dumps({'arrow_object':obj['id'],'path':key,'geometry_state':obj.get('geometry_state'),'style_state':obj.get('style_state')},ensure_ascii=False))
    prs.save(path);return counts


def render_object(obj,source,frame,folder,slots):
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'scene.json',obj);counts=write([obj],frame['source_size'],folder/'arrows.pptx')
    with slots:render_native(folder/'arrows.pptx',folder/'native_render',frame['source_size'])
    im,meta=render_pdf_crop(folder/'native.pdf',frame['source_size'],frame['box'],frame['size'],frame['scale'],frame['offset'])
    im.save(folder/'rendered.png');save(folder/'export.json',{'counts':counts,'rendered_source':meta,'pptx_sha256':sha(folder/'arrows.pptx')})
    return folder/'rendered.png'


def final(objects,source,folder,slots,base=None):
    with Image.open(source) as im:size=im.size
    counts=write(objects,size,folder/'arrows.pptx')
    with slots:render_native(folder/'arrows.pptx',folder/'native_render',size)
    compare([source,folder/'native.png'],['Source','Native arrows'],folder/'comparison.png',width_cap=1200)
    with Image.open(folder/'native.png') as im:im.thumbnail((1100,1100));im.save(folder/'preview.png')
    if base:
        write(objects,size,folder/'combined.pptx',base)
        output=folder/'combined';output.mkdir(exist_ok=True)
        with slots:render_native(folder/'combined.pptx',output/'native_render',size)
        compare([source,output/'native.png'],['Source','Shapes + arrows'],folder/'combined_comparison.png',width_cap=1200)
    save(folder/'export.json',{'counts':counts,'base_pptx':str(base) if base else None,'files':{p.name:sha(p) for p in [folder/'arrows.pptx',folder/'native.png',folder/'native.pdf']},'visual_accuracy_verified':False})
    return counts
