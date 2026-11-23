"""Crop original pixels from selected fitted geometry, with separate ID hints."""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely.ops import unary_union
from ..common import FONT, save, sha
from ..fitting.sizing import crop_size
from ..fitting.paths import sample_commands
from .geometry import crop_box, subpaths, bounds, footprint
from .export import render_pdf_crop, PDF_CROP_PROTOCOL


def crop_transform(box, size):
    """One isotropic scale, with subpixel padding for integer output dimensions."""
    extent=[box[2]-box[0],box[3]-box[1]]
    scale=min(size[i]/extent[i] for i in (0,1))
    offset=[(size[i]-extent[i]*scale)/2 for i in (0,1)]
    return scale,offset


def resample_crop(image, box, size, scale, offset):
    # Fractional resize bounds preserve one exact scale in both axes. A direct
    # resize to rounded width/height would introduce two different length units.
    pad=math.ceil(max(offset)/scale)+3
    crop=ImageOps.expand(image.crop(box),border=pad,fill='white')
    region=(pad-offset[0]/scale,pad-offset[1]/scale,
            pad+(size[0]-offset[0])/scale,pad+(size[1]-offset[1])/scale)
    return crop.resize(size,Image.Resampling.LANCZOS,box=region)


def label_position(anchors, size, text_size, occupied):
    """Place short labels by their own contour without hiding another label."""
    w,h=text_size
    for anchor in anchors:
        for dx,dy in ((7,7),(7,-h-7),(-w-7,7),(-w-7,-h-7)):
            x=max(0,min(size[0]-w,anchor[0]+dx));y=max(0,min(size[1]-h,anchor[1]+dy))
            rect=[x,y,x+w,y+h]
            if all(min(rect[2]+3,b[2])<=max(rect[0]-3,b[0]) or
                   min(rect[3]+3,b[3])<=max(rect[1]-3,b[1]) for b in occupied):
                return anchor,rect
    # Dense compounds stay intact: use the nearest free label slot, not batches.
    for y in range(0, max(1, size[1]-h+1), h+4):
        for x in range(0, max(1, size[0]-w+1), w+4):
            rect=[x,y,x+w,y+h]
            if all(min(rect[2]+3,b[2])<=max(rect[0]-3,b[0]) or
                   min(rect[3]+3,b[3])<=max(rect[1]-3,b[1]) for b in occupied):
                anchor=min(anchors,key=lambda a:(a[0]-x-w/2)**2+(a[1]-y-h/2)**2)
                return anchor,rect
    raise ValueError('Whole-object target view has no free label space')


def stroke_reference(images, size, minimum, scale):
    """Same footer on every image, with exact fractional-pixel line coverage."""
    width=minimum*scale
    footer_height=max(70,math.ceil(width)+48)
    footer=Image.new('RGB',(size[0],footer_height),'white')
    draw=ImageDraw.Draw(footer)
    label=f'MIN: {width:.3g} crop px ({minimum:g} source px)'
    if size[0]<350:label=f'MIN {width:.3g}px'
    font_size=16
    while font_size>6 and draw.textbbox((0,0),label,font=ImageFont.truetype(FONT,font_size))[2]>size[0]-8:
        font_size-=1
    draw.text((4,5),label,font=ImageFont.truetype(FONT,font_size),fill='black')
    # Area coverage keeps 0.8*scale exact, without rounding to an integer pen size.
    center=(28+footer_height)/2
    coverage=np.clip(np.minimum(np.arange(footer_height)+1,center+width/2)-
                     np.maximum(np.arange(footer_height),center-width/2),0,1)
    values=np.asarray(footer).copy()
    x0=max(0,min(size[0]-1,int(size[0]*.15)));x1=max(x0+1,min(size[0],int(size[0]*.85)))
    rows=coverage>0
    values[rows,x0:x1]=np.rint(255*(1-coverage[rows]))[:,None,None]
    footer=Image.fromarray(values)
    for _,path in images:
        with Image.open(path) as original:
            canvas=Image.new('RGB',(size[0],size[1]+footer_height),'white')
            canvas.paste(original,(0,0));canvas.paste(footer,(0,size[1]));canvas.save(path)
    return {'minimum_original_px':minimum,'minimum_crop_px':width,
            'footer_box_display':[0,size[1],size[0],size[1]+footer_height],
            'line_box_display':[x0,size[1]+center-width/2,x1,size[1]+center+width/2],
            'input_size_px':[size[0],size[1]+footer_height]}


def component_contours(table, keys):
    """Whole-component silhouettes, without internal cell edges or part labels."""
    geometry=unary_union([footprint(table[k])[0] for k in keys])
    def contours(g):
        if g.is_empty:return
        if g.geom_type=='Polygon':
            yield np.asarray(g.exterior.coords)
            for ring in g.interiors:yield np.asarray(ring.coords)
        elif hasattr(g,'geoms'):
            for child in g.geoms:yield from contours(child)
        else:yield np.asarray(g.coords)
    return list(contours(geometry))


def prepare(source, table, mapping, folder, options, *, native_pdf=None, assets=(), with_stroke_reference=False,
            component_members=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rendered_source=None
    groups={a:list(component_members[k]) if component_members is not None else [k] for a,k in mapping.items()}
    with Image.open(source) as src:
        image=src.convert('RGB');box=crop_box([table[k] for keys in groups.values() for k in keys],image.size,
                                  options['crop_padding'],options['crop_min_padding_px'])
        extent=[box[2]-box[0],box[3]-box[1]]
        global_scale=2048/max(image.size)
        size,sizing=crop_size([v*global_scale for v in extent],scale=options['crop_scale'],
                              min_long_edge=options['crop_min_long_edge'],max_long_edge=options['crop_max_long_edge'])
        scale,offset=crop_transform(box,size)
        clean=resample_crop(image,box,size,scale,offset)
        clean.save(folder/'clean.png');hint=Image.blend(clean,Image.new('RGB',size,'white'),.65)
        draw=ImageDraw.Draw(hint);font=ImageFont.truetype(FONT,18)
        def local(x,y):return ((x-box[0]+.5)*scale-.5+offset[0],(y-box[1]+.5)*scale-.5+offset[1])
        markers=[]
        for i,(alias,key) in enumerate(mapping.items()):
            color=['#E6007E','#0071BC','#00A651','#D35400'][i%4]
            anchors=[]
            contours=(component_contours(table,groups[alias]) if component_members is not None else
                      [sample_commands(path,96) for path in subpaths(table[key])])
            for pts in contours:
                if len(pts)>1:draw.line([local(*q) for q in pts],fill=color,width=3)
                anchors.extend(local(*pts[j]) for j in np.linspace(0,len(pts)-1,min(24,len(pts)),dtype=int))
            markers.append((alias,key,color,anchors))
        for i,asset in enumerate(assets):
            b=asset.get('bbox_2d')
            if not b or min(b[2],box[2])<=max(b[0],box[0]) or min(b[3],box[3])<=max(b[1],box[1]):continue
            a=local(b[0],b[1]);c=local(b[2],b[3]);draw.rectangle([a,c],outline='#666666',width=2)
            draw.text((max(0,a[0]),max(0,a[1])),f'IMG{i+1}',font=font,fill='#555555',stroke_width=2,stroke_fill='white')
        occupied=[];labels={}
        for alias,key,color,anchors in markers:
            tb=draw.textbbox((0,0),alias,font=font);width=tb[2]-tb[0]+6;height=tb[3]-tb[1]+6
            anchor,rect=label_position(anchors,size,(width,height),occupied);occupied.append(rect)
            endpoint=(max(rect[0],min(rect[2],anchor[0])),max(rect[1],min(rect[3],anchor[1])))
            draw.line([anchor,endpoint],fill=color,width=1)
            draw.rectangle(rect,fill='white')
            draw.text((rect[0]+3-tb[0],rect[1]+3-tb[1]),alias,font=font,fill=color)
            labels[alias]={'key':key,'anchor_display':anchor,'label_box_display':rect}
        hint.save(folder/'target.png')
        images=[('CLEAN original source: read appearance here',folder/'clean.png')]
        if native_pdf is not None:
            rendered,rendered_source=render_pdf_crop(native_pdf,image.size,box,size,scale,offset)
            rendered.save(folder/'rendered.png')
            images.append(('ACTUAL PPT render, same crop and scale; only existing shapes are rendered',folder/'rendered.png'))
        target_kind='whole-component' if component_members is not None else 'part'
        images.append((f'Target {target_kind} IDs with short leader lines; artificial contours, labels and IMG boxes are not source styles',folder/'target.png'))
    reference=None
    minimum=options['minimum_stroke_width_px']
    if with_stroke_reference and minimum>0:
        reference=stroke_reference(images,size,minimum,scale)
        images=[(label+'; separate bottom MIN strip is a same-scale minimum stroke reference, not source content',path)
                for label,path in images]
    metadata={'geometry_keys':mapping,'crop_original_xyxy':box,'original_size':list(image.size),
              'labels':labels,
              'display_size':reference['input_size_px'] if reference else list(size),
              'crop_display_size':list(size),'stroke_reference':reference,
              'scale_xy':[scale,scale],'sizing':sizing,
              'source_to_crop_scale':scale,'display_offset_xy':offset,
              'transform_version':'isotropic_crop.1',
              'pixel_mapping':'local=(original-crop_origin+0.5)*scale-0.5+display_offset',
              'rendered_source':rendered_source,
              'images':[{'label':label,'path':str(path.resolve()),'sha256':sha(path)} for label,path in images]}
    if component_members is not None:
        metadata.update(target_scope='components',component_members={a:groups[a] for a in mapping})
    save(folder/'frame.json',metadata)
    units={'length_unit':'pixels of the main CLEAN crop','main_crop_size_px':list(size)}
    if reference:units['minimum_stroke_width_px']=reference['minimum_crop_px']
    return images,units,metadata
