"""Pixel-center frames and endpoint-preserving interpolation for open paths."""
import copy
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import PchipInterpolator
from ..common import save,FONT,compare
from ..styling.views import crop_transform,resample_crop
from ..fitting.paths import sample_commands


def frame(source,box,folder,opt,*,padding=None,full=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:
        source_size=list(im.size)
        if full:crop=[0,0,*im.size]
        else:
            pad=max(opt['crop_min_padding_px'],max(box[2]-box[0],box[3]-box[1])*(opt['crop_padding'] if padding is None else padding))
            crop=[max(0,math.floor(box[0]-pad)),max(0,math.floor(box[1]-pad)),
                  min(im.width,math.ceil(box[2]+pad)+1),min(im.height,math.ceil(box[3]+pad)+1)]
        extent=[crop[2]-crop[0],crop[3]-crop[1]]
        edge=opt['crop_max_long_edge'] if full else min(opt['crop_max_long_edge'],max(opt['crop_min_long_edge'],max(extent)*opt['crop_scale']))
        size=[max(2,round(x*edge/max(extent))) for x in extent]
        scale,offset=crop_transform(crop,size)
        resample_crop(im.convert('RGB'),crop,size,scale,offset).save(folder/'clean.png')
    meta={'box':crop,'size':size,'scale':scale,'offset':offset,'source_size':source_size,
          'convention':'pixel centers: local=(source-origin+.5)*scale-.5+offset'}
    save(folder/'frame.json',meta);return meta


def point(xy,f,inverse=False):
    if inverse:return [(xy[i]+.5-f['offset'][i])/f['scale']+f['box'][i]-.5 for i in (0,1)]
    return [(xy[i]-f['box'][i]+.5)*f['scale']-.5+f['offset'][i] for i in (0,1)]


def box(b,f,inverse=False):
    return point(b[:2],f,inverse)+point(b[2:],f,inverse)


def mapped(g,f,inverse=False):
    result=copy.deepcopy(g);result['points']={k:point(v,f,inverse) for k,v in g['points'].items()};return result


def bounds(g):
    pts=list(g['points'].values());a=np.asarray(pts)
    return [float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())]


def commands(g,path):
    result=[]
    for seg in path['segments']:
        pts=np.asarray([g['points'][k] for k in seg['points']],float)
        if not result:result.append(['M',*pts[0].tolist()])
        if seg['kind']=='line' or len(pts)==2:
            result.extend(['L',*p.tolist()] for p in pts[1:]);continue
        t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        curve=PchipInterpolator(t,pts,axis=0);d=curve.derivative()(t)
        for i,dt in enumerate(np.diff(t)):
            c1=pts[i]+d[i]*dt/3;c2=pts[i+1]-d[i+1]*dt/3
            result.append(['C',*c1.tolist(),*c2.tolist(),*pts[i+1].tolist()])
    return result


def _display_image(clean,max_width=None):
    with Image.open(clean) as im:image=im.convert('RGBA')
    original=image.size
    if max_width is not None and image.width>max_width:
        image=image.resize((max_width,max(1,round(image.height*max_width/image.width))),Image.Resampling.LANCZOS)
    return image,(image.width/original[0],image.height/original[1])


def draw_geometry(clean,g,path,*,labels=True,terminals=False,max_width=None):
    """GLM/DeepSeek coordinate-trial markers, drawn AFTER any display resize.

    Adapted from ref/native_qwen/native_qwen/render.py: hollow points, shared
    junction rings, white ID badges, leaders and six-position label avoidance.
    max_width is for human previews only; model images keep their pixel frame.
    """
    image,scale=_display_image(clean,max_width)
    layer=Image.new('RGBA',image.size);d=ImageDraw.Draw(layer);edge=max(image.size)
    width=max(4,round(edge/200));radius=max(5,round(edge/160));ring_width=max(2,round(width/2))
    font=ImageFont.truetype(FONT,max(18,round(edge/40)));pad=max(3,round(font.size/6))
    colors=['#ca0079','#126ad4','#008768','#ae5a00','#7543be','#bd242a']
    def pos(q):return tuple((q[i]+.5)*scale[i]-.5 for i in (0,1))
    occupied=[];pending=[];point_colors={};point_owners={}
    def badge(label,q,color):
        tb=d.textbbox((0,0),label,font=font);tw=tb[2]-tb[0]+2*pad;th=tb[3]-tb[1]+2*pad
        gap=radius+pad;choices=[]
        for dx,dy in [(gap,-th-gap),(gap,gap),(-tw-gap,-th-gap),(-tw-gap,gap),(gap,-2*th-gap),(gap,th+gap)]:
            x=max(0,min(image.width-tw,q[0]+dx));y=max(0,min(image.height-th,q[1]+dy));rect=[x,y,x+tw,y+th]
            overlap=sum(max(0,min(rect[2],b[2])-max(rect[0],b[0]))*max(0,min(rect[3],b[3])-max(rect[1],b[1])) for b in occupied)
            choices.append((overlap,rect))
        _,rect=min(choices,key=lambda v:v[0]);occupied.append(rect)
        anchor=(max(rect[0],min(q[0],rect[2])),max(rect[1],min(q[1],rect[3])))
        d.line([q,anchor],fill=color+'90',width=max(1,ring_width//2))
        d.rounded_rectangle(rect,radius=pad,fill=(255,255,255,235),outline=color+'c0',width=1)
        d.text((rect[0]+pad-tb[0],rect[1]+pad-tb[1]),label,fill=color,font=font)
    for i,p in enumerate(g['paths']):
        pts=[pos(q) for q in sample_commands(commands(g,p),160)];color=colors[i%len(colors)]
        if len(pts)>1:d.line(pts,fill=color+'d0',width=width)
        for seg in p['segments']:
            for key in seg['points']:
                point_colors.setdefault(key,color);point_owners.setdefault(key,set()).add(p['id'])
        start=g['points'][p['segments'][0]['points'][0]];end=g['points'][p['segments'][-1]['points'][-1]]
        for name,headed,q in [('start',p['start_head'],start),('end',p['end_head'],end)]:
            if headed:
                x,y=pos(q);r=radius*1.5
                d.ellipse([x-r,y-r,x+r,y+r],outline=color,width=ring_width)
                if terminals:pending.append((p['id']+'.'+name,(x,y),color))
        pending.append((p['id'],pts[len(pts)//2],color))
    if labels:
        for key,q in g['points'].items():
            x,y=pos(q);color=point_colors.get(key,colors[0])
            d.ellipse([x-radius,y-radius,x+radius,y+radius],outline=color,width=ring_width)
            if len(point_owners.get(key,()))>1:
                r=radius*1.6;d.ellipse([x-r,y-r,x+r,y+r],outline=color,width=max(1,ring_width-1))
            badge(key,(x,y),color)
    for label,q,color in pending:badge(label,q,color)
    Image.alpha_composite(image,layer).convert('RGB').save(path)


def compare_geometry(clean,geometries,path,*,width_cap=580):
    """Readable preview panels: resize source first, then annotate each geometry."""
    path=Path(path);folder=path.parent/(path.stem+'_panels');folder.mkdir(parents=True,exist_ok=True)
    image,_=_display_image(clean,width_cap);image.convert('RGB').save(folder/'clean.png')
    paths=[folder/'clean.png'];names=['CLEAN']
    for i,(name,g) in enumerate(geometries):
        target=folder/f'geometry_{i:02d}.png'
        draw_geometry(clean,g,target,max_width=width_cap)
        paths.append(target);names.append(name)
    compare(paths,names,path,width_cap=width_cap)


def draw_boxes(clean,objects,f,path):
    with Image.open(clean) as im:out=im.convert('RGB')
    d=ImageDraw.Draw(out);font=ImageFont.truetype(FONT,max(14,max(out.size)//90))
    for o in objects:
        b=box(o['bbox_2d'],f)
        if b[2]<0 or b[3]<0 or b[0]>=out.width or b[1]>=out.height:continue
        d.rectangle(b,outline='#D80086',width=3)
        d.text((max(0,b[0]),max(0,b[1]-font.size)),o['id'],fill='#A90067',font=font,stroke_width=2,stroke_fill='white')
    out.save(path)


def local_styles(styles,f):
    out=copy.deepcopy(styles)
    for s in out.values():
        s['width_px']*=f['scale']
        if 'dash_pattern_px' in s:s['dash_pattern_px']=[v*f['scale'] for v in s['dash_pattern_px']]
    return out


def source_styles(rows,f,minimum):
    out={}
    for row in rows:
        s={k:copy.deepcopy(v) for k,v in row.items() if k!='id'}
        s['width_px']=max(minimum,s['width_px']/f['scale'])
        if 'dash_pattern_px' in s:s['dash_pattern_px']=[v/f['scale'] for v in s['dash_pattern_px']]
        out[row['id']]=s
    return out
