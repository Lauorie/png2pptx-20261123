"""Small, replaceable previews; all drawings are proposals, not source pixels."""
from pathlib import Path
import colorsys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import graph as gr

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def color(i):return tuple(round(255*x) for x in colorsys.hsv_to_rgb(i*.61803398875%1,.8,.85))
def font(image):return ImageFont.truetype(FONT,max(16,round(max(image.size)/70)))


def label(draw,xy,text,tint,image):
    f=font(image);bb=draw.textbbox((0,0),text,font=f);w=bb[2]-bb[0]
    x=max(1,min(image.width-w-3,xy[0]+5));y=max(1,min(image.height-f.size-3,xy[1]+5))
    draw.text((x,y),text,font=f,fill=tint,stroke_width=2,stroke_fill='white')


def masks(clean,layers,path,white=False):
    base=np.full((*np.asarray(clean).shape[:2],3),255,dtype=float) if white else np.asarray(clean).astype(float)
    counts=np.zeros(base.shape[:2]);tints=np.zeros_like(base)
    for i,(_,m) in enumerate(layers):counts[m]+=1;tints[m]+=color(i)
    covered=counts>0;alpha=1. if white else .43
    base[covered]=(1-alpha)*base[covered]+alpha*tints[covered]/counts[covered,None]
    image=Image.fromarray(np.rint(base).astype('uint8'));d=ImageDraw.Draw(image)
    for i,(name,m) in enumerate(layers):
        y,x=np.nonzero(m)
        if len(x):
            j=round((i+1)/(len(layers)+1)*(len(x)-1));label(d,(x[j],y[j]),name,color(i),image)
    image.save(path)


def head(draw,p,inner,kind,tint,width):
    if kind=='none':return
    if kind=='unknown':
        draw.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4],outline=tint,width=2);return
    direction=np.asarray(p)-inner;norm=np.linalg.norm(direction)
    if norm==0:return
    direction/=norm;normal=np.array([-direction[1],direction[0]]);length=max(9,width*4)
    back=p-direction*length;left=back+normal*length*.4;right=back-normal*length*.4
    if kind=='arrow':draw.line([tuple(left),tuple(p),tuple(right)],fill=tint,width=max(2,width))
    elif kind=='oval':draw.ellipse([p[0]-length*.4,p[1]-length*.4,p[0]+length*.4,p[1]+length*.4],fill=tint)
    elif kind=='diamond':draw.polygon([tuple(p),tuple(left),tuple(p-direction*length*1.6),tuple(right)],fill=tint)
    elif kind=='stealth':draw.polygon([tuple(p),tuple(left),tuple(p-direction*length*.65),tuple(right)],fill=tint)
    else:draw.polygon([tuple(p),tuple(left),tuple(right)],fill=tint)


def targets(clean,items,path,white=False,boxes=False):
    image=Image.new('RGB',clean.size,'white') if white else clean.copy();d=ImageDraw.Draw(image)
    width=max(2,round(max(clean.size)/700))
    for i,t in enumerate(items):
        g=t['graph'];c=color(i);name=t.get('label',t['id'])
        if boxes:
            x0,y0,x1,y1=gr.box(g);d.rectangle([x0-2,y0-2,x1+2,y1+2],outline=c,width=width)
            label(d,(x0,y0),name,c,image);continue
        drawn=set();drawn_heads=set()
        for r in g['routes']:
            p=gr.points(g,r)
            if r['kind']=='block_arrow':d.line([*map(tuple,p),tuple(p[0])],fill=c,width=width)
            else:
                for j,(a,b) in enumerate(zip(r['nodes'],r['nodes'][1:])):
                    key=tuple(sorted((a,b)))
                    if key not in drawn:d.line([tuple(p[j]),tuple(p[j+1])],fill=c,width=width);drawn.add(key)
                for j,k,h in [(0,1,r['heads'][0]),(-1,-2,r['heads'][1])]:
                    key=(r['nodes'][j],r['nodes'][k],h)
                    if key not in drawn_heads:head(d,p[j],p[k],h,c,width);drawn_heads.add(key)
            for x,y in p:d.ellipse([x-2,y-2,x+2,y+2],fill=c)
        r=g['routes'][0];p=gr.points(g,r);label(d,p[len(p)//2],name,c,image)
    image.save(path)


def component_guides(clean,layers,path):
    """Unconnected per-component centerline preview; no diameter, MST, or fitting."""
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_dilation
    guides=[(name,binary_dilation(skeletonize(m),iterations=1)) for name,m in layers]
    masks(clean,guides,path)
