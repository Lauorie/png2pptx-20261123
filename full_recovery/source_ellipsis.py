"""Source-measured standalone punctuation, independent of font fallback metrics."""
from pathlib import Path
from itertools import combinations
import cv2
import numpy as np
from PIL import Image


def native_dot_identity(name):
    """Read the exporter-owned occurrence and component, including merge suffixes."""
    name = name.split(' [', 1)[0]
    if not name.startswith('SOURCE_ELLIPSIS '):
        return None
    occurrence, separator, component = name[len('SOURCE_ELLIPSIS '):].rpartition('/')
    if not occurrence or not separator or component not in ('0', '1', '2'):
        raise ValueError('Invalid native source punctuation identity')
    return occurrence, int(component)


def compact_component(w,h,area):
    """A small round dot can differ by one pixel across its sampled axes."""
    return area>=2 and max(w,h)<=1.8*min(w,h)+1 and area>=.65*w*h


def unique_triplet(dots):
    """An imperfect text box can include other compact context components.

    Require one complete, contiguous three-dot pattern. Four equally spaced
    dots or two neighboring patterns remain ambiguous; never skip an interior
    component to fabricate a longer three-dot occurrence.
    """
    if not 3 <= len(dots) <= 12:
        return None
    points=np.array([d[0] for d in dots]);groups=[]
    for indices in combinations(range(len(dots)),3):
        subset=[dots[i] for i in indices]
        xy=np.array([d[0] for d in subset]);diam=np.array([max(d[1:3]) for d in subset])
        if diam.max()>1.8*diam.min():continue
        _,_,axes=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
        along=np.sort(xy@axes[0]);spacing=np.diff(along)
        if min(spacing)<1.2*diam.max() or max(spacing)>1.8*min(spacing):continue
        if np.ptp(xy@axes[1])>.4*diam.max():continue
        other=[i for i in range(len(dots)) if i not in indices]
        if any(along[0] <= points[i]@axes[0] <= along[-1] and
               abs((points[i]-xy.mean(0))@axes[1])<=.6*diam.max() for i in other):continue
        groups.append(subset)
    return groups[0] if len(groups)==1 else None


def visible_dot(rgb,center,core_diameter,pigment,spacing):
    """Measure antialiased ink area; the dark core understates small dots."""
    radius=min(.45*spacing,max(3.,core_diameter+1.))
    lo=np.maximum(0,np.floor(center-radius)).astype(int)
    hi=np.minimum(rgb.shape[1::-1],np.ceil(center+radius+1)).astype(int)
    tile=rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    border=np.concatenate([tile[0],tile[-1],tile[:,0],tile[:,-1]])
    background=np.median(border,axis=0);direction=background-pigment
    contrast=float(direction@direction)
    if contrast<900:return center,core_diameter,None
    alpha=np.clip(((background-tile)@direction)/contrast,0,1)
    residual=np.linalg.norm(tile-(background-alpha[:,:,None]*direction),axis=2)
    ink=((alpha>.03)&(residual<30)).astype('uint8')
    _,labels,_,_=cv2.connectedComponentsWithStats(ink,8)
    at=np.clip(np.rint(center-lo).astype(int),0,np.array(tile.shape[1::-1])-1)
    label=labels[at[1],at[0]]
    if label==0:return center,core_diameter,None
    weights=alpha*(labels==label);area=float(weights.sum())
    diameter=float(2*np.sqrt(area/np.pi))
    if not .7*core_diameter<=diameter<=2.2*core_diameter:return center,core_diameter,None
    y,x=np.indices(weights.shape);weighted=np.array([(x*weights).sum(),(y*weights).sum()])/area+lo
    return weighted,diameter,area


def observe(source, rows, arrows=()):
    rgb=np.asarray(source.convert('RGB'));result=[];occurrences={}
    from .handoff import arrow_strokes, distances
    for row in rows:
        if ''.join(row.get('text','').split()) not in ('...','…','⋯','⋮','⋱','⋰'):continue
        color=row.get('color','#000000')
        if not isinstance(color,str) or len(color)!=7:continue
        pigment=np.array([int(color[i:i+2],16) for i in (1,3,5)])
        b=np.asarray(row['bbox_2d'],float);span=b[2:]-b[:2]
        if min(span)<=0:continue
        margin=max(2.,min(span)/2,.25*max(span))
        lo=np.maximum(0,np.floor(b[:2]-margin)).astype(int)
        hi=np.minimum(rgb.shape[1::-1],np.ceil(b[2:]+margin)).astype(int)
        if np.any(hi<=lo):continue
        mask=(np.linalg.norm(rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-pigment,axis=2)<70).astype('uint8')
        _,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        dots=[(center+lo,float(w),float(h),int(area)) for (x,y,w,h,area),center in zip(stats[1:],centers[1:])
              if compact_component(w,h,area)]
        dots=unique_triplet(dots)
        if dots is None:continue
        xy=np.array([d[0] for d in dots]);diam=np.array([max(d[1:3]) for d in dots])
        if diam.max()>1.8*diam.min():continue
        _,_,axes=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
        along=np.sort(xy@axes[0]);spacing=np.diff(along)
        if min(spacing)<1.2*diam.max() or max(spacing)>1.8*min(spacing):continue
        if np.ptp(xy@axes[1])>.4*diam.max():continue
        measured=[visible_dot(rgb,p,max(w,h),pigment,min(spacing)) for p,w,h,area in dots]
        # Three shaft dots are still part of their arrow, not punctuation.
        on_arrow=False
        for arrow in arrows:
            heads=arrow.get('heads') or [arrow.get('style',{}).get(k) for k in ('start','end')]
            if not any(h not in (None,'none') for h in heads):continue
            paths=arrow_strokes(arrow)
            if paths and np.all(distances(xy,paths)<max(1.,.4*diam.min())):
                on_arrow=True;break
        if on_arrow:continue
        # Different inventories may name the SAME original components twice.
        # Their exact source ink, rather than spelling or text-box proximity,
        # establishes one physical punctuation occurrence.
        identity=tuple(sorted(tuple(np.round(p,6)) for p in xy))
        if identity in occurrences:
            owned=occurrences[identity]
            if row['id']!=owned['text_id']:
                owned.setdefault('duplicate_text_ids',[]).append(row['id'])
            continue
        mark={'text_id':row['id'],'text':row['text'],'color':color,
            'dots':[{'center':p.tolist(),'diameter':float(d),'source_ink_area_px':area,
                     'diameter_method':'source_ink_area' if area is not None else 'compact_component'} for p,d,area in measured],
            'reason':'three_compact_collinear_source_components', 'font_independent':True}
        occurrences[identity]=mark;result.append(mark)
    return result


def clear_image_copies(image,placement,marks):
    """Retire raster copies of registered native dots on a flat local surface.

    Source occurrence registration authorizes the cleanup. An actual complete
    dot pattern in the asset and its surrounding flat RGBA are still required;
    neither a text rectangle nor arbitrary small image components are erased.
    """
    rgba=np.asarray(image.convert('RGBA')).copy();audit=[]
    scale=np.asarray(image.size)/(np.asarray(placement[2:])-placement[:2])
    for mark in marks:
        centers=(np.asarray([d['center'] for d in mark['dots']])+.5-placement[:2])*scale-.5
        diam=np.asarray([d['diameter'] for d in mark['dots']])*min(scale)
        lo=(centers-diam[:,None]/2).min(0);hi=(centers+diam[:,None]/2).max(0)
        if np.any(lo<0) or np.any(hi>=image.size):continue
        painted=Image.new('RGBA',image.size,'white');painted.alpha_composite(Image.fromarray(rgba))
        found=observe(painted.convert('RGB'),[{'id':mark['text_id'],'text':mark['text'],
            'color':mark['color'],'bbox_2d':[*lo,*hi]}])
        if len(found)!=1:continue
        dots=found[0]['dots'];actual=np.asarray([d['center'] for d in dots])
        distances=np.linalg.norm(actual[:,None]-centers[None,:],axis=2)
        if max(distances.min(0).max(),distances.min(1).max())>max(2.,diam.max()):continue
        patches=[]
        for dot in dots:
            center=np.asarray(dot['center']);radius=dot['diameter']/2+1
            lower=np.maximum(0,np.floor(center-radius-3)).astype(int)
            upper=np.minimum(image.size,np.ceil(center+radius+4)).astype(int)
            tile=rgba[lower[1]:upper[1],lower[0]:upper[0]]
            y,x=np.indices(tile.shape[:2]);distance=np.hypot(x+lower[0]-center[0],y+lower[1]-center[1])
            ring=tile[(distance>radius)&(distance<=radius+2)]
            if len(ring)<8:break
            pigment=ring.astype(float);pigment[:,:3]*=pigment[:,3:]/255
            if np.quantile(np.max(abs(pigment-np.median(pigment,axis=0)),axis=1),.95)>12:break
            patches.append((lower,upper,distance<=radius,np.median(ring,axis=0).astype('uint8')))
        if len(patches)!=3:continue
        for lower,upper,mask,background in patches:
            rgba[lower[1]:upper[1],lower[0]:upper[0]][mask]=background
        audit.append({'text_id':mark['text_id'],'source_dot_centers':[d['center'] for d in mark['dots']],
                      'raster_dot_centers':actual.tolist(),'replacement':'source_measured_native_dots',
                      'cleanup':'complete_raster_triplet_on_flat_local_surface'})
    return Image.fromarray(rgba),audit


def replace(path, marks, size):
    if not marks:return
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.dml.color import RGBColor
    from pptx.oxml.xmlchemy import OxmlElement
    prs=Presentation(str(path));slide=prs.slides[0];scale=prs.slide_width/size[0]
    ids={ident for m in marks for ident in [m['text_id'],*m.get('duplicate_text_ids',[])]}
    primary={prefix+i for i in ids for prefix in ('TXT ','EQ ')}
    removals=[]
    for node in list(slide.shapes._spTree):
        names=set(node.xpath('.//*[local-name()="cNvPr"]/@name'))
        dot_owners={identity[0] for name in names if (identity:=native_dot_identity(name))}
        if dot_owners & ids:
            # Recomposition replaces the same physical occurrence; an old
            # measured triplet must not survive beside its regenerated copy.
            if len(dot_owners)!=1 or dot_owners-ids or any(native_dot_identity(name) is None for name in names):
                raise ValueError('Ambiguous standalone punctuation owner')
            removals.append(node)
            continue
        owned=names&primary
        if not owned:continue
        ident=next(iter(owned)).split(' ',1)[1]
        if len(owned)!=1 or names-{*owned,'MATH_PREVIEW '+ident}:
            raise ValueError('Ambiguous standalone punctuation owner')
        removals.append(node)
    for node in removals:
        node.getparent().remove(node)
    if len({mark['text_id'] for mark in marks}) != len(marks):
        raise ValueError('Duplicate source punctuation occurrence')
    for mark in marks:
        for index,dot in enumerate(mark['dots']):
            x,y=dot['center'];d=dot['diameter']
            shape=slide.shapes.add_shape(MSO_SHAPE.OVAL,round((x+.5-d/2)*scale),
                round((y+.5-d/2)*scale),max(1,round(d*scale)),max(1,round(d*scale)))
            shape.name='SOURCE_ELLIPSIS '+mark['text_id']+'/'+str(index)
            shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(mark['color'][1:])
            shape.line.fill.background()
            # Measured ink has no theme shadow, bevel or glow.
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            for style in shape._element.xpath('./p:style'):
                shape._element.remove(style)
    target=Path(path);temp=target.with_name(target.stem+'.source_ellipsis.pptx');prs.save(temp);temp.replace(target)
