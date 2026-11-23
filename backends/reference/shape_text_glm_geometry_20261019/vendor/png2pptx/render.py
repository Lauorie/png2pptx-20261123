"""Diagnostic overlays and editable native PowerPoint geometry (no source styles yet)."""
import base64
import hashlib
import math
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image,ImageDraw,ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches,Pt
from .fitting import sample_commands
from .frames import scale_point


def color(ident):
    palette=['#d73027','#4575b4','#1a9850','#984ea3','#e08214','#008b8b']
    return palette[int(hashlib.sha256(ident.encode()).hexdigest()[:8],16)%len(palette)]


def font(size=18):
    try:return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:return ImageFont.load_default()


def candidate_overlay(image,box,ident):
    out=image.copy();d=ImageDraw.Draw(out);d.rectangle(box,outline='#ff00cc',width=3)
    d.text((max(0,box[0]),max(0,box[1]-22)),ident,fill='#ff00cc',font=font())
    return out


def global_overlay(scene,image,include_candidates=True,points=False):
    out=image.copy();draw=ImageDraw.Draw(out)
    if include_candidates:
        for o in scene['objects'].values():
            b=o['bbox_global_pixels'];draw.rectangle(b,outline='#999999',width=1)
            draw.text((max(0,b[0]),max(0,b[1]-18)),o['id'],fill='#777777',font=font(15))
    for o in scene['objects'].values():
        if 'fit' not in o or not o.get('export_enabled',True):continue
        c=color(o['id'])
        for part in o['fit']['parts']:
            pts=[scale_point(p,scene['image']['source_size'],image.size) for p in sample_commands(part['commands'],32)]
            if len(pts)>1:draw.line([tuple(p) for p in pts],fill=c,width=2)
        b=o['fitted_bbox_global_pixels'];draw.text((max(0,b[0]),max(0,b[1]-24)),o['id'],fill=c,font=font(21),stroke_width=1,stroke_fill='white')
        if points:
            for pid in o['active_point_ids']:
                x,y=scene['points'][pid]['xy_global_input'];draw.ellipse((x-2,y-2,x+2,y+2),fill=c)
    return out


def local_overlay(scene,ident,crop,image):
    out=image.copy();draw=ImageDraw.Draw(out);o=scene['objects'][ident]
    if 'fit' not in o:return out
    for part in o['fit']['parts']:
        pts=[crop.to_local(scale_point(p,scene['image']['source_size'],crop.global_size)) for p in sample_commands(part['commands'],32)]
        draw.line([tuple(p) for p in pts],fill=color(ident),width=3)
    for pid in o['active_point_ids']:
        p=scene['points'][pid];x,y=crop.to_local(p['xy_global_input'])
        draw.ellipse((x-4,y-4,x+4,y+4),fill='#ff00cc')
        draw.text((x+5,y+5),p['local_id'],fill='#9c009c',font=font(17))
    return out


def fitted_objects(scene,accepted_only=False):
    return [o for o in scene['objects'].values() if 'fit' in o and o.get('export_enabled',True) and (not accepted_only or o['status']=='accepted_by_model')]


def svg(scene,accepted_only=False,asset_root=None):
    from .images.scene import exportable_assets,asset_path
    w,h=scene['image']['source_size']
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="-0.5 -0.5 {w} {h}">',
           '<rect x="-0.5" y="-0.5" width="100%" height="100%" fill="white"/>']
    for asset in exportable_assets(scene,accepted_only):
        x0,y0,x1,y1=asset['placement_source_half_open']
        encoded=base64.b64encode(asset_path(scene,asset,asset_root).read_bytes()).decode()
        identity=escape(asset['id'],{'"':'&quot;'})
        lines.append(f'<image id="image_{identity}" x="{x0-.5}" y="{y0-.5}" width="{x1-x0}" height="{y1-y0}" preserveAspectRatio="none" xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="data:image/png;base64,{encoded}"/>')
    for o in fitted_objects(scene,accepted_only):
        for p in o['fit']['parts']:
            path=' '.join(c[0]+' '+' '.join(f'{x:.5f}' for x in c[1:]) for c in p['commands'])
            identity=escape(o['id']+'_'+p['id'],{'"':'&quot;'})
            lines.append(f'<path id="{identity}" d="{path}" fill="none" stroke="{color(o["id"])}" stroke-width="1.2"/>')
    return '\n'.join(lines+['</svg>'])


def custom_geometry(shape,commands,bounds):
    bx,by,ex,ey=bounds;unit=1000
    geom=OxmlElement('a:custGeom')
    for name in ['avLst','gdLst','ahLst','cxnLst']:geom.append(OxmlElement('a:'+name))
    rect=OxmlElement('a:rect')
    for k,v in {'l':'0','t':'0','r':'r','b':'b'}.items():rect.set(k,v)
    geom.append(rect);paths=OxmlElement('a:pathLst');path=OxmlElement('a:path')
    path.set('w',str(max(1,round((ex-bx)*unit))));path.set('h',str(max(1,round((ey-by)*unit))));path.set('fill','none')
    for cmd in commands:
        node=OxmlElement('a:'+{'M':'moveTo','L':'lnTo','C':'cubicBezTo','Z':'close'}[cmd[0]])
        for i in range(1,len(cmd),2):
            pt=OxmlElement('a:pt');pt.set('x',str(round((cmd[i]-bx)*unit)));pt.set('y',str(round((cmd[i+1]-by)*unit)));node.append(pt)
        path.append(node)
    paths.append(path);geom.append(paths)
    sppr=shape._element.spPr;old=sppr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom')
    sppr.replace(old,geom)


def write_pptx(scene,path,accepted_only=False,asset_root=None):
    from .images.scene import exportable_assets,asset_path
    prs=Presentation();w,h=scene['image']['source_size'];scale=Inches(13.333333)/w
    prs.slide_width=round(w*scale);prs.slide_height=round(h*scale)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);count=0
    for asset in exportable_assets(scene,accepted_only):
        x0,y0,x1,y1=asset['placement_source_half_open']
        picture=slide.shapes.add_picture(str(asset_path(scene,asset,asset_root)),round(x0*scale),round(y0*scale),max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale)))
        picture.name=asset['id']+' ['+asset['status']+']'
    # Return value remains the native geometry count for existing callers.
    for o in fitted_objects(scene,accepted_only):
        for p in o['fit']['parts']:
            params=p['parameters'];kind=p['kind']
            if kind in ('rectangle','rounded_rectangle','ellipse'):
                cx,cy=params['center']
                if kind=='ellipse':width,height=[2*r for r in params['radii']];shape_type=MSO_AUTO_SHAPE_TYPE.OVAL
                else:width,height=params['width'],params['height'];shape_type=MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if kind=='rounded_rectangle' else MSO_AUTO_SHAPE_TYPE.RECTANGLE
                shape=slide.shapes.add_shape(shape_type,round((cx-width/2+0.5)*scale),round((cy-height/2+0.5)*scale),max(1,round(width*scale)),max(1,round(height*scale)))
                shape.rotation=math.degrees(params['angle_radians'])%360
                if kind=='rounded_rectangle':shape.adjustments[0]=params['radius']/min(width,height)
            else:
                all_points=[(c[i],c[i+1]) for c in p['commands'] for i in range(1,len(c),2)]
                xs,ys=zip(*all_points);bx,by,ex,ey=min(xs),min(ys),max(xs),max(ys)
                # A straight open span can have zero extent on one axis; a tiny representational extent retains its path.
                ex=max(ex,bx+1e-3);ey=max(ey,by+1e-3)
                shape=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,round((bx+0.5)*scale),round((by+0.5)*scale),max(1,round((ex-bx)*scale)),max(1,round((ey-by)*scale)))
                custom_geometry(shape,p['commands'],(bx,by,ex,ey))
            shape.name=o['id']+'/'+p['id']+' ['+o['status']+']';shape.fill.background()
            shape.line.color.rgb=RGBColor.from_string(color(o['id'])[1:]);shape.line.width=Pt(1)
            # python-pptx's default theme otherwise adds shadow effects to new AutoShapes.
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            style=shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
            if style is not None:shape._element.remove(style)
            count+=1
    prs.save(path);return count


def export_scene(scene,folder,image):
    import cairosvg
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    global_overlay(scene,image).save(folder/'inventory_overlay.png')
    global_overlay(scene,image,False,True).save(folder/'geometry_overlay.png')
    counts={}
    for name,accepted in [('geometry_diagnostic',False),('geometry_accepted',True)]:
        content=svg(scene,accepted);(folder/(name+'.svg')).write_text(content)
        cairosvg.svg2png(bytestring=content.encode(),write_to=str(folder/(name+'.png')))
        counts[name]=write_pptx(scene,folder/(name+'.pptx'),accepted)
    return counts
