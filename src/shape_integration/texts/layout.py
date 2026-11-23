"""Font-file measurement and optional one-batch native metrics; no VLM sizes."""
import math
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont
from ..common import save


@lru_cache(maxsize=64)
def resolve_font(family, bold=False, italic=False):
    pattern = family + ':weight=' + ('bold' if bold else 'regular') + ':slant=' + ('italic' if italic else 'roman')
    proc = subprocess.run(['fc-match', '-f', '%{file}\n%{family[0]}\n', pattern],
                          capture_output=True, text=True, check=True, timeout=10)
    path, actual = proc.stdout.strip().splitlines()[:2]
    if not Path(path).is_file():
        raise ValueError('Font file unavailable: ' + family)
    return {'requested': family, 'resolved': actual, 'path': path, 'fallback': actual.casefold() != family.casefold()}


def pillow_metrics(row, size=24.):
    """Measure scripted runs on a shared baseline in a high-resolution font."""
    font_info = resolve_font(row['font'], row['bold'], row['italic'])
    cursor = 0.; boxes = []
    for run in row['runs']:
        small = run['script'] != 'normal'
        font = ImageFont.truetype(font_info['path'], max(1, round(size * (0.58 if small else 1) * 4)))
        shift = {'normal': 0, 'sub': .25, 'super': -.3}[run['script']] * size
        x0,y0,x1,y1 = [v/4 for v in font.getbbox(run['text'], anchor='ls')]
        boxes.append([cursor+x0, y0+shift, cursor+x1, y1+shift])
        cursor += font.getlength(run['text'])/4
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
    x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
    # Baseline placement is approximate only when native measurement is disabled.
    return {'width': max(.25,x1-x0), 'height': max(.25,y1-y0),
            'dx': x0, 'dy': size*.9+y0, 'reference_size_px': size,
            'font': font_info, 'method': 'pillow'}


def projected(width, height, angle):
    a=math.radians(angle);c=abs(math.cos(a));s=abs(math.sin(a))
    return c*width+s*height, s*width+c*height


def native_metrics(rows, folder, slots, opt):
    """One black-on-white measurement sheet; source images are never modified."""
    from .export import write
    from ..styling.export import render_native
    folder.mkdir(parents=True, exist_ok=True)
    reference=24.;padding=24;limit=4096;x=y=0;row_height=0;cells={};atlas=[];width=1
    for item in rows:
        m=pillow_metrics(item, reference)
        fw=max(100, math.ceil(m['width']+reference*5));fh=max(100, math.ceil(m['height']+reference*4))
        cw=fw+padding*2;ch=fh+padding*2
        if x and x+cw>limit:x=0;y+=row_height;row_height=0
        cells[item['id']]={'cell':[x,y,cw,ch], 'frame':[x+padding,y+padding,fw,fh], 'initial':m,
                          'rotation_deg':item['rotation_deg']}
        atlas.append({**item,'color':'#000000','rotation_deg':0,'font_size_px':reference,
                      'resolved_font':m['font']['resolved'],'frame_px':cells[item['id']]['frame']})
        x+=cw;row_height=max(row_height,ch);width=max(width,x)
    size=[width,max(1,y+row_height)]
    # A single atlas row of short labels can be wider than Office's minimum
    # slide-height ratio. Pad unused white space, without moving measurement cells.
    size=[max(size[0],math.ceil(size[1]*13.333333/56)),
          max(size[1],math.ceil(size[0]/13.333333))]
    # Fail explicitly instead of silently clipping a pathological measurement sheet.
    if size[0]*size[1]>24_000_000 or max(size)>16000:
        raise ValueError('Text measurement sheet too large')
    write(atlas,size,folder/'metrics.pptx')
    with slots:render_native(folder/'metrics.pptx',folder/'native_render',[v*2 for v in size],opt['render_timeout_seconds'])
    with Image.open(folder/'native.png') as im:pixels=np.asarray(im.convert('RGB'))
    output={}
    for ident,cell in cells.items():
        x,y,w,h=cell['cell'];fx,fy,fw,fh=cell['frame']
        tile=pixels[y*2:(y+h)*2,x*2:(x+w)*2]
        yy,xx=np.where(tile.min(axis=2)<200)
        if not len(xx) or min(xx.min(),yy.min())<2 or xx.max()>=tile.shape[1]-2 or yy.max()>=tile.shape[0]-2:
            raise ValueError('Empty or clipped native text measurement: '+ident)
        output[ident]={**cell['initial'],'width':(xx.max()-xx.min()+1)/2,'height':(yy.max()-yy.min()+1)/2,
                       'dx':xx.min()/2+x-fx,'dy':yy.min()/2+y-fy,'method':'native_ppt_atlas'}
        # Rotate actual glyph ink: the empty corners of an unrotated ink bbox
        # are not glyphs and can shift the fitted center at non-right angles.
        a=math.radians(cell['rotation_deg']);c=math.cos(a);s=math.sin(a)
        px=(xx+.5)/2+x-fx;py=(yy+.5)/2+y-fy
        rx=c*px-s*py;ry=s*px+c*py;pad=(abs(c)+abs(s))*.25
        output[ident]['rotated_ink_bounds']=[float(rx.min()-pad),float(ry.min()-pad),
                                            float(rx.max()+pad),float(ry.max()+pad)]
    save(folder/'cells.json',cells);save(folder/'measurements.json',output)
    return output


def compact_frame_geometry(measurement, size):
    """Keep the native line origin, with room only for ink and rounding.

    Native metrics include the font's ascent/baseline offset. Positive insets
    accommodate italic/script overhang without negative Office margins.
    """
    factor=size/measurement['reference_size_px']
    dx=measurement['dx']*factor;dy=measurement['dy']*factor
    left=max(0.,-dx);top=max(0.,-dy)
    pad=max(.5,size*.02)
    width=left+dx+measurement['width']*factor+pad
    height=top+dy+measurement['height']*factor+pad
    return width,height,[left,top,0.,0.]


def tighten_frame(row):
    """Shrink a saved layout without refitting its font, bbox or visible origin."""
    fw,fh,insets=compact_frame_geometry(row['layout']['measurement'],row['font_size_px'])
    x,y,w,h=row['frame_px']
    old=row.get('text_insets_px',[0.,0.,0.,0.])
    a=math.radians(row['rotation_deg']);c=math.cos(a);s=math.sin(a)
    # Preserve the world position of the line origin when the rotation center
    # changes, including text-inset compensation for negative glyph bearings.
    vx=old[0]-w/2-insets[0]+fw/2
    vy=old[1]-h/2-insets[1]+fh/2
    frame=[x+w/2+c*vx-s*vy-fw/2,y+h/2+s*vx+c*vy-fh/2,fw,fh]
    return {**row,'frame_px':frame,'text_insets_px':insets,
            'layout':{**row['layout'],'frame_policy':'native_ink_with_line_origin_v1'}}


def fit(row, measurement, fraction=.98):
    """Fit rotated ink, then compensate the Office frame's ink offset."""
    b=row['bbox_2d'];bw=b[2]-b[0]+1;bh=b[3]-b[1]+1
    rotated=measurement.get('rotated_ink_bounds')
    pw,ph=(rotated[2]-rotated[0],rotated[3]-rotated[1]) if rotated else projected(measurement['width'],measurement['height'],row['rotation_deg'])
    factor=min(bw/pw,bh/ph)*fraction;size=measurement['reference_size_px']*factor
    mw=measurement['width']*factor;mh=measurement['height']*factor
    dx=measurement['dx']*factor;dy=measurement['dy']*factor
    fw,fh,insets=compact_frame_geometry(measurement,size)
    left,top=insets[:2]
    a=math.radians(row['rotation_deg']);c=math.cos(a);s=math.sin(a)
    if rotated:
        rx=(rotated[0]+rotated[2])/2*factor+c*left-s*top-(c*fw/2-s*fh/2)
        ry=(rotated[1]+rotated[3])/2*factor+s*left+c*top-(s*fw/2+c*fh/2)
    else:
        vx=left+dx+mw/2-fw/2;vy=top+dy+mh/2-fh/2
        rx=c*vx-s*vy;ry=s*vx+c*vy
    cx=(b[0]+b[2])/2-rx;cy=(b[1]+b[3])/2-ry
    return {**row,'font_size_px':size,'resolved_font':measurement['font']['resolved'],
            'frame_px':[cx-fw/2,cy-fh/2,fw,fh],'text_insets_px':insets,
            'layout':{'measurement':measurement,'scale':factor,'font_fit_fraction':fraction,
                      'predicted_ink_size_px':[mw,mh],'projected_ink_size_px':[pw*factor,ph*factor],
                      'unit':'original_image_pixels','frame_policy':'native_ink_with_line_origin_v1'}}


def fit_all(rows, folder, slots, opt):
    visible=[r for r in rows if r['text'].strip()];errors=[]
    if not visible:return [],errors
    if opt['native_font_metrics']:
        try:measurements=native_metrics(visible,folder/'metrics',slots,opt)
        except Exception as exc:
            errors.append('Native font measurement failed; explicit Pillow fallback: '+str(exc))
            save(folder/'metrics_error.json',{'error':str(exc),'fallback':'pillow'})
            measurements={r['id']:pillow_metrics(r) for r in visible}
    else:measurements={r['id']:pillow_metrics(r) for r in visible}
    fitted=[fit(r,measurements[r['id']],opt['font_fit_fraction']) for r in visible]
    save(folder/'layouts.json',fitted)
    return fitted,errors
