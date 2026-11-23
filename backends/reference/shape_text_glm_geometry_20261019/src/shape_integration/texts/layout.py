"""Font-file measurement and optional one-batch native metrics; no VLM sizes."""
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
import fitz
from PIL import Image, ImageFont
from ..common import save
from ..resource_limits import bounded


@lru_cache(maxsize=64)
def resolve_font(family, bold=False, italic=False):
    pattern = family + ':weight=' + ('bold' if bold else 'regular') + ':slant=' + ('italic' if italic else 'roman')
    # fc-match is a system utility. Loading Conda's shared-storage libraries
    # makes cold font queries stall; keep fontconfig settings but isolate libs.
    font_env={k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','LD_PRELOAD')}
    proc = subprocess.run(['fc-match', '-f', '%{file}\n%{family[0]}\n', pattern],
                          capture_output=True, text=True, check=True, timeout=10,env=font_env)
    path, actual = proc.stdout.strip().splitlines()[:2]
    if not Path(path).is_file():
        raise ValueError('Font file unavailable: ' + family)
    return {'requested': family, 'resolved': actual, 'path': path, 'fallback': actual.casefold() != family.casefold()}


def pillow_metrics(row, size=24.):
    """Measure scripted runs on a shared baseline in a high-resolution font."""
    if row.get('layout_mode')=='stacked_vertical' and row.get('kind')!='formula':
        from .layout_integrity import stacked_runs
        glyphs=stacked_runs(row)
        if not glyphs:raise ValueError('Empty stacked label')
        metrics=[pillow_metrics({**row,'layout_mode':'single_line','runs':runs},size) for runs in glyphs]
        # Export uses this exact baseline advance, not font-dependent wrapping.
        top=min(m['dy']+i*1.2*size for i,m in enumerate(metrics))
        bottom=max(m['dy']+m['height']+i*1.2*size for i,m in enumerate(metrics))
        return {**metrics[0],'width':max(m['dx']+m['width'] for m in metrics)-min(m['dx'] for m in metrics),
            'height':bottom-top,'dx':min(m['dx'] for m in metrics),'dy':top,
            'method':'pillow_stacked_vertical'}
    font_info = resolve_font(row['font'], row['bold'], row['italic'])
    cursor = 0.; boxes = []
    for run in row['runs']:
        small = run['script'] != 'normal'
        run_font = resolve_font(row['font'],run.get('bold',row['bold']),run.get('italic',row['italic']))
        font = ImageFont.truetype(run_font['path'], max(1, round(size * (0.58 if small else 1) * 4)))
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
        atlas.append({**item,'color':'#000000','runs':[{**r,'color':'#000000'} for r in item['runs']],
                      'background_fill':None,'background_evidence':'',
                      'rotation_deg':0,'font_size_px':reference,
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
    characters=[]
    with fitz.open(folder/'native.pdf') as document:
        page=document[0];sx=size[0]/page.rect.width;sy=size[1]/page.rect.height
        for block in page.get_text('rawdict')['blocks']:
            for line in block.get('lines',[]):
                for span in line.get('spans',[]):
                    for char in span.get('chars',[]):
                        b=char['bbox']
                        characters.append({'text':char['c'],'bbox':[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy]})
    output={}
    for ident,cell in cells.items():
        x,y,w,h=cell['cell'];fx,fy,fw,fh=cell['frame']
        tile=pixels[y*2:(y+h)*2,x*2:(x+w)*2]
        yy,xx=np.where(tile.min(axis=2)<200)
        if not len(xx) or min(xx.min(),yy.min())<2 or xx.max()>=tile.shape[1]-2 or yy.max()>=tile.shape[0]-2:
            raise ValueError('Empty or clipped native text measurement: '+ident)
        output[ident]={**cell['initial'],'width':(xx.max()-xx.min()+1)/2,'height':(yy.max()-yy.min()+1)/2,
                       'dx':xx.min()/2+x-fx,'dy':yy.min()/2+y-fy,'method':'native_ppt_atlas'}
        bounds=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        ink=255-tile[bounds[1]:bounds[3],bounds[0]:bounds[2]].min(axis=2)
        rgba=np.zeros((*ink.shape,4),dtype=np.uint8);rgba[:,:,3]=ink
        template=folder/(ident+'_ink.png');Image.fromarray(rgba).save(template)
        output[ident]['ink_template']=str(template.resolve())
        glyphs=[]
        for char in characters:
            b=char['bbox'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            if not (x<=cx<x+w and y<=cy<y+h):continue
            x0,y0=max(0,int(math.floor(b[0]*2))),max(0,int(math.floor(b[1]*2)))
            x1,y1=min(pixels.shape[1],int(math.ceil(b[2]*2))),min(pixels.shape[0],int(math.ceil(b[3]*2)))
            gy,gx=np.where(pixels[y0:y1,x0:x1].min(axis=2)<180)
            stride=max(1,int(math.ceil(len(gx)/256)))
            points=np.column_stack(((gx[::stride]+x0+.5)/2-fx,(gy[::stride]+y0+.5)/2-fy)).tolist()
            glyphs.append({'text':char['text'],'bbox':[b[0]-fx,b[1]-fy,b[2]-fx,b[3]-fy],'ink_points':points})
        output[ident]['characters']=glyphs
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


@bounded('cpu')
def fit_all(rows, folder, slots, opt, source=None):
    from .occurrence_dedup import deduplicate
    visible=[r for r in rows if r['text'].strip()]
    visible,audit=deduplicate(visible,opt.get('duplicate_iou',.65)) if source is not None else (visible,[])
    save(folder/'occurrence_dedup.json',audit)
    errors=[]
    if not visible:return [],errors
    if source is not None and opt.get('pixel_reflow_enabled',True):
        from .reflow import refine as reflow
        visible=reflow(visible,source,folder)
    plain=[r for r in visible if r.get('kind','text')!='formula']
    formulas=[r for r in visible if r.get('kind')=='formula']
    measurements={}
    if plain and opt['native_font_metrics']:
        try:
            if source is not None and opt.get('source_font_selection',True):
                from .font_selection import select
                plain,measurements=select(plain,source,folder/'font_selection',slots,opt)
                selected={r['id']:r for r in plain}
                visible=[selected.get(r['id'],r) for r in visible]
            else:measurements=native_metrics(plain,folder/'metrics',slots,opt)
        except Exception as exc:
            errors.append('Native font measurement failed; explicit Pillow fallback: '+str(exc))
            save(folder/'metrics_error.json',{'error':str(exc),'fallback':'pillow'})
            measurements={r['id']:pillow_metrics(r) for r in plain}
    elif plain:measurements={r['id']:pillow_metrics(r) for r in plain}
    if formulas:
        if not opt.get('native_equations',True):raise ValueError('Formula input requires native_equations=true')
        from .equation_recovery import measure_resilient
        recovered,audit=measure_resilient(formulas,folder/'equations',slots,opt['render_timeout_seconds'])
        measurements.update(recovered)
        errors.extend('Formula preview fallback '+entry['id']+': '+entry['policy'] for entry in audit)
        if source is not None and opt.get('formula_layout_enabled',True):
            from .formula_layout import select as select_formula_layout
            try:
                formulas,measurements=select_formula_layout(formulas,measurements,source,
                    folder/'formula_layout',slots,opt)
            except Exception as exc:
                save(folder/'formula_layout/error.json',{'error':str(exc),'fallback':'native_baseline'})
                errors.append('Formula typography retained native baseline: '+str(exc))
            selected={r['id']:r for r in formulas}
            visible=[selected.get(r['id'],r) for r in visible]
    if source is not None and opt.get('glyph_registration_enabled',False):
        from .registration import refine
        visible,measurements=refine(visible,measurements,source,folder)
    fitted=[fit(r,measurements[r['id']],opt['font_fit_fraction']) for r in visible]
    save(folder/'layouts.json',fitted)
    return fitted,errors
