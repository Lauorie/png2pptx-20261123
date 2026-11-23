"""Translate subpixel text-frame overflow without changing glyphs or box size."""
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree


def normalize_native_text_frames(path, source_size, *, texts=()):
    """Represent on-page glyphs with on-page native frames, without moving ink.

    Quarter-turn labels use native vertical text rather than a wide rotated
    frame. Font leading outside the canvas becomes an inset compensation;
    the inner text rectangle, paragraphs and runs remain unchanged.
    """
    import math
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with ZipFile(path) as archive:package={n:archive.read(n) for n in archive.namelist()}
    canvas=etree.fromstring(package['ppt/presentation.xml']).find('p:sldSz',ns)
    size=[int(canvas.get(k)) for k in ('cx','cy')];scale=[size[i]/source_size[i] for i in (0,1)]
    slide=etree.fromstring(package['ppt/slides/slide1.xml']);rows={r['id']:r for r in texts};changes=[]
    for shape in slide.findall('p:cSld/p:spTree/p:sp',ns):
        name=shape.find('p:nvSpPr/p:cNvPr',ns);t=shape.find('p:spPr/a:xfrm',ns)
        body=shape.find('p:txBody/a:bodyPr',ns)
        if name is None or t is None or body is None or not name.get('name','').startswith('TXT '):continue
        if body.get('wrap')!='none' or body.find('a:noAutofit',ns) is None:continue
        ident=name.get('name')[4:];row=rows.get(ident,{})
        if row.get('kind')!='text' or row.get('layout_mode','single_line')!='single_line':continue
        if any(t.get(k,'0') not in ('0','false') for k in ('flipH','flipV')):continue
        off=t.find('a:off',ns);ext=t.find('a:ext',ns)
        if off is None or ext is None:continue
        x,y=int(off.get('x')),int(off.get('y'));w,h=int(ext.get('cx')),int(ext.get('cy'))
        if min(w,h)<=0 or (x>=0 and y>=0 and x+w<=size[0] and y+h<=size[1]):continue
        angle=int(t.get('rot','0'))%21600000
        if angle not in (0,5400000,16200000) or body.get('vert','horz')!='horz':continue
        b=row.get('bbox_2d',[]);m=row.get('layout',{}).get('measurement',{})
        if (len(b)!=4 or not (0<=b[0]<b[2]<=source_size[0] and 0<=b[1]<b[3]<=source_size[1])
                or m.get('method')!='native_ppt_atlas' or m.get('reference_size_px',0)<=0):continue
        factor=row.get('font_size_px',0)/m['reference_size_px']
        if factor<=0:continue
        margins=[int(body.get(k,'0')) for k in ('lIns','tIns','rIns','bIns')]
        # Atlas metrics come from a 2x raster: each measured edge is quantized
        # to half a reference pixel. Preserve that uncertainty after scaling,
        # rather than treating its subpixel phase as genuinely off-page ink.
        # Larger off-page ink remains ineligible for frame normalization.
        a=angle/60000*math.pi/180;c,s=math.cos(a),math.sin(a)
        ink=[]
        for u in (m['dx'],m['dx']+m['width']):
            for v in (m['dy'],m['dy']+m['height']):
                px=margins[0]+u*factor*scale[0]-w/2
                py=margins[1]+v*factor*scale[1]-h/2
                ink.append((x+w/2+c*px-s*py,y+h/2+s*px+c*py))
        quantization=[.5*factor*scale[i] for i in (0,1)]
        tolerance=[abs(c)*quantization[0]+abs(s)*quantization[1],
                   abs(s)*quantization[0]+abs(c)*quantization[1]]
        if any(p[i]<-tolerance[i] or p[i]>size[i]+tolerance[i] for p in ink for i in (0,1)):continue
        before={'frame_emu':[x,y,w,h],'rotation':angle,'insets_emu':margins[:]}
        if angle:
            x,y,w,h=round(x+(w-h)/2),round(y+(h-w)/2),h,w
            margins=([margins[3],margins[0],margins[1],margins[2]] if angle==5400000
                     else [margins[1],margins[2],margins[3],margins[0]])
            t.set('rot','0');body.set('vert','vert' if angle==5400000 else 'vert270')
        cuts=[max(0,-x),max(0,-y),max(0,x+w-size[0]),max(0,y+h-size[1])]
        x+=cuts[0];y+=cuts[1];w-=cuts[0]+cuts[2];h-=cuts[1]+cuts[3]
        assert w>0 and h>0
        margins=[margins[i]-cuts[i] for i in range(4)]
        off.set('x',str(x));off.set('y',str(y));ext.set('cx',str(w));ext.set('cy',str(h))
        for key,value in zip(('lIns','tIns','rIns','bIns'),margins):body.set(key,str(value))
        changes.append({'text_id':ident,'reason':'native_vertical_frame' if angle else 'native_font_leading',
            'before':before,'after':{'frame_emu':[x,y,w,h],'rotation':0,'vert':body.get('vert','horz'),'insets_emu':margins},
            'frame_px':[x/scale[0],y/scale[1],w/scale[0],h/scale[1]],
            'insets_px':[margins[i]/scale[i%2] for i in range(4)],'glyph_translation_px':[0.,0.],
            'ink_quantization_tolerance_px':[tolerance[i]/scale[i] for i in (0,1)]})
    if changes:
        package['ppt/slides/slide1.xml']=etree.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
        with ZipFile(path,'w',ZIP_DEFLATED) as archive:
            for n,data in package.items():archive.writestr(n,data)
    return changes


def clamp_small_text_overflow(path, source_size, max_shift_px=1.0, *, texts=()):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with ZipFile(path) as archive:
        package={name:archive.read(name) for name in archive.namelist()}
    canvas=etree.fromstring(package['ppt/presentation.xml']).find('p:sldSz',ns)
    size=[int(canvas.get(k)) for k in ('cx','cy')]
    scale=[size[i]/source_size[i] for i in (0,1)]
    slide=etree.fromstring(package['ppt/slides/slide1.xml']);changes=[]
    rows={r['id']:r for r in texts}
    # Run on the standalone text deck before composition; grouped/rotated text
    # requires transformed bounds and is deliberately outside this small fix.
    for shape in slide.findall('p:cSld/p:spTree/p:sp',ns):
        ident=shape.find('p:nvSpPr/p:cNvPr',ns)
        transform=shape.find('p:spPr/a:xfrm',ns)
        if ident is None or not ident.get('name','').startswith('TXT '):continue
        if transform is None or int(transform.get('rot','0'))%21600000:continue
        off=transform.find('a:off',ns);extent=transform.find('a:ext',ns)
        if off is None or extent is None:continue
        old=[int(off.get(k)) for k in ('x','y')]
        wh=[int(extent.get(k)) for k in ('cx','cy')]
        if any(wh[i]<=0 or wh[i]>size[i] for i in (0,1)):continue
        new=[min(max(old[i],0),size[i]-wh[i]) for i in (0,1)]
        if new==old:continue
        allowance=[max_shift_px,max_shift_px];reason='subpixel_frame_overflow'
        row=rows.get(ident.get('name')[4:],{})
        measured=row.get('layout',{}).get('measurement',{})
        box=row.get('bbox_2d',[])
        # Native font ascender space may put the frame above the page while
        # the measured/source ink is entirely on-page. Bound the translation
        # by that measured leading, rather than an arbitrary larger threshold.
        if (old[1]<0 and row.get('kind')=='text' and len(box)==4
                and 0<=box[0]<box[2]<=source_size[0] and 0<=box[1]<box[3]<=source_size[1]
                and measured.get('method')=='native_ppt_atlas' and measured.get('reference_size_px',0)>0):
            font=row.get('font_size_px',0)
            leading=max(0,measured.get('dy',0))*font/measured['reference_size_px']
            allowance[1]=max(max_shift_px,min(leading,.35*font))
            reason='measured_font_leading_overflow'
        if any(abs(new[i]-old[i])>allowance[i]*scale[i] for i in (0,1)):continue
        if len(box)==4:
            shift=[(new[i]-old[i])/scale[i] for i in (0,1)]
            moved=[box[i]+shift[i%2] for i in range(4)]
            def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if any(len(other.get('bbox_2d',[]))==4 and other['id']!=row['id']
                   and area(moved,other['bbox_2d'])>area(box,other['bbox_2d'])+.5 for other in texts):continue
        for i,k in enumerate(('x','y')):off.set(k,str(new[i]))
        changes.append({'text_id':ident.get('name')[4:],'reason':reason,'before_xy_emu':old,'after_xy_emu':new,
                        'after_xy_px':[new[i]/scale[i] for i in (0,1)],
                        'shift_px':[(new[i]-old[i])/scale[i] for i in (0,1)]})
    if changes:
        package['ppt/slides/slide1.xml']=etree.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
        with ZipFile(path,'w',ZIP_DEFLATED) as archive:
            for name,data in package.items():archive.writestr(name,data)
    return changes
