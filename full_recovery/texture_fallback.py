"""One fill review: native patterns first, bounded source pixels if irregular.

Keep the editable boundary and replace only confirmed intrinsic hatch parts.
"""
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image
from lxml import etree
from shapely.geometry import Polygon, LineString, box as rectangle
from shapely.ops import unary_union

from .handoff import strokes, json_part_id

PATTERNS=('smCheck','smGrid','diagCross','ltDnDiag','ltUpDiag','pct25','pct50')
A='http://schemas.openxmlformats.org/drawingml/2006/main'


def apply_patterns(source, output, decisions):
    """Paint existing editable boundaries with native DrawingML patterns."""
    from zipfile import ZipFile, ZIP_DEFLATED
    from lxml import etree
    from .compose import A,P,matches_name
    ns={'a':A,'p':P}
    with ZipFile(source) as archive:package={n:archive.read(n) for n in archive.namelist()}
    slide=etree.fromstring(package['ppt/slides/slide1.xml'])
    for decision in decisions:
        name=decision['shape_name'];matched=[]
        for sp in slide.findall('.//p:sp',ns):
            prop=sp.find('p:nvSpPr/p:cNvPr',ns)
            if prop is not None and matches_name(prop.get('name',''),name):matched.append(sp)
        if len(matched)!=1:raise ValueError('Pattern surface identity is not unique: '+name)
        props=matched[0].find('p:spPr',ns)
        for node in list(props):
            if etree.QName(node).localname in ('noFill','solidFill','gradFill','blipFill','pattFill','grpFill'):props.remove(node)
        fill=etree.Element('{'+A+'}pattFill',prst=decision['pattern'])
        for tag,key in [('fgClr','foreground'),('bgClr','background')]:
            parent=etree.SubElement(fill,'{'+A+'}'+tag)
            etree.SubElement(parent,'{'+A+'}srgbClr',val=decision[key].lstrip('#'))
        line=props.find('a:ln',ns);props.insert(list(props).index(line) if line is not None else len(props),fill)
    package['ppt/slides/slide1.xml']=etree.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for name,data in package.items():archive.writestr(name,data)
    return output


def texture_candidates(scene, edit_enabled=False):
    out=[]
    for oid,obj in scene['objects'].items():
        evidence=' '.join([obj.get('candidate',{}).get('label',''),
                         *[h.get('observation',{}).get('reason','') for h in obj.get('history',[])],
                         *[str(scene['parts'][k].get('id','')) for k in obj.get('part_keys',[]) if k in scene['parts']]])
        expression=r'hatch|textur|pattern|\bgrid\b|网格|纹理|填充图案'
        if edit_enabled:expression+=r'|cloud|backdrop|image.stack|image.plane|mosaic|云朵|背景图案'
        if re.search(expression,evidence,re.I):out.append(oid)
    return out


def polygon(part):
    cs=part.get('commands',[])
    if not cs or cs[-1][0]!='Z':return None
    paths=strokes(cs,samples=48)
    if len(paths)!=1:return None
    value=Polygon(paths[0])
    return value if value.is_valid and value.area>4 else None


def safe_surface(scene,styles,arrows,texts,surface_key,replace_keys,images=()):
    """Keep source pixels only within an owned interior, away from all peers."""
    part=scene['parts'][surface_key];surface=polygon(part)
    if surface is None:raise ValueError('Fallback requires an existing valid closed surface')
    width=styles.get(surface_key,{}).get('width_px') or 0
    safe=surface.buffer(-max(1.25,width/2+1.))
    protected=[];replacement=set(replace_keys)
    for key,p in scene['parts'].items():
        if key==surface_key or key in replacement:continue
        s=styles.get(key,{});poly=polygon(p)
        if poly is not None:
            # A containing background is behind the owner, not an occluding peer.
            if poly.covers(surface):continue
            protected.append(poly.buffer(max(1.5,(s.get('width_px') or 0)/2+1.)))
        elif s.get('stroke'):
            protected.extend(LineString(line).buffer(max(2.,(s.get('width_px') or 1)/2+1.5))
                             for line in strokes(p.get('commands',[])) if len(line)>1)
    for r in arrows:
        from .text_occlusion import arrow_region
        protected.append(arrow_region(r))
    for r in [*texts,*images]:
        b=r.get('bbox_2d') or r.get('source_box_half_open')
        if b:protected.append(rectangle(*b).buffer(1.5))
    return safe.difference(unary_union(protected)) if protected else safe


def make_asset(scene,styles,arrows,texts,source,folder,decision,*,images=()):
    from shape_integration.common import save,sha
    from shape_integration.styling.geometry import raster
    key=decision['surface_part'];owner=scene['parts'][key]['object_id']
    replace=decision['replace_parts'];surface=polygon(scene['parts'][key])
    for k in replace:
        p=scene['parts'].get(k,{})
        if p.get('object_id')!=owner or polygon(p) is not None or k==key:
            raise ValueError('Only same-owner open intrinsic texture parts may be replaced')
        lines=strokes(p.get('commands',[]))
        if not lines or not all(surface.buffer(1).covers(LineString(line)) for line in lines):
            raise ValueError('Replacement extends outside the owned surface')
    safe=safe_surface(scene,styles,arrows,texts,key,replace,images)
    if safe.is_empty or safe.area<max(8.,.12*surface.area):return None
    bounds=surface.bounds
    with Image.open(source) as im:
        clean=im.convert('RGBA')
        b=[max(0,int(np.floor(bounds[0]))),max(0,int(np.floor(bounds[1]))),
           min(clean.width,int(np.ceil(bounds[2]))+1),min(clean.height,int(np.ceil(bounds[3]))+1)]
        mask=raster(safe,b);patch=clean.crop(b)
        if mask.sum()<8:return None
        patch.putalpha(Image.fromarray((mask*255).astype('uint8')))
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'texture.png';patch.save(path)
    record={'id':'F'+hashlib.sha256(key.encode()).hexdigest()[:12],
            'category':'failed_complex_texture_fill','target':owner,'path':str(path.resolve()),
            'sha256':sha(path),'source_sha256':sha(source),'source_box_half_open':b,
            'placement_source_half_open':b,'raster_size':list(patch.size),
            'extraction_method':'source_pixels_protected_surface_alpha','surface_part':key,
            'replace_parts':replace,'protected_alpha_pixels':int(mask.size-mask.sum()),
            'visible_pixels':int(mask.sum()),'reason':decision['reason']}
    save(folder/'asset.json',record);return record


def review(*args, **kwargs):
    """Compatibility entry: the unified planner owns registry and texture assets."""
    from .image_assets import review as unified_review
    return unified_review(*args, **kwargs)
