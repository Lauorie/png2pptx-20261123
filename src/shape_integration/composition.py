"""Place source-bound image assets immediately below native text/formulas.

No inference, image editing, re-OCR or geometry modification happens here.
Image registries are resolved through saved run provenance, never by image ID.
"""
import hashlib
import math
import re
import shutil
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation

from .common import read, save, sha, compare
from .styling.export import render_native

PROTOCOL='shape_integration.complete.1'
SOURCE_KEYS=('localization_source','geometry_source','style_source','base_source',
             'arrow_source','recognition_source','composition_source')


def resolve_registry(workspace, run, item):
    """Prefer the case's own registry, then follow its declared source lineage."""
    workspace=Path(workspace);queue=[Path(run).resolve()];seen=set();chain=[]
    while queue:
        root=queue.pop(0)
        if root in seen:continue
        seen.add(root)
        if len(seen)>40:raise ValueError('Image source provenance chain too long')
        manifest_path=root/'manifest.json'
        if not manifest_path.is_file():continue
        manifest=read(manifest_path);old=next((v for v in manifest.get('images',[]) if v['id']==item['id']),None)
        if old is None:continue
        if old.get('source_sha256')!=item['source_sha256'] or old.get('source_size')!=item['source_size']:
            raise ValueError('Image source provenance differs: '+str(root))
        chain.append({'path':str(manifest_path),'sha256':sha(manifest_path)})
        for relative in ('dedup/image_boxes_original.json','images/inventory.json'):
            path=root/'cases'/item['id']/relative
            if path.is_file():return path,chain
        for key in SOURCE_KEYS:
            value=manifest.get(key)
            if isinstance(value,dict):value=value.get('path')
            if not isinstance(value,str) or not value:continue
            p=Path(value)
            choices=[p] if p.is_absolute() else [root/p,workspace/p]
            found=next((p.resolve() for p in choices if (p/'manifest.json').is_file()),None)
            if found:queue.append(found)
    raise ValueError('No image registry in source lineage for '+item['id'])


def base_ppt(case, *, with_text=True, with_images=True):
    paths=[]
    if with_text and with_images:paths.append('complete/combined.pptx')
    if with_text:paths.append('texts/combined.pptx')
    if with_images:paths.append('image_composition/combined.pptx')
    paths+=['arrows/combined.pptx','styling/styled.pptx','arrows/arrows.pptx','fitting/geometry_final.pptx']
    if with_text:paths.append('texts/text.pptx')
    return next((Path(case)/p for p in paths if (Path(case)/p).is_file()),None)


def valid_rect(box,size,*,integer=False):
    return (isinstance(box,(list,tuple)) and len(box)==4
            and all(type(x) in ((int,) if integer else (int,float)) and math.isfinite(x) for x in box)
            and 0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1])


def crop_box(asset, inventory, size):
    box=asset.get('source_box_half_open')
    if box is None:
        if 'bbox_2d' in asset:
            b=asset['bbox_2d']
            # Pixel-center transport can put an edge up to half a pixel
            # outside the source. Never fall back to a differently scaled
            # detector bbox merely because of that rounding convention.
            if (not isinstance(b,list) or len(b)!=4 or
                    any(type(v) not in (int,float) or not math.isfinite(v) or
                        v<-.5 or v>size[i%2]+.5 for i,v in enumerate(b))):
                raise ValueError('Invalid source image bbox')
            b=[max(0,min(v,size[i%2])) for i,v in enumerate(b)]
            if not valid_rect(b,size):raise ValueError('Invalid source image bbox')
            box=[math.floor(b[0]),math.floor(b[1]),min(size[0],math.ceil(b[2])+1),min(size[1],math.ceil(b[3])+1)]
        else:
            from png2pptx.images.detect import source_box
            if not valid_rect(asset.get('bbox'),inventory.get('input_size',size)):
                message='Invalid source image bbox' if 'bbox_2d' in asset else 'Invalid image inventory bbox'
                raise ValueError(message)
            box=source_box(asset['bbox'],inventory.get('input_size',size),size)
    if not valid_rect(box,size,integer=True):raise ValueError('Invalid source_box_half_open for '+asset['id'])
    return box


def extract(workspace,item,registry,folder):
    """Use matching exported RGBA when present, otherwise exact original pixels."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=Path(workspace)/item['original'];inventory=read(registry)
    if sha(source)!=item['source_sha256'] or inventory.get('source_sha256')!=item['source_sha256']:
        raise ValueError('Raster source hash differs from image inventory')
    if not isinstance(inventory.get('assets'),list):raise ValueError('Image registry needs an assets array')
    ids=[a['id'] for a in inventory['assets']]
    if len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not re.fullmatch(r'I\d+',i) for i in ids):
        raise ValueError('Image IDs must be unique I-number identifiers')
    source_case=Path(registry).parent.parent
    records=[]
    with Image.open(source) as opened:
        image=opened.convert('RGBA')
        if list(image.size)!=item['source_size']:raise ValueError('Raster source size differs')
        for asset in inventory['assets']:
            ident=asset['id'];box=crop_box(asset,inventory,image.size)
            local=folder/'assets'/ident;local.mkdir(parents=True,exist_ok=True)
            crop=image.crop(box);crop.save(local/'source_crop.png')
            result_path=source_case/'images/assets'/ident/'result.json'
            previous=read(result_path) if result_path.is_file() else asset
            exported=previous.get('rgba');method='source_crop';placement=box;notes=[];rgba_source=None
            if previous.get('export_enabled'):
                # Identity includes the original crop, not merely a reused I ID.
                bound_sha=previous.get('source_sha256')
                stage_path=source_case/'images/image_stage.json'
                if bound_sha is None and stage_path.is_file():bound_sha=read(stage_path).get('image',{}).get('source_sha256')
                if bound_sha!=item['source_sha256'] or previous.get('source_box_half_open')!=box:
                    raise ValueError('Exported RGBA is bound to a different image crop: '+ident)
                if not exported:raise ValueError('Exported RGBA path missing: '+ident)
                rgba_source=Path(exported)
                if not rgba_source.is_absolute():rgba_source=result_path.parent/rgba_source
                if not rgba_source.is_file() or sha(rgba_source)!=previous.get('rgba_sha256'):
                    raise ValueError('Exported RGBA missing or changed: '+ident)
                placement=previous.get('placement_source_half_open')
                if not valid_rect(placement,image.size):raise ValueError('Invalid exported RGBA placement: '+ident)
                shutil.copy2(rgba_source,local/'image.png');method='existing_rgba'
                notes.extend(previous.get('quality_flags',[]))
            else:
                crop.save(local/'image.png')
                if asset.get('background')=='alpha':notes.append('alpha_unavailable_original_rectangle_preserved')
            with Image.open(local/'image.png') as check:
                if check.convert('RGBA').getchannel('A').getbbox() is None:raise ValueError('Empty image asset: '+ident)
                raster_size=list(check.size)
            boundary=asset.get('boundary_review',{})
            row={'id':ident,'category':asset.get('category'),'target':asset.get('target'),
                 'keep':asset.get('keep'),'remove':asset.get('remove'),'background':asset.get('background'),
                 'path':str((local/'image.png').resolve()),'sha256':sha(local/'image.png'),
                 'source_box_half_open':box,'placement_source_half_open':placement,'raster_size':raster_size,
                 'extraction_method':method,'boundary_status':boundary.get('status','unknown'),
                 'boundary_reason':boundary.get('reason'),'quality_flags':notes,
                 'source_registry':str(Path(registry).resolve()),'source_registry_sha256':sha(registry),
                 'source_sha256':item['source_sha256'],'rgba_source':str(rgba_source) if rgba_source else None,
                 'extraction_record':str(result_path) if rgba_source else None,
                 'extraction_record_sha256':sha(result_path) if rgba_source and result_path.is_file() else None,
                 'visual_accuracy_verified':False}
            save(local/'record.json',row);records.append(row)
    save(folder/'assets.json',{'protocol':PROTOCOL,'source_sha256':item['source_sha256'],
                              'source_size':item['source_size'],'assets':records})
    return records


def is_text(shape):
    # Reading shape.text can create a missing txBody in an imported native
    # arrow. Classification must be read-only, including inside groups.
    if any((n.text or '').strip() for n in shape._element.iter(
            '{http://schemas.openxmlformats.org/drawingml/2006/main}t')):return True
    return bool(shape._element.xpath('.//*[local-name()="oMath" or local-name()="oMathPara"]'))


def inject(assets,size,base,path):
    base=Path(base);path=Path(path)
    if base.resolve()==path.resolve():raise ValueError('Composition output must not replace the base')
    prs=Presentation(str(base))
    if len(prs.slides)!=1 or abs(prs.slide_width/prs.slide_height-size[0]/size[1])>1e-4:
        raise ValueError('Composition requires one slide matching source aspect ratio')
    slide=prs.slides[0];original=list(slide.shapes);before=[etree.tostring(s._element) for s in original]
    scale=prs.slide_width/size[0];tree=slide.shapes._spTree
    first_text=next((s._element for s in original if is_text(s)),None)
    names={s.name:s for s in original};inserted=[];reused=[]
    for asset in assets:
        name='IMG '+asset['id']
        if name in names:
            existing=names[name]
            b=asset['placement_source_half_open'];expected=[round(b[0]*scale),round(b[1]*scale),round((b[2]-b[0])*scale),round((b[3]-b[1])*scale)]
            if existing.shape_type!=13 or hashlib.sha256(existing.image.blob).hexdigest()!=asset['sha256'] or [existing.left,existing.top,existing.width,existing.height]!=expected:
                raise ValueError('Existing image identity/placement conflict: '+name)
            reused.append(name);continue
        b=asset['placement_source_half_open']
        picture=slide.shapes.add_picture(asset['path'],round(b[0]*scale),round(b[1]*scale),
                                        round((b[2]-b[0])*scale),round((b[3]-b[1])*scale))
        picture.name=name
        import json
        picture._element.nvPicPr.cNvPr.set('descr',json.dumps({'image_asset':asset['id'],
            'source_sha256':asset['source_sha256'],'asset_sha256':asset['sha256'],
            'method':asset['extraction_method'],'placement_source_half_open':b},ensure_ascii=False))
        if first_text is not None:
            tree.remove(picture._element);tree.insert(tree.index(first_text),picture._element)
        inserted.append(picture._element)
    if before!=[etree.tostring(s._element) for s in slide.shapes if s._element not in inserted]:
        raise ValueError('Composition modified an existing vector/text object')
    picture_indices=[i for i,s in enumerate(slide.shapes) if s.name in {'IMG '+a['id'] for a in assets}]
    text_indices=[i for i,s in enumerate(slide.shapes) if is_text(s)]
    if picture_indices and text_indices and max(picture_indices)>=min(text_indices):
        raise ValueError('Images must be below every text/formula object')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.stem+'.partial.pptx');prs.save(temp);temp.replace(path)
    # Check again after PPT serialization, including all original shape XML.
    loaded=Presentation(str(path));new_names={'IMG '+a['id'] for a in assets if 'IMG '+a['id'] not in names}
    if before!=[etree.tostring(s._element) for s in loaded.slides[0].shapes if s.name not in new_names]:
        raise ValueError('Saving composition altered a base object')
    return {'images':len(assets),'inserted_images':len(inserted),'reused_images':len(reused),
            'base_objects':len(original),'base_objects_unchanged':True,'images_below_text':True,
            'image_layer_indices':picture_indices,'text_layer_indices':text_indices,
            'editable_textboxes':sum(s.name.startswith('TXT ') for s in loaded.slides[0].shapes)}


def compose_case(workspace,folder,item,base,registry,slots,*,render=True,timeout=180,text_scene=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    assets=extract(workspace,item,registry,folder)
    counts=inject(assets,item['source_size'],base,folder/'combined.pptx')
    overlaps=[]
    if text_scene:
        for text in read(text_scene)['texts']:
            b=text['bbox_2d'];area=max(1,(b[2]-b[0])*(b[3]-b[1]))
            for asset in assets:
                a=asset['source_box_half_open']
                fraction=max(0,min(b[2],a[2])-max(b[0],a[0]))*max(0,min(b[3],a[3])-max(b[1],a[1]))/area
                if fraction>=.9:overlaps.append({'text_id':text['id'],'image_id':asset['id'],'text':text['text'],
                    'text_bbox_covered_fraction':fraction,'action':'retained_above_image','status':'possible_raster_text_duplicate'})
    scene={'protocol':PROTOCOL,'case_id':item['id'],'source_size':item['source_size'],
           'source_sha256':item['source_sha256'],'base_pptx':str(Path(base).resolve()),'base_sha256':sha(base),
           'image_registry':str(Path(registry).resolve()),'image_registry_sha256':sha(registry),
           'images':assets,'counts':counts,'text_scene':str(text_scene) if text_scene else None,
           'layers_back_to_front':['existing_shapes_and_arrows','raster_images','native_text_and_formulas'],
           'text_image_overlaps':overlaps,'model_calls':0,'visual_accuracy_verified':False}
    save(folder/'scene.json',scene);save(folder/'text_image_overlaps.json',overlaps)
    if render:
        with slots:render_native(folder/'combined.pptx',folder/'native_render',item['source_size'],timeout)
        compare([Path(workspace)/item['original'],folder/'native.png'],['Source','Complete PPT'],folder/'comparison.png',width_cap=1400)
    unresolved=[a['id'] for a in assets if a['boundary_status'] not in ('model_review_complete','accepted')]
    state='completed_with_unresolved' if unresolved or any(a['quality_flags'] for a in assets) or overlaps else 'completed'
    result={'state':state,**counts,'unresolved_image_boundaries':unresolved,
            'opaque_alpha_fallbacks':[a['id'] for a in assets if 'alpha_unavailable_original_rectangle_preserved' in a['quality_flags']],
            'possible_raster_text_duplicates':len(overlaps),'model_calls':0,'visual_accuracy_verified':False}
    save(folder/'summary.json',result)
    (folder/'REPORT.zh-CN.md').write_text('\n'.join(['# 完整组合','',
        f'图片 {len(assets)} 个，原生文字框 {counts["editable_textboxes"]} 个。图片置于公式/文字下方；原有对象保持不变。',
        '', '[完整 PPT](combined.pptx) · [图层与来源](scene.json) · [图片清单](assets.json) · [文字重叠候选](text_image_overlaps.json)',
        '', '![原图与完整 PPT](comparison.png)' if render else '',
        '', '图片边界未决、原样矩形裁剪的背景、文字重复候选仍需视觉检查；流程完成不是视觉准确率。','']))
    return result


def append(workspace,folder,item,summary,config,slots):
    """Integrated hook before OCR, so retained picture annotations reach text OCR."""
    registry=Path(folder)/'dedup/image_boxes_original.json'
    if not registry.exists():registry=resolve_registry(workspace,Path(folder).parents[1],item)[0]
    base=base_ppt(folder,with_text=False,with_images=False)
    if base is None:raise ValueError('No vector base for image composition')
    result=compose_case(workspace,Path(folder)/'image_composition',item,base,registry,slots,render=False)
    out={**summary,'image_composition':result}
    if result['state']!='completed' and out['state']=='completed':out['state']='completed_with_unresolved'
    save(Path(folder)/'fitting/summary.json',out);return out
