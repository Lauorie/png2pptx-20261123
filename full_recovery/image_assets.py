"""One owner, one visual decision, one raster commit (including old registries)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import box, LineString
from shapely.ops import unary_union

from .handoff import json_part_id, strokes, arrow_strokes
from .texture_fallback import PATTERNS, polygon, texture_candidates
from png2pptx.images.ownership import IMAGE_OWNERSHIP

ASSET_SCOPE_PROTOCOL = 'visual_unit.19'


def clip_source_edge(path, placement, canvas, source_box):
    """Keep the original slide clipping, without stretching edited pixels."""
    b=np.asarray(placement,float);size=np.asarray(canvas,float)
    outside=[b[0]<0,b[1]<0,b[2]>size[0],b[3]>size[1]]
    if not any(outside):return list(placement)
    allowed=[source_box[0]==0,source_box[1]==0,source_box[2]==canvas[0],source_box[3]==canvas[1]]
    if any(o and not a for o,a in zip(outside,allowed)):raise ValueError('Edited silhouette leaves source target/slide')
    with Image.open(path) as im:
        scale=(b[2:]-b[:2])/np.asarray(im.size)
        lo=np.maximum(0,np.ceil(-b[:2]/scale)).astype(int)
        hi=np.minimum(im.size,np.floor((size-b[:2])/scale)).astype(int)
        if np.any(hi<=lo):raise ValueError('Edited target lies outside slide')
        cropped=im.crop((*lo,*hi))
    cropped.save(path)
    return [*(b[:2]+lo*scale),*(b[:2]+hi*scale)]


def clip_owned_crop(rgba, source_box, scene, styles, keys, *, owned_box=None):
    """Exclude crop padding beyond an existing complete enclosure.

    The crop decision retains source pixels inside an already fitted frame;
    nearby incoming arrowheads must not be baked into its padded rectangle.
    The enclosure must also cover the source-owned target, which can contain
    missing native parts. The observation crop may include external padding.
    Edited silhouettes and targets without a complete enclosure stay unchanged.
    """
    paths=[p for k in keys for p in strokes(scene['parts'][k].get('commands',[])) if len(p)>1]
    if not paths:return rgba,None
    points=np.concatenate(paths);lower=points.min(0);upper=points.max(0);options=[]
    for k in keys:
        surface=polygon(scene['parts'][k])
        if surface is None:continue
        bounds=surface.bounds
        margin=max(1.,(styles.get(k,{}).get('width_px') or 0)/2+1)
        if owned_box is not None:
            lo=np.asarray(bounds[:2])-margin;hi=np.asarray(bounds[2:])+margin
            if np.any(np.asarray(owned_box[:2])<lo) or np.any(np.asarray(owned_box[2:])>hi):continue
        if np.any(np.asarray(bounds[:2])>lower+1) or np.any(np.asarray(bounds[2:])<upper-1):continue
        if all(surface.buffer(1).covers(LineString(p)) for p in paths):options.append((surface.area,k,surface))
    if not options:return rgba,None
    _,key,surface=min(options,key=lambda v:v[0])
    boundary=surface.buffer(max(1.,(styles.get(key,{}).get('width_px') or 0)/2+1))
    mask=Image.new('L',rgba.size,0);draw=ImageDraw.Draw(mask)
    x,y=source_box[:2];sx=rgba.width/(source_box[2]-x);sy=rgba.height/(source_box[3]-y)
    draw.polygon([((px-x)*sx,(py-y)*sy) for px,py in boundary.exterior.coords],fill=255)
    alpha=np.minimum(np.asarray(rgba.getchannel('A')),np.asarray(mask))
    result=rgba.copy();result.putalpha(Image.fromarray(alpha));return result,key


def native_frames(scene, keys):
    """Keep the unit's outer enclosure, not every repeated cell inside it."""
    result=[]
    for key in keys:
        part=scene['parts'][key]
        if part.get('kind') not in ('rectangle','rounded_rectangle','ellipse') or not part.get('native_parameters_original'):continue
        surface=polygon(part)
        if surface is None:continue
        others=[k for k in keys if k!=key]
        def contained(k):
            other=scene['parts'][k];footprint=polygon(other)
            if footprint is None:
                paths=[LineString(p) for p in strokes(other.get('commands',[])) if len(p)>1]
                footprint=unary_union(paths) if paths else box(*other['bbox_2d'])
            return surface.covers(footprint)
        if others and all(contained(k) for k in others):result.append(key)
    return result


def clip_asset_scope(rgba, placement, target_box, scene, styles, frames=(), foreground=(), owned_parts=()):
    """Clip pixels, never resize the editor's object or repaint native borders."""
    region=box(*target_box)
    enclosed=[(key,polygon(scene['parts'][key])) for key in frames]
    enclosed=[(key,surface) for key,surface in enclosed if surface is not None]
    if owned_parts and enclosed and all(any(surface.covers(box(*scene['parts'][key]['bbox_2d']))
            for _,surface in enclosed) for key in owned_parts):
        # Several failed symbols inside separate native cells own those cell
        # interiors, not the grid/frames between their union bounding box.
        region=region.intersection(unary_union([surface.buffer(-max(1.,(styles.get(key,{}).get('width_px') or 0)/2+1))
            for key,surface in enclosed if any(surface.covers(box(*scene['parts'][part]['bbox_2d'])) for part in owned_parts)]))
    for key in frames:
        surface=polygon(scene['parts'][key])
        if surface is not None:
            margin=max(1.,(styles.get(key,{}).get('width_px') or 0)/2+1)
            if surface.buffer(1).covers(region):region=region.intersection(surface.buffer(-margin))
            else:region=region.difference(surface.boundary.buffer(margin))
    for key in foreground:
        surface=polygon(scene['parts'][key])
        if (surface is not None and isinstance(styles.get(key,{}).get('fill'),(str,dict))
                and styles.get(key,{}).get('fill_opacity',1)==1):
            region=region.difference(surface.buffer((styles.get(key,{}).get('width_px') or 0)/2))
        else:
            lines=[LineString(path) for path in strokes(scene['parts'][key].get('commands',[])) if len(path)>1]
            if lines:
                region=region.difference(unary_union(lines).buffer(
                    max(1.,(styles.get(key,{}).get('width_px') or 1)/2+1)))
    from shapely import contains_xy
    x=placement[0]+(np.arange(rgba.width)+.5)*(placement[2]-placement[0])/rgba.width
    y=placement[1]+(np.arange(rgba.height)+.5)*(placement[3]-placement[1])/rgba.height
    inside=contains_xy(region,x[None,:],y[:,None])
    result=rgba.copy();result.putalpha(Image.fromarray(np.where(inside,np.asarray(rgba.getchannel('A')),0).astype('uint8')))
    if result.getchannel('A').getbbox() is None:raise ValueError('No pixels remain inside the owned image scope')
    return result


def owner_matches(scene, images):
    """Match proposals to localized objects, never to a containing network."""
    matches={};unmatched=[]
    for image in images:
        a=box(*image['source_box_half_open']);scores=[]
        for oid,obj in scene['objects'].items():
            candidate=obj.get('candidate',{});b=candidate.get('bbox_2d')
            if not b:continue
            parts=[p for p in scene['parts'].values() if p.get('object_id')==oid]
            if not parts and obj.get('state') not in ('unresolved','rejected','failed'):continue
            # A containing frame cannot stand in for its registered interior.
            # Bbox overlap nominates an owner only when it actually has native
            # content inside that image's replacement footprint. Word labels
            # such as "network" or "icon" do not establish this relation.
            scoped=[box(*p['bbox_2d']) for p in parts if a.buffer(3).covers(box(*p['bbox_2d']))]
            if parts and not scoped:continue
            g=box(*b);score=a.intersection(g).area/max(a.union(g).area,1)
            # A registered icon can occupy only part of a compound card. Bind
            # it to that owner so it cannot be inserted again beside a whole-
            # card replacement. The review below limits replacement to its parts.
            if scoped:
                footprint=unary_union(scoped)
                score=max(score,a.intersection(footprint).area/max(a.union(footprint).area,1))
            if score>=.45:scores.append((score,oid))
        scores.sort(reverse=True)
        if scores and (len(scores)==1 or scores[0][0]-scores[1][0]>=.05):
            matches.setdefault(scores[0][1],[]).append(image)
        else:unmatched.append(image)
    return matches,unmatched


def remove_frame_copies(rgba, placement, source, scene, frames, replaced=()):
    """Remove measured source copies of retained native enclosure strokes."""
    from .source_enclosures import retained_rim, remove_retained_rim
    copies=[]
    for key in frames:
        if key in replaced:continue
        measured=retained_rim(source,scene['parts'][key])
        if measured is None:continue
        rgba,removed=remove_retained_rim(rgba,placement,measured)
        if removed['removed_nontransparent_pixels']:
            copies.append({'native_frame':key,**removed})
    return rgba,copies


def compact_unit(scene, oid):
    """Nominate bounded composite units without a description-word classifier."""
    obj=scene['objects'][oid].get('candidate',{});b=obj.get('bbox_2d')
    if not b or obj.get('kind')!='compound_symbol':return False
    return (b[2]>b[0] and b[3]>b[1]
            and (b[2]-b[0])*(b[3]-b[1])<=.025*np.prod(scene['source_size']))


def connected_owner(scene, oid, keys, arrows=()):
    """Use the geometry observation's roles, including delegated connections.

    Area nominates a review; it cannot turn a connected node group into a
    single image. A leaf illustration with no connections remains eligible.
    """
    from shape_integration.fitting.connectivity import fitted_connections
    parts={k:scene['parts'][k] for k in keys}
    owners={p.get('object_id') for p in parts.values()}|{oid}
    receipts=[scene['objects'].get(owner,{}).get('declared_ownership') for owner in owners if owner is not None]
    # The source roles distinguish a geometric solid's intrinsic edges from
    # a node-link network. Infer routes only for legacy owners without roles.
    if any(not r for r in receipts) and fitted_connections(parts)['routes']:return True
    receipts=[r for r in receipts if r]
    links=any(r in ('connection','directional_connection','directional_head')
              for receipt in receipts for r in receipt.get('retained_roles',{}).values())
    delegated=any(p.get('owner')=='arrow' for r in receipts for p in r.get('delegated_parts',[]))
    links=links or delegated and any(r.get('unit')=='neural_network' for r in receipts)
    # Missing delegated headless links still own the network's connections.
    # An absent arrow product cannot authorize rasterizing its native nodes.
    links=links or any(p.get('owner')=='arrow' and p.get('role')=='connection'
                       for r in receipts for p in r.get('delegated_parts',[]))
    from shape_integration.fitting.connectivity import node_catalog, node_pair
    nodes,boxes=node_catalog(parts)
    # Delegation can describe an external arrow leaving an image. Only a route
    # joining two nodes in this visual unit establishes internal connectivity.
    if not links and delegated:
        links=any(node_pair(path,nodes,boxes) for arrow in arrows
                  if arrow.get('beziers') or ('start' in arrow and 'end' in arrow)
                  for path in arrow_strokes(arrow))
    endpoints=nodes+boxes
    distinct=[]
    for key,center,radius in endpoints:
        if not any(np.linalg.norm(center-old)<2 for old in distinct):distinct.append(center)
    return bool(links and len(distinct)>1)


def matched_subowners(scene, matched, covered=(), protected_keys=()):
    """Nominate complete subowners inside an already matched image's bounds."""
    result={};claimed=set(covered)
    by_owner={oid:[k for k,p in scene['parts'].items() if p['object_id']==oid]
              for oid in scene['objects']}
    for oid,assets in matched.items():
        regions=[box(*a['source_box_half_open']) for a in assets
                 if a.get('category') in ('complex_icon','illustration')]
        for child,keys in by_owner.items():
            if child in matched or not keys or set(keys)&claimed:continue
            if any(all(region.covers(box(*scene['parts'][k]['bbox_2d'])) for k in keys) for region in regions):
                eligible=[k for k in keys if k not in protected_keys]
                result.setdefault(oid,[]).extend(eligible)
                claimed.update(eligible)
    return result


def registry_groups(assets, parts):
    """Review shared native replacement claims once, before generating pixels."""
    groups=[]
    for asset in assets:
        keys=set(parts.get(asset['id'],[]));members=[asset];remaining=[]
        for group,claimed in groups:
            if keys & claimed:
                members.extend(group);keys.update(claimed)
            else:remaining.append((group,claimed))
        groups=[*remaining,(members,keys)]
    # Recompute the leader's complete leaf set; the review already supports
    # several registered source regions belonging to one representation task.
    for members,keys in groups:
        if keys:parts[members[0]['id']]=sorted(keys)
    return [members for members,keys in groups]


def review(scene,styles,arrows,texts,source,native_preview,folder,runtime,*,images=(),selected=None,protected_keys=(),editor=None,cached_assets=(),defer_cleanup=False,preserve_icon_pixels=False):
    """The old registry is evidence for this decision, not a later insertion queue."""
    from shape_integration.common import save,sha
    from .image_edit import transparent_exterior
    # Failed fits remain in the source scene for geometry repair, but have no
    # native footprint to compare, replace or protect. Keep their owners and
    # localized source bounds: an empty/partial owner still needs review.
    unfitted={k:p for k,p in scene['parts'].items()
              if p.get('status','ok')!='ok' or p.get('bbox_2d') is None}
    if unfitted:
        scene={**scene,'parts':{k:p for k,p in scene['parts'].items() if k not in unfitted}}
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:clean=im.convert('RGB')
    with Image.open(native_preview) as im:native=im.convert('RGB')
    if native.size!=clean.size:raise ValueError('Asset review requires original coordinates')
    from .source_ellipsis import observe as observe_ellipsis,clear_image_copies
    native_ellipsis=observe_ellipsis(clean,texts,arrows)
    matched,unmatched=owner_matches(scene,images)
    ids=set(texture_candidates(scene,editor is not None) if selected is None else selected)
    ids.update(matched)
    ids.update(oid for oid in scene['objects'] if compact_unit(scene,oid))
    # A selected partial fit can retain a correct frame while omitting its
    # payload. Its existing completeness record nominates the same visual
    # decision; prose claiming delegation does not create an image task.
    incomplete={oid for oid,obj in scene['objects'].items()
                if obj.get('completeness_status')=='incomplete'}
    ids.update(incomplete)
    # A failed observation has no part to trigger texture/icon heuristics.
    # Its localized target still needs the same single representation decision.
    ids.update(oid for oid,obj in scene['objects'].items()
               if not obj.get('part_keys') and obj.get('state') in ('unresolved','rejected','failed')
               and (bounds:=obj.get('candidate',{}).get('bbox_2d'))
               and (bounds[2]-bounds[0])*(bounds[3]-bounds[1])<=.025*np.prod(scene['source_size']))
    # Image delegation still needs an asset task, including matched registry
    # images with no native parts. Only other branches bypass image realization.
    delegated_only={oid for oid,obj in scene['objects'].items()
                    if not obj.get('part_keys') and (receipt:=obj.get('declared_ownership'))
                    and receipt.get('delegated_parts') and not receipt.get('retained_roles')
                    and not any(p.get('owner')=='image' for p in receipt['delegated_parts'])}
    ids.difference_update(delegated_only)
    # A small registered illustration may span several native owners. Give
    # that single existing image task the exact enclosed parts, instead of
    # separately replacing its small sub-icon and then inserting the whole crop.
    registry_parts={}
    for a in unmatched:
        region=box(*a['source_box_half_open'])
        if a.get('category') not in ('complex_icon','illustration') or region.area>.025*np.prod(scene['source_size']):continue
        keys=[k for k,p in scene['parts'].items() if region.buffer(3).covers(box(*p['bbox_2d']))]
        keys=[k for k in keys if not any(box(*other['bbox_2d']).covers(box(*scene['parts'][k]['bbox_2d']))
              and not region.buffer(3).covers(box(*other['bbox_2d']))
              for other in scene['parts'].values() if other.get('native_parameters_original')
              and other.get('kind') in ('ellipse','rectangle','rounded_rectangle')
              and other is not scene['parts'][k])]
        if keys:registry_parts[a['id']]=keys
    covered={k for keys in registry_parts.values() for k in keys}
    # A matched image can contain subowners too. Include their complete native
    # footprint in the same representation decision, within the existing crop.
    matched_parts=matched_subowners(scene,matched,covered,protected_keys)
    covered.update(k for keys in matched_parts.values() for k in keys)
    ids={oid for oid in ids if not (owned:={k for k,p in scene['parts'].items() if p['object_id']==oid})
         or not owned-set(native_frames(scene,list(owned)))-set(protected_keys)<=covered}
    # No native competitor means there is no representation choice to repeat.
    # Preserve already registered photos/icons; decoration backgrounds still
    # need foreground removal and cannot take this pass-through path.
    def needs_cleanup(asset):
        region=box(*asset['source_box_half_open'])
        if any(t.get('bbox_2d') and region.intersection(box(*t['bbox_2d'])).area>4 for t in texts):return True
        return any(isinstance(styles.get(k,{}).get('fill'),str) and
                   region.intersection(box(*p['bbox_2d'])).area>4 and not box(*p['bbox_2d']).covers(region)
                   for k,p in scene['parts'].items())
    # A registered raster/illustration is already source evidence.  If its
    # crop contains neither native text nor a competing foreground surface,
    # retaining that exact crop is the only representation decision needed.
    # Sending it through the model can return keep_native and silently drop
    # the whole asset (particularly for small scientific panels).  Keep the
    # same cleanup and registry-scope gates used by icons; no asset identity
    # or case-specific rule is involved here.
    source_asset_categories={'natural_photo','complex_icon','illustration','scientific_raster'}
    preserved=[a for a in unmatched if (a.get('category') in source_asset_categories and
               not needs_cleanup(a)) and a['id'] not in registry_parts]
    tasks=[(oid,matched.get(oid,[])) for oid in sorted(ids)]
    tasks.extend((None,group) for group in registry_groups([a for a in unmatched if a not in preserved],registry_parts))

    def one(oid,registered):
        ident=oid or registered[0]['id'];local=folder/ident;local.mkdir(exist_ok=True)
        keys=[k for k,p in scene['parts'].items() if p['object_id']==oid and k not in covered] if oid else registry_parts.get(ident,[])
        keys=[*keys,*matched_parts.get(oid,[])]
        owner_frames=native_frames(scene,keys)
        if registered and keys:
            # Registry scope owns the icon, not its enclosing card/background.
            regions=[box(*a['source_box_half_open']).buffer(3) for a in registered]
            scoped=[k for k in keys if any(g.covers(box(*scene['parts'][k]['bbox_2d'])) for g in regions)]
            partial=bool(scoped and len(scoped)<len(keys))
            if partial:keys=scoped
        else:partial=False
        surfaces=[k for k in keys if polygon(scene['parts'][k]) is not None]
        frames=list(dict.fromkeys([*owner_frames,*native_frames(scene,keys)]))
        obj=scene['objects'].get(oid,{}).get('candidate',{})
        label=('; '.join(a.get('target','image') for a in registered)
               if registered else obj.get('label') or oid)
        boxes=[scene['parts'][k]['bbox_2d'] for k in keys]
        if obj.get('bbox_2d') and not partial:boxes.append(obj['bbox_2d'])
        boxes.extend(a['source_box_half_open'] for a in registered)
        if not boxes:return {'owner':oid,'action':'keep_native','reason':'No source bounds'}
        target_box=[max(0,min(x[0] for x in boxes)),max(0,min(x[1] for x in boxes)),
                    min(clean.width,max(x[2] for x in boxes)),min(clean.height,max(x[3] for x in boxes))]
        b=[max(0,int(min(x[0] for x in boxes))-4),max(0,int(min(x[1] for x in boxes))-4),
           min(clean.width,int(np.ceil(max(x[2] for x in boxes)))+5),
           min(clean.height,int(np.ceil(max(x[3] for x in boxes)))+5)]
        overlay=clean.copy();outline=ImageDraw.Draw(overlay)
        for key in surfaces:
            for points in strokes(scene['parts'][key].get('commands',[])):
                outline.line([tuple(p) for p in points],fill='#FF00BB',width=2)
        sheet=Image.new('RGB',(1200,430),'#edf0f3');draw=ImageDraw.Draw(sheet)
        for i,(title,im) in enumerate([('SOURCE',clean),('CURRENT NATIVE PPT',native),('SOURCE + FITTED OUTLINE',overlay)]):
            draw.text((400*i+8,8),title,fill='black')
            tile=ImageOps.contain(im.crop(b),(384,380),Image.Resampling.NEAREST)
            sheet.paste(tile,(400*i+(400-tile.width)//2,32+(380-tile.height)//2))
        sheet.save(local/'comparison.png')
        network=connected_owner(scene,oid,keys,arrows)
        allowed=(b[2]-b[0])*(b[3]-b[1])<=.65*clean.width*clean.height
        hint=[{k:a.get(k) for k in ('id','target','keep','remove','background','extraction_method',
                                   'boundary_status','quality_flags')} for a in registered]
        labels=[{'text':t.get('text'),'bbox':t['bbox_2d']} for t in texts
                if t.get('bbox_2d') and box(*b).intersection(box(*t['bbox_2d'])).area>0]
        contained=[]
        foreign=[{'key':k,'bbox':p['bbox_2d']} for k,p in scene['parts'].items()
                 if k not in keys and isinstance(styles.get(k,{}).get('fill'),str)
                 and p.get('kind') in ('ellipse','rectangle','rounded_rectangle')
                 and box(*target_box).intersection(box(*p['bbox_2d'])).area>4
                 and not box(*p['bbox_2d']).covers(box(*target_box))]
        for arrow in arrows:
            paths=arrow_strokes(arrow)
            if not paths:continue
            points=np.concatenate(paths)
            if np.all(points>=np.asarray(b[:2])) and np.all(points<=np.asarray(b[2:])):
                contained.append({'id':arrow['id'],'bbox':[*points.min(0),*points.max(0)],
                                  'heads':arrow.get('heads'),'color':arrow.get('style',{}).get('color')})
        aliases={f'P{i}':key for i,key in enumerate(keys)}
        short={key:alias for alias,key in aliases.items()}
        foreign_aliases={f'O{i}':part['key'] for i,part in enumerate(foreign)}
        missing=[{'id':p.get('id'),'kind':p.get('kind'),'error':p.get('error')}
                 for p in unfitted.values() if p.get('object_id')==oid]
        owner=scene['objects'].get(oid,{})
        ownership=owner.get('declared_ownership') or {}
        selected_observation=next((row.get('observation',{}) for row in owner.get('history',[])
                                   if row.get('directory')==owner.get('selected_revision')),{})
        prompt={'task':'对比 SOURCE 与当前原生PPT，仅对指定目标选择一种表示。截图登记是候选，不会自动再次插入。',
            'target':label,'source_box':b,'target_box':target_box,'existing_image_proposals':hint,'whole_owner_image_available':allowed and not network,'local_image_available':allowed,
            'editor_available':editor is not None,'separate_native_labels':labels,'contained_arrow_candidates':contained,
            'compact_visual_unit':bool(oid and compact_unit(scene,oid)),
            'connected_native_unit':network,
            'source_part_ownership':ownership,
            **({'unfitted_parts':missing} if missing else {}),
            **({'incomplete_native_target':{
                'observation_status':owner.get('observation_status'),
                'recorded_reason':selected_observation.get('reason',owner.get('reason','')),
                'rule':'拟合记录仅说明待核对的缺口，不构成截图理由。按统一source表示边界决定局部image，已移交的文字和箭头不属于shape缺口。'}}
               if oid in incomplete else {}),
            'overlapping_native_parts':[{**part,'key':alias} for alias,part in zip(foreign_aliases,foreign)],
            'rules':[
                IMAGE_OWNERSHIP,
                'parts是当前已拟合内容，source_part_ownership是同次原图观察的归属。compact_visual_unit只提名一次审查；粉色轮廓和旧截图登记都不能改变归属。connected_native_unit不能整组截图；复杂立体的固有棱边和顶点不因此成为连接网络。reason说明原图中确实需要位图的视觉内容，而非仅描述拟合失败。',
                'separate_native_labels会由文字分支加入。不得仅因原生预览缺少这些文字而转图片，也不得把这些文字再画进图片。',
                '截图登记中的keep/remove/background是旧提案，不是处理结果；与本次separate_native_labels或protected归属冲突时，以本次归属为准。source_crop仍是原始矩形；crop只去相连白底，需要清理覆盖物时选edit。',
                'keep_native原样保留当前parts，不会再补画。按当前可见结果选择表示，不能以理论上可原生重建为由丢弃尚未实现的图像内容。规则细密的双色纹理选 native_pattern，保留边界，只列出被替代的同面纹理线。',
                '有机曲线轮廓（如云形）若已明显变为折线或丢失凹凸，可将整个非文字轮廓作为同一复杂单元编辑；不要只编辑内部图标却无条件保留失真的外轮廓。普通矩形边框、有限网格仍保留可编辑几何。',
                'replace_parts列出被图片完整替代的P编号。保留正确的外框和文字；内部图案的圆圈、符号应作为同一视觉单元处理，不能混用错位的原生圆圈和截图弧线。protected外框保留。edit_box仅在明确只修复局部时填写，坐标用原图像素。',
                'source_box扩边只用于观察。overlapping_native_parts仅是相交候选，不代表前景。foreground_parts只列原图中确实挡在目标前面的O编号；后方底板/卡片不要列入。crop中混入的背景、文字或外部连线需要edit清理。',
                '局部编辑会提供完整目标作参照，仅提交edit_box内像素；其余登记像素保留。编辑提示词只能修复该区域，不要把整个目标重画进局部。',
                'role=background 保留后方背景并清理前景；role=icon 保留获准的复杂非文字几何；role=texture 保留纹理/透视。三种角色均不能接收明显箭头。编辑描述与已登记的原生文字、边框、节点和箭头归属一致。',
                'replace_arrows仅用于真实图片内不可分的图像内容，不得接收独立连接或明显箭头；按source_part_ownership移交的箭头继续使用原生产物，replace_arrows为空。需要移除外部覆盖物时选edit；crop不能完成此类清理。',
                '仅method=edit填写edit_target和remove_overlays。edit_target只描述非文字目标的外观；remove_overlays只列外部覆盖物。文字移除、角色和坐标由程序统一加入，不要重复写指令。清理截图时保留原画法；只有局部错误时用最小edit_box。已有原生对象且证据不足时选 keep_native。'],
            'parts':[{'key':short[k],'kind':scene['parts'][k]['kind'],'closed_surface':k in surfaces,
                      'source_role':ownership.get('retained_roles',{}).get(json_part_id(k),'uncertain'),
                      'bbox':scene['parts'][k]['bbox_2d'],'protected':k in protected_keys or k in frames} for k in keys],
            'patterns':list(PATTERNS),
            'output':{'action':'keep_native|native_pattern|image','reason':'可见依据',
                'role':'background|texture|icon','method':'crop|edit','edit_target':'仅edit填写：非文字目标的外观',
                'remove_overlays':['仅edit填写：待移除的外部覆盖物；不要重复列原生文字'],
                'replace_arrows':['仅不可分的真实图像内容；独立或明显箭头使用空数组'] if contained else [],
                'foreground_parts':['原图中挡在目标前面的O编号；没有则[]'],
                'edit_box':'optional [x0,y0,x1,y1] inside target_box; only the failed interior',
                'surface_part':'native_pattern: exact closed key','replace_parts':['exact keys replaced by image or native_pattern'],
                'pattern':'native_pattern: listed preset','foreground':'#RRGGBB','background':'#RRGGBB'}}

        def image_scope(d):
            if d.get('edit_box'):return d['edit_box']
            replace=d['replace_parts']
            if replace and set(replace)!=set(keys):
                bounds=[scene['parts'][k]['bbox_2d'] for k in replace]
                return [min(v[0] for v in bounds),min(v[1] for v in bounds),max(v[2] for v in bounds),max(v[3] for v in bounds)]
            return target_box

        def pixel_box(scope):
            return [max(0,int(np.floor(scope[0]))),max(0,int(np.floor(scope[1]))),
                    min(clean.width,int(np.ceil(scope[2]))),min(clean.height,int(np.ceil(scope[3])))]

        def validate(d):
            if not isinstance(d,dict) or d.get('action') not in ('keep_native','native_pattern','image') or not d.get('reason'):
                raise ValueError('Choose one action and visible reason')
            if not keys and d['action']=='native_pattern':raise ValueError('Native pattern requires an existing surface')
            if isinstance(d.get('replace_parts'),list):
                if not all(isinstance(k,str) for k in d['replace_parts']):raise ValueError('replace_parts must contain part IDs')
                d['replace_parts']=[aliases.get(k,k) for k in d['replace_parts']]
            if isinstance(d.get('surface_part'),str):d['surface_part']=aliases.get(d['surface_part'],d['surface_part'])
            if d['action']=='image':
                if not allowed:raise ValueError('This owner must stay native')
                if d.get('role') not in ('background','texture','icon') or d.get('method') not in ('crop','edit'):
                    raise ValueError('Choose image role and method')
                if d['method']=='edit' and editor is None:raise ValueError('Editor unavailable')
                if d['method']=='edit':
                    if not isinstance(d.get('edit_target'),str) or not d['edit_target'].strip():
                        raise ValueError('Describe the non-text edit_target')
                    if not isinstance(d.get('remove_overlays',[]),list) or not all(isinstance(v,str) for v in d.get('remove_overlays',[])):
                        raise ValueError('remove_overlays must list external overlays')
                owned=d.get('replace_arrows',[] if not contained else None)
                if not isinstance(owned,list) or any(not isinstance(v,str) for v in owned) or not set(owned)<={a['id'] for a in contained}:
                    raise ValueError('image requires replace_arrows from contained candidates (or explicit [])')
                if d['role']=='background' and owned:raise ValueError('Background cannot own foreground arrows')
                if owned and any(p.get('owner')=='arrow' for p in ownership.get('delegated_parts',[])):
                    raise ValueError('Source-delegated arrows retain native ownership; image cannot reclaim them')
                d['replace_arrows']=list(dict.fromkeys(owned))
                replace=d.get('replace_parts',[] if not keys else None)
                if not isinstance(replace,list) or not set(replace)<=set(keys)-set(frames)-set(protected_keys):
                    raise ValueError('Image replace_parts must name only failed, unprotected parts')
                d['replace_parts']=replace
                if network:
                    if d.get('role')=='icon' or d['replace_arrows']:
                        raise ValueError('Connected units keep editable nodes and native connections; only local background or texture can be imaged')
                    if not replace or set(replace)==set(keys) or any(
                            scene['parts'][k].get('kind') in ('ellipse','rectangle','rounded_rectangle')
                            and k not in owner_frames for k in replace):
                        raise ValueError('Local image must replace only the background/texture part, retaining node bodies and connections')
                foreground=d.get('foreground_parts',[])
                if not isinstance(foreground,list) or not all(isinstance(k,str) for k in foreground):
                    raise ValueError('foreground_parts must be a list of overlapping part IDs')
                foreground=[foreign_aliases.get(k,k) for k in foreground]
                if not set(foreground)<={p['key'] for p in foreign}:raise ValueError('Unknown foreground part')
                d['foreground_parts']=foreground
                requested=d.get('edit_box')
                if requested == '' or requested == []:
                    d.pop('edit_box',None);requested=None
                # The full observation crop is context, not a requested local
                # repair. Accept that exact full-crop spelling as whole-target
                # editing; the committed pixels still use target_box.
                if d.get('method')=='edit' and requested==b:
                    d.pop('edit_box',None);requested=None
                if requested is not None:
                    if not isinstance(requested,list) or len(requested)!=4 or not all(isinstance(x,(int,float)) and np.isfinite(x) for x in requested):
                        raise ValueError('edit_box must be four finite source coordinates')
                    if requested[2]<=requested[0] or requested[3]<=requested[1] or not box(*target_box).buffer(1).covers(box(*requested)):
                        raise ValueError('edit_box leaves target_box. For whole-target editing omit edit_box; source_box is observation context only.')
                replacement_scope=box(*image_scope(d)).buffer(1)
                if any(not replacement_scope.covers(box(*scene['parts'][k]['bbox_2d'])) for k in replace):
                    raise ValueError('Image scope does not cover complete replace_parts. Expand edit_box to cover those parts, or retain the parts outside the edit region.')
                if requested and not keys and registered and d['method']=='edit':
                    # Retained source pixels must not reintroduce native labels.
                    # Their underlying pigment belongs to the editor, not a
                    # later transparent rectangle cut from the restored icon.
                    if any(not replacement_scope.covers(box(*t['bbox']).intersection(box(*target_box)))
                           for t in labels if box(*t['bbox']).intersection(box(*target_box)).area>0):
                        raise ValueError('Local edit must include native labels inside the retained target. Expand edit_box to remove those labels and restore their underlying background; do not erase label rectangles.')
                existing=len(registered)==1 and registered[0].get('extraction_method')=='existing_rgba'
                # Observation padding is not retained image ownership. Labels
                # outside the final clip cannot require generative removal.
                retained_labels=[label for label in labels
                                 if box(*image_scope(d)).intersection(box(*label['bbox'])).area>0]
                if d['method']=='crop' and retained_labels and not existing:
                    # Check feasibility in the existing decision validator so
                    # its normal correction can select edit. Do not accept an
                    # impossible removal request and discard the asset later.
                    bounds=pixel_box(image_scope(d))
                    alpha=np.asarray(transparent_exterior(clean.crop(bounds),tolerance=0,adaptive=False).getchannel('A'))
                    for text in labels:
                        g=box(*text['bbox']).intersection(box(*bounds))
                        if g.is_empty or g.area<4:continue
                        x0,y0,x1,y1=g.bounds
                        mask=alpha[max(0,int(y0-bounds[1])):int(np.ceil(y1-bounds[1])),
                                   max(0,int(x0-bounds[0])):int(np.ceil(x1-bounds[0]))]
                        if mask.size and np.mean(mask>0)>.1:
                            raise ValueError('Crop contains separate native text; choose edit to remove the label while preserving the target, or keep_native when available.')
            if d['action']=='native_pattern':
                key=d.get('surface_part');replace=d.get('replace_parts')
                if key not in surfaces or not isinstance(replace,list) or not set(replace)<=set(keys)-set(surfaces)-set(protected_keys):
                    raise ValueError('Pattern needs an owned surface and only unprotected open hatch parts')
                surface=polygon(scene['parts'][key])
                for k in replace:
                    lines=strokes(scene['parts'][k].get('commands',[]))
                    if not lines or not all(surface.buffer(1).covers(LineString(p)) for p in lines):raise ValueError('Hatch leaves surface')
                if d.get('pattern') not in PATTERNS or any(not re.fullmatch('#[0-9a-fA-F]{6}',str(d.get(k,''))) for k in ('foreground','background')):
                    raise ValueError('Use a native pattern and RGB colors')
                d['shape_name']=oid+'/'+json_part_id(key)
                d['replace_names']=[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in replace]
            return d

        if preserve_icon_pixels and registered and not keys and not foreign:
            # There is no native alternative to vote on. The registered
            # original supplies the target pixels; source ownership later
            # decides external lines, and only explicit labels are cleaned.
            d=validate({'action':'image','method':'edit','role':'icon',
                'reason':'Registered source image has no native competitor or foreign foreground',
                'replace_parts':[],'replace_arrows':[],'foreground_parts':[],
                'edit_target':registered[0].get('keep') or registered[0].get('target') or '原始图形',
                'remove_overlays':[]})
            d['model_call_skipped']='no_representation_choice'
        else:
            d=runtime.call(local/'calls','review','只根据给定图像做一次选择，返回JSON。',prompt,
                           [('Source and current native',local/'comparison.png')],validate,8192)
        if not keys and d['action']=='keep_native':
            d.update(action='unresolved',unresolved=True)
        d.update(owner=oid,registry_ids=[a['id'] for a in registered])
        save(local/'decision.json',d)
        if d['action']=='image':
            asset_box=image_scope(d)
            all_keys=keys;keys=d['replace_parts']
            # Sampling bounds are integer; ownership clipping retains the exact
            # original target, including targets without a native enclosure.
            b=pixel_box(asset_box)
            edit_context=pixel_box(target_box) if d['method']=='edit' and d.get('edit_box') and registered and not all_keys else b
            edit_request=None
            source_enclosure=None
            if registered and not all_keys and len(registered)==1 and d['role']=='icon' and d['method']=='edit':
                from .source_enclosures import ellipse as source_ellipse
                source_enclosure=source_ellipse(clean,asset_box)
            if d['method']=='edit':
                local_box=lambda bounds:[bounds[i]-edit_context[i%2] for i in range(4)]
                edit_request={'target':d['edit_target'],'remove_overlays':d.get('remove_overlays',[]),
                    'native_labels':[{'text':t['text'],'box':local_box(t['bbox'])} for t in labels
                        if box(*edit_context).intersection(box(*t['bbox'])).area>0],
                    'native_frames':bool(frames or source_enclosure),
                    'edit_box':local_box(asset_box) if edit_context!=b else None}
                if source_enclosure:
                    edit_request['source_enclosure']=source_enclosure
                    edit_request['target']='仅保留原生外框内部的复杂图形；外框只作定位参照。'+edit_request['target']
                if d['role'] == 'icon' and (preserve_icon_pixels or registered and not all_keys):
                    edit_request['source_pixels_policy'] = 'explicit_cleanup_masks.1'
                    edit_request['removal_paths'] = []
                    for arrow in arrows:
                        if arrow['id'] in d.get('replace_arrows', []):
                            continue
                        paths = arrow_strokes(arrow)
                        if not paths:
                            continue
                        points = np.concatenate(paths)
                        bb = np.r_[points.min(0), points.max(0)]
                        if box(*asset_box).buffer(1).covers(box(*bb)):
                            continue  # Internal source content belongs to the image.
                        if not any(LineString(p).intersects(box(*asset_box)) for p in paths if len(p) > 1):
                            continue
                        edit_request['removal_paths'].extend({'points': [[float(p[0]-edit_context[0]), float(p[1]-edit_context[1])] for p in path],
                            'width_px': arrow.get('style', {}).get('width_px') or 1.,
                            'native_id':arrow['id']} for path in paths if len(path) > 1)
            # An explicitly supplied successful asset is reusable only for the
            # same source, target, placement request, role and replacement set.
            previous_asset=None
            for asset in cached_assets:
                if (asset.get('owner')==oid and asset.get('source_sha256')==scene['source_sha256']
                    and asset.get('scope_protocol') == ASSET_SCOPE_PROTOCOL
                    and asset.get('source_box_half_open')==b and asset.get('role')==d['role']
                    and asset.get('extraction_method')==d['method']
                    and set(asset.get('registry_ids',[]))==set(d['registry_ids'])
                    and set(asset.get('replace_parts',[]))==set(keys)
                    and set(asset.get('replace_arrows',[]))==set(d.get('replace_arrows',[]))
                    and asset.get('edit',{}).get('separate_foreground_parts',[])==d.get('foreground_parts',[])
                    and asset.get('edit',{}).get('preserved_native_frames',[])==frames
                    and Path(asset['path']).is_file() and sha(asset['path'])==asset['sha256']):
                    if asset.get('edit_request')==edit_request:
                        d.update(asset=asset,reused_validated_asset=True)
                        return d
                    # A rejected new edit must not erase an image-only target.
                    # Only the description may differ: native text ownership,
                    # crop, foreground, replacement sets and bytes stay fixed.
                    old_request=asset.get('edit_request') or {}
                    if (not all_keys and edit_request is not None
                        and (not edit_request.get('source_pixels_policy') or asset.get('edit', {}).get('source_pixels_preserved'))
                        and all(old_request.get(k)==edit_request.get(k)
                                for k in ('native_labels','native_frames','edit_box'))):
                        previous_asset=asset
            try:
                crop=clean.crop(edit_context);crop.save(local/'source.png');path=local/'image.png';placement=b
                if defer_cleanup and d['method']=='edit':
                    crop.convert('RGBA').save(path)
                    d['asset']={'id':'U_'+ident,'owner':oid,'role':d['role'],'path':str(path.resolve()),'sha256':sha(path),
                        'scope_protocol':ASSET_SCOPE_PROTOCOL,'edit_request':edit_request,'native_enclosure':source_enclosure,
                        'native_ellipsis':native_ellipsis,
                        'source_size':scene['source_size'],'source_sha256':scene['source_sha256'],
                        'source_box_half_open':b,'placement_source_half_open':edit_context,
                        'edit_context_half_open':edit_context,'pending_edit':True,
                        'replace_parts':keys,'replace_names':[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in keys],
                        'replace_arrows':d.get('replace_arrows',[]),'extraction_method':d['method'],'registry_ids':d['registry_ids'],
                        'edit':{'pending_source_ownership':True,'owned_box':asset_box,'preserved_native_frames':frames,
                                'separate_foreground_parts':d.get('foreground_parts',[])}}
                    return d
                if d['method']=='edit':
                    edited=local/'edited.png'
                    try:
                        receipt=editor.edit(crop,edit_request,edited,role=d['role'])
                    except Exception:
                        if not edit_request.get('source_pixels_policy'):
                            raise
                        receipt = {'source_offset_xy': [0, 0], 'source_extent_wh': list(crop.size), 'requests': 1}
                        edited = None
                    if edit_request.get('source_pixels_policy'):
                        from .image_edit import source_locked_icon
                        opened = Image.open(edited) if edited is not None else None
                        try:
                            locked, audit = source_locked_icon(crop, opened,
                                [*receipt['source_offset_xy'], *receipt['source_extent_wh']], edit_request)
                        finally:
                            if opened is not None:opened.close()
                        locked.save(local / 'source_locked.png')
                        edited = local / 'source_locked.png'
                        receipt.update(audit, source_offset_xy=[0, 0], source_extent_wh=list(crop.size))
                    shutil.copy2(edited,path)
                    x,y=[edit_context[i]+receipt['source_offset_xy'][i] for i in (0,1)]
                    w,h=receipt['source_extent_wh'];placement=[x,y,x+w,y+h]
                    clipped=clip_source_edge(path,placement,clean.size,edit_context)
                    if clipped!=placement:
                        receipt={**receipt,'slide_clip':{'before':placement,'after':clipped,
                                  'original_editor_output':str(edited.resolve())}}
                    placement=clipped
                else:
                    existing=registered[0] if len(registered)==1 and registered[0].get('extraction_method')=='existing_rgba' else None
                    if existing:
                        with Image.open(existing['path']) as im:result=im.convert('RGBA')
                        placement=existing['placement_source_half_open']
                    else:result=transparent_exterior(crop,tolerance=0,adaptive=False)
                    boundary=None
                    if not existing:result,boundary=clip_owned_crop(result,b,scene,styles,keys,owned_box=asset_box)
                    result.save(path);receipt={'method':'source_crop','alpha_method':'border_connected_white',
                                               'native_crop_boundary':boundary,'requests':0}
                foreground=d.get('foreground_parts',[]) if d['role']!='background' else []
                with Image.open(path) as im:scoped=clip_asset_scope(im.convert('RGBA'),placement,asset_box,scene,styles,frames,foreground,keys)
                if (not all_keys and registered and d['method']=='edit' and d.get('edit_box')
                    and not box(*asset_box).covers(box(*target_box))):
                    # Editing an interior must not regenerate or discard the
                    # registered outer silhouette. Retain those source pixels.
                    outer=[int(np.floor(target_box[0])),int(np.floor(target_box[1])),int(np.ceil(target_box[2])),int(np.ceil(target_box[3]))]
                    original=transparent_exterior(clean.crop(outer),tolerance=0,adaptive=False);erase=ImageDraw.Draw(original)
                    erase.rectangle((asset_box[0]-outer[0],asset_box[1]-outer[1],asset_box[2]-outer[0],asset_box[3]-outer[1]),fill=(0,0,0,0))
                    size=(max(1,round(placement[2]-placement[0])),max(1,round(placement[3]-placement[1])))
                    original.alpha_composite(scoped.resize(size,Image.Resampling.LANCZOS),(round(placement[0]-outer[0]),round(placement[1]-outer[1])))
                    scoped=clip_asset_scope(original,outer,target_box,scene,styles,foreground=foreground);placement=outer
                    receipt['unedited_registry_pixels_preserved']=True
                if source_enclosure:
                    from .source_enclosures import clip_interior
                    scoped=clip_interior(scoped,placement,source_enclosure)
                    receipt['source_enclosure']=source_enclosure
                # A registry may own only an enclosure's interior. Its source
                # crop can still copy the retained outer frame where fitted
                # and original boundaries differ. Remove that source ring,
                # rather than clipping the complete icon to the fitted frame.
                scoped,frame_copies=remove_frame_copies(scoped,placement,clean,scene,frames,keys)
                if frame_copies:receipt['removed_retained_frame_copies']=frame_copies
                scoped,ellipsis_cleanup=clear_image_copies(scoped,placement,native_ellipsis)
                if ellipsis_cleanup:receipt['native_ellipsis_cleanup']=ellipsis_cleanup
                scoped.save(path)
                receipt.update(owned_box=asset_box,preserved_native_frames=frames,
                               separate_foreground_parts=foreground)
                d['asset']={'id':'U_'+ident,'owner':oid,'role':d['role'],'path':str(path.resolve()),'sha256':sha(path),
                    'scope_protocol':ASSET_SCOPE_PROTOCOL,'edit_request':edit_request,'native_enclosure':source_enclosure,
                    'native_ellipsis':native_ellipsis,
                    'source_size':scene['source_size'],'source_sha256':scene['source_sha256'],
                    'source_box_half_open':b,'placement_source_half_open':placement,'replace_parts':keys,
                    'replace_names':[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in keys],
                    'replace_arrows':d.get('replace_arrows',[]),
                    'extraction_method':d['method'],'registry_ids':d['registry_ids'],'edit':receipt}
            except Exception as exc:
                d.update(action='keep_native' if keys else 'unresolved',asset_error=str(exc))
                if previous_asset is not None:
                    d.update(action='image',asset=previous_asset,fallback_previous_asset=True,
                        fallback_reason='New edit rejected; same source/scope/native ownership, previous validated pixels preserved')
                save(local/'asset_failure.json',d)
        return d

    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[(oid,reg,pool.submit(one,oid,reg)) for oid,reg in tasks]
        for oid,reg,future in pending:
            try:results.append(future.result())
            except (ValueError,RuntimeError) as exc:
                results.append({'owner':oid,'registry_ids':[a['id'] for a in reg],
                                'action':'keep_native' if any(p['object_id']==oid for p in scene['parts'].values()) else 'unresolved',
                                'reason':str(exc),'unresolved':True})
    assets=[]
    for a in preserved:
        row={**a,'owner':None,'role':'photo' if a['category']=='natural_photo' else 'icon',
             'source_size':scene['source_size'],'replace_parts':[],'registry_preserved_without_rejudgment':True}
        if a['category']=='complex_icon' and a.get('extraction_method')=='source_crop':
            # Preserve the known target pixels and original placement. Only
            # border-connected white becomes transparent; no generative retry.
            try:
                with Image.open(a['path']) as im:rgba=transparent_exterior(im,tolerance=0,adaptive=False)
                path=folder/(a['id']+'_alpha.png');rgba.save(path)
                row.update(path=str(path.resolve()),sha256=sha(path),extraction_method='source_crop_alpha')
            except ValueError:pass
        assets.append(row)
    assets.extend(r['asset'] for r in results if r.get('asset'))
    patterns=[r for r in results if r.get('action')=='native_pattern']
    pattern_keys={k for r in patterns for k in r['replace_parts']}
    dropped=pattern_keys|{k for a in assets for k in a['replace_parts']}
    unresolved={i for r in results if r.get('action')=='unresolved' for i in r['registry_ids']}
    result={'protocol':'unified_image_scoped_ownership.9','assets':assets,'native_patterns':patterns,
        'delegated_only_owners_not_reintroduced':sorted(delegated_only),
        'incomplete_owner_candidates':sorted(incomplete),
        'unfitted_shape_keys':sorted(unfitted),
        'drop_shape_keys':sorted(dropped),
        'drop_arrow_names':sorted({n for a in assets for n in a.get('replace_arrows',[])}),
        # Whole-owner raster replacement commits before arrow merge; only
        # pattern strokes still require the native segment-removal path.
        'drop_shape_names':sorted(scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in pattern_keys),
        'reviews':results,'model_calls':sum(not r.get('model_call_skipped') for r in results),'registry_ids_consumed':[a['id'] for a in images if a['id'] not in unresolved],
        'unresolved_registry_ids':sorted(unresolved),
        'separate_legacy_image_insertion':False}
    save(folder/'audit.json',result);return result


def consolidate_nested_assets(assets, folder=None):
    """One raster commit for nested proposals with the same native leaves.

    A larger registered crop can overlap a separately recovered source icon.
    Preserve both pixel products in one commit, with the smaller verified
    source pixels above the larger asset. Placement containment alone cannot
    prove that an editor retained the smaller target.
    Partial intersections remain explicit conflicts, never bbox deletions.
    """
    from copy import deepcopy
    live = deepcopy(assets)
    receipts = []
    for small in list(live):
        if small not in live or not small.get('replace_names'):
            continue
        refs = set(small['replace_names'])
        if small.get('pending_edit') or small.get('realization_failure'):
            continue
        bounds = box(*small['placement_source_half_open'])
        options = [large for large in live if large is not small
                   and not large.get('pending_edit') and not large.get('realization_failure')
                   and refs <= set(large.get('replace_names', []))
                   and set(small.get('replace_arrows', [])) <= set(large.get('replace_arrows', []))
                   and box(*large['placement_source_half_open']).covers(bounds)
                   and box(*large['placement_source_half_open']).area > bounds.area]
        if not options:
            continue
        large = min(options, key=lambda a: box(*a['placement_source_half_open']).area)
        if folder is None or not small.get('edit', {}).get('source_pixels_preserved'):
            continue
        if not small.get('path') or not large.get('path'):
            continue
        # One shared native target is removed once. Alpha compositing keeps
        # the source-locked inner mark even if the larger crop's cleanup
        # explicitly removed it as an external overlay.
        from .prepare import sha
        with Image.open(large['path']) as opened:combined=opened.convert('RGBA')
        with Image.open(small['path']) as opened:inner=opened.convert('RGBA')
        lb=large['placement_source_half_open'];sb=small['placement_source_half_open']
        sx=combined.width/(lb[2]-lb[0]);sy=combined.height/(lb[3]-lb[1])
        target=(max(1,round((sb[2]-sb[0])*sx)),max(1,round((sb[3]-sb[1])*sy)))
        inner=inner.resize(target,Image.Resampling.LANCZOS)
        combined.alpha_composite(inner,(round((sb[0]-lb[0])*sx),round((sb[1]-lb[1])*sy)))
        destination=Path(folder)/'nested_assets'/(large['id']+'_'+small['id']+'.png')
        destination.parent.mkdir(parents=True,exist_ok=True);combined.save(destination)
        large.update(path=str(destination.resolve()),sha256=sha(destination))
        large['registry_ids'] = sorted(set(large.get('registry_ids', [])) |
                                       set(small.get('registry_ids', [])))
        live.remove(small)
        receipts.append({'retired_asset': small['id'], 'retained_asset': large['id'],
                         'reason': 'Shared native replacement committed once; source-locked inner pixels retained above larger image',
                         'source_inner_pixels_preserved':True})
    return live, receipts


def commit(base, output, assets, size):
    """Replace exact leaves, including nested parts; preserve parent transforms."""
    from lxml import etree
    from pptx import Presentation
    from shape_integration.common import sha
    from .compose import matches_name, P, A
    deck=Presentation(str(base));slide=deck.slides[0];tree=slide.shapes._spTree
    if len(deck.slides)!=1 or abs(deck.slide_width/deck.slide_height-size[0]/size[1])>1e-4:
        raise ValueError('Asset commit requires a single slide in source coordinates')
    def leaves(root):
        return [n for n in root.iter() if etree.QName(n).localname in ('sp','pic','cxnSp','graphicFrame')]
    originals={n:etree.tostring(n) for n in leaves(tree)}
    claims={}
    for asset in assets:
        for ref in asset.get('replace_names', []):
            if ref in claims:
                raise ValueError('Conflicting image replacement ownership: '+ref)
            claims[ref]=asset['id']
    groups={n:etree.tostring(n.find('{'+P+'}grpSpPr')) for n in tree.iter('{'+P+'}grpSp')}
    removed=set();inserted=[];seen=set();scale=deck.slide_width/size[0]
    def name(node):
        n=node.find('.//{'+P+'}cNvPr');return n.get('name','') if n is not None else ''
    def ancestors(node):
        out=[];parent=node.getparent()
        while parent is not None:
            out.append(parent)
            if parent is tree:break
            parent=parent.getparent()
        return out
    for asset in assets:
        owner=asset.get('owner');identity=owner or asset['id']
        if identity in seen:raise ValueError('Two assets claim one owner: '+identity)
        seen.add(identity)
        if sha(asset['path'])!=asset['sha256']:raise ValueError('Asset bytes changed')
        refs=set(asset.get('replace_names',[])) or ({owner+'/'+json_part_id(k) for k in asset.get('replace_parts',[])} if owner else set())
        scoped='replace_parts' in asset or 'replace_names' in asset
        targets=[n for n in leaves(tree) if any(matches_name(name(n),ref) for ref in (refs if scoped else [owner] if owner else []))]
        if (refs or owner and not scoped) and not targets:raise ValueError('Missing native replacement parts: '+identity)
        if any(not any(matches_name(name(n),ref) for n in targets) for ref in refs):
            raise ValueError('Incomplete native replacement parts: '+identity)
        parent=next((n for n in ancestors(targets[0]) if all(n in ancestors(t) for t in targets)),tree) if targets else tree
        if asset['role']=='background':parent=tree
        def child(node):
            while node.getparent() is not parent:node=node.getparent()
            return node
        indices=[parent.index(child(n)) for n in targets]
        # Replacing several pieces creates one visual unit at their frontmost
        # layer. The first piece may sit behind a retained card/cell fill and
        # would hide the whole replacement, even with no protected frame.
        index=max(indices)+1 if indices else len(parent)
        if asset['role']=='background':index=2
        b=np.asarray(asset['placement_source_half_open'],float)*scale
        # Invert each ancestor's axis-aligned coordinate frame. The bitmap is
        # already in slide coordinates; moving it into a group must not rescale
        # or offset it a second time. Native leaf transforms remain untouched.
        chain=[];node=parent
        while node is not tree:
            chain.append(node);node=node.getparent()
        for group in reversed(chain):
            x=group.find('{'+P+'}grpSpPr/{'+A+'}xfrm')
            if x is None:continue
            if any(x.get(k,'0') not in ('0','false') for k in ('rot','flipH','flipV')):
                raise ValueError('Raster replacement requires an axis-aligned parent group')
            off=x.find('{'+A+'}off');ext=x.find('{'+A+'}ext');co=x.find('{'+A+'}chOff');ce=x.find('{'+A+'}chExt')
            for i,(axis,extent) in enumerate((('x','cx'),('y','cy'))):
                factor=int(ce.get(extent))/max(1,int(ext.get(extent)))
                b[[i,i+2]]=(b[[i,i+2]]-int(off.get(axis)))*factor+int(co.get(axis))
        pic=slide.shapes.add_picture(asset['path'],round(b[0]),round(b[1]),round(b[2]-b[0]),round(b[3]-b[1]))
        pic.name='IMG '+asset['id'];tree.remove(pic._element);parent.insert(index,pic._element);inserted.append(pic._element)
        enclosure=asset.get('native_enclosure')
        if enclosure:
            from pptx.enum.shapes import MSO_SHAPE
            from pptx.dml.color import RGBColor
            from pptx.util import Pt
            cx,cy=enclosure['center'];w,h=enclosure['size']
            frame=slide.shapes.add_shape(MSO_SHAPE.OVAL,round((cx+.5-w/2)*scale),round((cy+.5-h/2)*scale),
                max(1,round(w*scale)),max(1,round(h*scale)))
            frame.name='NATIVE_ENCLOSURE '+asset['id'];frame.rotation=enclosure['rotation_deg']
            frame.fill.background();frame.line.color.rgb=RGBColor.from_string(enclosure['stroke'][1:])
            frame.line.width=Pt(enclosure['width_px']*scale/12700)
            inserted.append(frame._element)
        for node in targets:removed.add(node);node.getparent().remove(node)
    expected=[xml for node,xml in originals.items() if node not in removed]
    if expected!=[etree.tostring(n) for n in leaves(tree) if n not in inserted]:
        raise ValueError('Image commit modified unrelated vector or text XML')
    if any(etree.tostring(n.find('{'+P+'}grpSpPr'))!=value for n,value in groups.items()):
        raise ValueError('Image commit changed parent transforms')
    deck.save(output)
    loaded=Presentation(str(output));image_names={'IMG '+a['id'] for a in assets}
    image_names.update('NATIVE_ENCLOSURE '+a['id'] for a in assets if a.get('native_enclosure'))
    if expected!=[etree.tostring(n) for n in leaves(loaded.slides[0].shapes._spTree) if name(n) not in image_names]:
        raise ValueError('Saving asset commit changed unrelated objects')
    return {'images':len(assets),'removed_owner_groups':len(removed),'unrelated_objects_unchanged':True,
            'images_before_arrow_and_text_merge':True,'owners':[a.get('owner') for a in assets]}
