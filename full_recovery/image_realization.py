"""Generate image pixels after original-source line ownership is available."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from PIL import Image


def realize(texture, scene, styles, source, plan, editor):
    from .image_edit import source_locked_icon
    from .image_assets import clip_asset_scope,clip_source_edge,remove_frame_copies
    from .compose import sha
    from .source_enclosures import clip_interior
    from .source_ellipsis import clear_image_copies
    approved=set()
    for receipt in plan.get('quality_choice',{}).get('source_receipts',[]):
        if receipt.get('source_kind') in ('line','arrow') and receipt.get('status') in ('resolved','incomplete'):
            approved.update(c[6:] for c in receipt.get('candidate_ids',[]) if c.startswith('arrow:'))
    def one(asset):
        if not asset.get('pending_edit'):return
        local=Path(asset['path']).parent;context=asset['edit_context_half_open'];request=deepcopy(asset['edit_request'])
        paths=request.get('removal_paths',[])
        # A crossing prediction is not source ownership. Unreviewed native
        # content cannot authorize erasing intrinsic image pixels.
        hidden={ident for ident,spec in plan.get('image_source_occlusion_order',{}).items()
                if 'IMG '+asset['id'] in spec.get('before',[])} if asset.get('native_enclosure') else set()
        request['removal_paths']=[r for r in paths if r.get('native_id') in approved and r.get('native_id') not in hidden]
        with Image.open(source) as opened:crop=opened.convert('RGBA').crop(context)
        output=local/'edited.png'
        try:
            # An empty source-locked cleanup mask cannot use generated pixels.
            # Avoid a model call whose entire result would be discarded.
            if request.get('source_pixels_policy') and not request.get('native_labels') and not request.get('removal_paths'):
                receipt={'source_offset_xy':[0,0],'source_extent_wh':list(crop.size),'requests':0,
                         'generation_skipped':'empty_source_cleanup_mask'}
                output=None
            else:
                if editor is None:raise RuntimeError('Image editor is not configured')
                receipt=editor.edit(crop,request,output,role=asset['role'])
        except Exception as exc:
            if not request.get('source_pixels_policy'):
                asset['realization_failure']={'error_type':type(exc).__name__,'fallback':'retain_native_owner'}
                return
            receipt={'source_offset_xy':[0,0],'source_extent_wh':list(crop.size),'requests':1}
            receipt['editor_error_type']=type(exc).__name__
            output=None
        if request.get('source_pixels_policy'):
            edited=Image.open(output) if output else None
            try:image,audit=source_locked_icon(crop,edited,[*receipt['source_offset_xy'],*receipt['source_extent_wh']],request)
            finally:
                if edited is not None:edited.close()
            image.save(local/'source_locked.png');receipt.update(audit)
            placement=context
        else:
            with Image.open(output) as opened:image=opened.convert('RGBA')
            x,y=[context[i]+receipt['source_offset_xy'][i] for i in (0,1)]
            w,h=receipt['source_extent_wh'];placement=[x,y,x+w,y+h]
        meta=asset['edit'];owned=meta['owned_box']
        retained_scope=(context if request.get('edit_box') is not None and asset.get('registry_ids')
                        and not asset['replace_parts'] else owned)
        try:
            image=clip_asset_scope(image,placement,retained_scope,scene,styles,meta['preserved_native_frames'],
                                   meta['separate_foreground_parts'],asset['replace_parts'])
        except ValueError as exc:
            # A generated crop whose pixels do not intersect its owned scope
            # is an invalid replacement, not a case-level failure.  Let the
            # cleanup below restore the native owner and leave the asset
            # unresolved for the next review.  This keeps a bad editor crop
            # from dropping shapes or aborting unrelated objects.
            asset['realization_failure']={
                'error_type':type(exc).__name__,
                'error':str(exc),
                'fallback':'retain_native_owner',
            }
            return
        if meta['preserved_native_frames']:
            with Image.open(source) as original:
                image,frame_copies=remove_frame_copies(image,placement,original.convert('RGB'),scene,
                    meta['preserved_native_frames'],asset['replace_parts'])
            if frame_copies:receipt['removed_retained_frame_copies']=frame_copies
        if asset.get('native_enclosure'):image=clip_interior(image,placement,asset['native_enclosure'])
        image,ellipsis_cleanup=clear_image_copies(image,placement,asset.get('native_ellipsis',()))
        if ellipsis_cleanup:receipt['native_ellipsis_cleanup']=ellipsis_cleanup
        image.save(asset['path'])
        placement=clip_source_edge(Path(asset['path']),placement,scene['source_size'],context)
        receipt.update({k:v for k,v in meta.items() if k!='pending_source_ownership'})
        receipt.update(source_ownership_finalized_before_edit=True,
                       unreviewed_cleanup_paths_preserved=len(paths)-len(request['removal_paths']))
        asset.update(sha256=sha(asset['path']),placement_source_half_open=placement,
                     edit_request=request,edit=receipt,pending_edit=False)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,texture['assets']))
    failed=[a for a in texture['assets'] if a.get('realization_failure')]
    texture['realization_failures']=[{'id':a['id'],'owner':a.get('owner'),**a['realization_failure']} for a in failed]
    if failed:
        restored={k for a in failed for k in a['replace_parts']}
        texture['assets']=[a for a in texture['assets'] if a not in failed]
        live={k for a in texture['assets'] for k in a['replace_parts']}
        texture['drop_shape_keys']=[k for k in texture.get('drop_shape_keys',[]) if k not in restored or k in live]
        names={n for a in failed for n in a.get('replace_names',[])}
        texture['drop_shape_names']=[n for n in texture.get('drop_shape_names',[]) if n not in names]
        failed_registry={i for a in failed for i in a.get('registry_ids',[])}
        texture['registry_ids_consumed']=[i for i in texture.get('registry_ids_consumed',[]) if i not in failed_registry]
        texture['unresolved_registry_ids']=sorted(set(texture.get('unresolved_registry_ids',[]))|failed_registry)
        for review in texture.get('reviews',[]):
            if any(review.get('asset') is a for a in failed):
                review['action']='keep_native' if review.get('owner') else 'unresolved'
                review['realization_failed']=True
    from .image_assets import consolidate_nested_assets
    folder=next((Path(a['path']).parent for a in texture['assets'] if a.get('path')),None)
    texture['assets'], nested = consolidate_nested_assets(texture['assets'],folder)
    texture['nested_asset_ownership'] = nested
    # Proposed image ownership cannot remove native arrows before the pixels
    # exist. Once realization finishes, rebuild the commit set from successful
    # assets; retaining the earlier cleared list duplicates their internal ink.
    texture['drop_arrow_names']=sorted({ident for asset in texture['assets']
        if not asset.get('pending_edit') and not asset.get('realization_failure')
        for ident in asset.get('replace_arrows',[])})
    return texture
