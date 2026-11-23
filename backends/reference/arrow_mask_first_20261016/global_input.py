"""Global SAM only: no crop augmentation and no pooled local observations."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from transport.common import read_json as read,write_json as write,sha256 as sha
from sam_client import SamClient
from mask_filters import clean_mask


def load_case(case,config,out):
    import evidence as ev
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=Path(config['input_run'])/'cases'/case/'clean.png'
    clean=Image.open(source).convert('RGB');clean.save(out/'clean.png');digest=sha(source)
    cache=Path(config['global_predictions_run'])/'cases'/case/'sam'
    def validate(folder):
        req=read(folder/'request.json');data=read(folder/'predictions.json')
        if req['image_sha256']!=digest:raise ValueError('Global SAM input hash differs')
        if req['categories']!=config['sam_categories']:raise ValueError('Global SAM categories differ')
        if data['threshold']!=config['score_threshold']:raise ValueError('Global SAM score threshold differs')
        model=data['model']
        if model['mask_pixel_threshold']!=config['pixel_threshold']:raise ValueError('Global SAM pixel threshold differs')
        if model['checkpoint_sha256']!=config['sam_checkpoint_sha256'] or model['checkpoint']['epoch']!=config['sam_epoch']:raise ValueError('Global SAM checkpoint differs')
        if data['image_size']!=list(clean.size):raise ValueError('Global SAM coordinate frame differs')
        return data
    reused=True;data=None;rejected=[]
    choices=[cache]+[Path(root)/'cases'/case/'sam' for root in config.get('sam_cache_from',[])]
    for candidate in choices:
        try:data=validate(candidate);cache=candidate;break
        except (OSError,ValueError,KeyError) as exc:rejected.append(dict(error=str(exc),path=str(candidate)))
    if rejected:write(out/'global_cache_rejected.json',dict(candidates=rejected))
    if data is None:
        cache=out/'sam';reused=False
        if not (cache/'predictions.json').exists():
            SamClient(config['sam_url'],config['sam_timeout_seconds']).segment(source,cache,config['sam_categories'],threshold=config['score_threshold'])
        data=validate(cache)
    rows=[];view=ev.identity_view(clean.size);folder=out/'global_masks';folder.mkdir(exist_ok=True)
    audits=[]
    for i,pred in enumerate(data['predictions'],1):
        raw=Path(pred['mask_path']);mask=ev.mask(raw)
        if mask.shape!=tuple(clean.size[::-1]):raise ValueError('Wrong global mask grid')
        filtered,removed,audit=clean_mask(mask,config['mask_filter']);path=folder/f'G{i}.png'
        Image.fromarray(filtered.astype('uint8')*255).save(path)
        if removed.any():Image.fromarray(removed.astype('uint8')*255).save(folder/f'G{i}_removed.png')
        support,box=ev.support(filtered)
        rows.append(dict(id=f'G{i}',source_id=pred['id'],path=str(path.resolve()),sha256=sha(path),raw_path=str(raw),raw_sha256=sha(raw),
            view=view,box=box,area=int(filtered.sum()),category=pred['category'],score=pred['score'],filter=audit,_support=support))
        audits.append(dict(id=f'G{i}',**audit))
    write(out/'global_hints.json',ev.compact(rows));write(out/'local_observations.json',[]);write(out/'mask_filter_audit.json',audits)
    import visual
    canvas=clean.copy();canvas.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    layers=[(r['id'],np.asarray(Image.open(r['path']).resize(canvas.size,Image.Resampling.NEAREST))>0) for r in rows]
    visual.masks(canvas,layers,out/'global_masks.png',white=True);visual.masks(canvas,layers,out/'global_overlay.png')
    write(out/'source_manifest.json',dict(case=case,input_mode='global_sam_only',image_sha256=digest,
        sam_prediction_path=str(cache/'predictions.json'),sam_prediction_sha256=sha(cache/'predictions.json'),
        checkpoint_sha256=config['sam_checkpoint_sha256'],epoch=config['sam_epoch'],sam_url=config['sam_url'],
        global_hints=len(rows),local_observations=0,crop_sam_calls=0,global_sam_cache_reused=reused,new_global_sam_calls=int(not reused)))
    return clean,rows,[]


def attach_references(clean,targets,hints,out,config,cpu):
    """Record already declared global evidence; no second GLM mask-selection request."""
    import evidence as ev
    import graph as gr
    import visual
    index={r['id']:r for r in hints};result=[]
    for target in targets:
        folder=out/'targets'/target['id'];folder.mkdir(parents=True,exist_ok=True)
        refs=[index[s] for s in target['sources'] if s in index]
        row=dict(target_id=target['id'],status='global_reference' if refs else 'missing',
            geometry_ids=[r['id'] for r in refs],style_ids=[r['id'] for r in refs],geometry_sources=ev.compact(refs),style_sources=ev.compact(refs),
            observations_reviewed=0,observations_total=len(hints),more_requested_but_deferred=False,
            all_observations_retained=True,pixels_unioned_for_fitting=False,new_targets_created=0,pages=[],
            evidence_quality_assessed=False,note='GLM-declared global references only; no local mask pool or selection call')
        write(folder/'association.json',row);result.append(row)
        with cpu:
            crop,view=ev.frame(clean,[gr.box(target['graph'])],folder/'view',config)
            visual.targets(crop,[dict(id=target['id'],label='T',graph=gr.local(target['graph'],view))],folder/'target.png')
            layers=[(r['id'],ev.project(r,view)) for r in refs]
            visual.masks(crop,layers,folder/'selected_masks.png',white=True);visual.masks(crop,layers,folder/'selected_overlay.png')
        (folder/'README.md').write_text('# 目标与全局 SAM 线索\n\n此处只是查看原有全局 mask 的局部区域，没有重新调用 SAM，也没有判断精细拟合质量。\n\n'
            '| 原图局部 | 目标粗点图 | 全局 mask 线索 | 叠图 |\n|---|---|---|---|\n'
            '| ![](view/clean.png) | ![](target.png) | ![](selected_masks.png) | ![](selected_overlay.png) |\n\n'
            '[原始引用与状态](association.json)\n')
    return result
