"""One crop SAM call per observation group; native evidence never creates identities."""
from checked_copy import copy_file
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev
import visual
from mask_filters import clean_mask
from sam_client import SamClient
from transport.common import read_json as read,write_json as write,sha256 as sha


def prepare(clean,rows,out,config,box=None):
    out=Path(out);bounds=np.asarray(box if box is not None else ev.bounds([r['box'] for r in rows]),float)
    center=(bounds[:2]+bounds[2:])/2;extent=(bounds[2:]-bounds[:2])*config['crop_context_scale']
    # Preserve native resolution when a large observation group needs context.
    max_extent=min(max(clean.size),max(extent)+2*config['context_padding'])
    frame_config={**config,'display_side':max(config['display_side'],int(np.ceil(max_extent)))}
    crop,view=ev.frame(clean,[[*(center-extent/2),*(center+extent/2)]],out/'input',frame_config)
    # Global context stays clean; dense candidate labels belong to the local
    # evidence view. Otherwise icon wires and overlay text obscure semantics.
    ev.overview(clean,out/'input',config)
    context=Image.open(out/'input/context.png').convert('RGB')
    xy=np.asarray(view['bbox_global'])*np.tile(np.asarray(context.size)/clean.size,2)
    ImageDraw.Draw(context).rectangle(xy.tolist(),outline='#c00080',width=3);context.save(out/'input/context.png')
    visual.masks(crop,[(r['id'],ev.project(r,view)) for r in rows],out/'input/global_hints.png')
    return crop,view


def sample(crop,view,rows,all_hints,group_of,gid,out,config,sam_slot):
    out=Path(out);folder=out/'sam';source=out/'input/clean.png';started=time.monotonic();cached=False
    for root in config.get('sam_cache_from',[]):
        if folder.exists():break
        parts=out.parts;relative=Path(*parts[parts.index('cases'):])
        old=Path(root)/relative/'sam'
        if not (old/'predictions.json').exists():continue
        previous=read(old/'predictions.json');request=read(old/'request.json')
        if (request.get('image_sha256')==sha(source) and previous['model']['checkpoint_sha256']==config['sam_checkpoint_sha256']
                and previous['threshold']==config['score_threshold'] and previous['model']['mask_pixel_threshold']==config['pixel_threshold']):
            shutil.copytree(old,folder,copy_function=copy_file);write(out/'sam_cache.json',dict(source=str(old),image_sha256=sha(source)))
    if (folder/'predictions.json').exists():
        request=read(folder/'request.json')
        if request['image_sha256']!=sha(source):raise ValueError('Changed SAM crop')
        cached=True
    else:
        if config.get('sam_evidence_mode')=='cache_only':
            # A provider outage must not turn a missing optional measurement
            # into a fabricated mask or silently prevent a GLM-only repair.
            write(out/'sam_summary.json',dict(state='no_matching_cache',cached=False,
                new_sam_calls=0,selected_candidates=0,selected_ids=[],crop_size=list(crop.size),
                view=view,local_candidates_create_targets=False,
                reason='Explicit cache-only mode; use CLEAN and retained evidence'))
            write(out/'candidates.json',[])
            visual.masks(crop,[],out/'input/local_masks.png')
            visual.masks(crop,[],out/'input/all_local_masks.png')
            return [],[]
        with sam_slot:SamClient(config['sam_url'],config['sam_timeout_seconds']).segment(source,folder,config['sam_categories'],config['score_threshold'])
    raw=read(folder/'predictions.json')
    if raw['image_size']!=list(crop.size):raise ValueError('SAM returned another coordinate grid')
    if raw['model']['checkpoint_sha256']!=config['sam_checkpoint_sha256']:raise ValueError('SAM checkpoint changed')
    if raw['threshold']!=config['score_threshold'] or raw['model']['mask_pixel_threshold']!=config['pixel_threshold']:raise ValueError('SAM thresholds changed')
    nearby=[r for r in all_hints if ev.gap(r['box'],view['bbox_global'])==0]
    radius=config['local_owner_radius'];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(radius)*2+1,)*2)
    anchors={r['id']:cv2.dilate(ev.project(r,view).astype('uint8'),kernel)>0 for r in nearby}
    candidates=[];mask_dir=out/'masks';mask_dir.mkdir(exist_ok=True)
    for i,p in enumerate(raw['predictions'],1):
        original=ev.mask(p['mask_path']);m,removed,audit=clean_mask(original,config['mask_filter'])
        label=f'M{i}';dest=mask_dir/(label+'.png');Image.fromarray(m.astype('uint8')*255).save(dest)
        if removed.any():Image.fromarray(removed.astype('uint8')*255).save(mask_dir/(label+'_removed.png'))
        scores={g:float(2*(m&a).sum()/max(1,int(m.sum()+a.sum()))) for g,a in anchors.items()}
        best=max(scores,key=scores.get) if scores and max(scores.values())>0 else None
        owner=group_of.get(best)
        # Ownership ranks evidence; it must not hide a relevant local mask.
        # A related mask may score highest against another observation's G.
        related=any(scores.get(r['id'],0)>0 for r in rows)
        selected=(related or best is None or not all_hints) if config.get('local_evidence_retrieval')=='related' else (owner==gid or not all_hints)
        candidates.append(dict(id=label,path=str(dest.resolve()),raw_path=p['mask_path'],category=p['category'],score=p['score'],
            native_area=int(m.sum()),mask=m,filter=audit,best_global=best,owner=owner,selected=selected,scores=scores))
    # Fixed representative containment avoids fragment bridges connecting distinct
    # candidates. Members remain available, and the complete representative is used.
    selected=[]
    for row in sorted([r for r in candidates if r['selected']],key=lambda r:-r['native_area']):
        parents=[r for r in selected if (r['mask']&row['mask']).sum()/max(1,row['native_area'])>=config['duplicate_containment']]
        if len(parents)==1:
            row['represented_by']=parents[0]['id'];parents[0]['members'].append(row['id'])
        else:row['members']=[row['id']];selected.append(row)
    write(out/'candidates.json',[{k:v for k,v in r.items() if k!='mask'} for r in candidates])
    write(out/'sam_summary.json',dict(seconds=time.monotonic()-started,cached=cached,raw_candidates=len(candidates),
        selected_candidates=len(selected),selected_ids=[r['id'] for r in selected],crop_size=list(crop.size),
        internal_resolution=raw['model']['resolution'],view=view,local_candidates_create_targets=False,
        ownership_role='Evidence retrieval only; GLM may request repair or use another preserved mask'))
    visual.masks(crop,[(r['id'],r['mask']) for r in selected],out/'input/local_masks.png')
    visual.masks(crop,[(r['id'],r['mask']) for r in candidates],out/'input/all_local_masks.png')
    return selected,candidates
