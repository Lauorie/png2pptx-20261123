"""Bounded numeric control on an already selected six-point native L; no AI calls."""
from pathlib import Path
import argparse
import copy
import json
import os
import numpy as np
from PIL import Image,ImageDraw
from shape_integration.common import read,save,compare


def refine(raw,mask,rgb=None):
    out=copy.deepcopy(raw);points={p['id']:p['xy'] for p in out['points']};audit=[]
    radius=max(4,round(min(mask.shape)*.05))
    colour=None
    if rgb is not None and mask.any():
        samples=rgb[mask].astype(float)
        background=np.median(np.array([rgb[0,0],rgb[0,-1],rgb[-1,0],rgb[-1,-1]],float),axis=0)
        contrast=np.linalg.norm(samples-background,axis=1)
        samples=samples[contrast>=max(48.,np.percentile(contrast,60))]
        if len(samples)>=20:
            centre=np.median(samples,axis=0);dist=np.linalg.norm(samples-centre,axis=1)
            tolerance=float(np.clip(np.percentile(dist,80),16,48))
            if np.mean(dist<=tolerance)>=.7:colour=(centre,tolerance,background)
    for segment in raw['parts'][0]['segments']:
        ids=segment['point_ids'];a,b=[np.array(points[i],float) for i in ids]
        if abs(a[0]-b[0])<1e-7:axis=0
        elif abs(a[1]-b[1])<1e-7:axis=1
        else:return raw,[{'accepted':False,'reason':'Not an axis-aligned native-L observation'}]
        normal=float(a[axis]);lo,hi=sorted((a[1-axis],b[1-axis]));pad=max(2,(hi-lo)*.025)
        lo=max(0,int(np.ceil(lo+pad)));hi=min(mask.shape[axis],int(np.floor(hi-pad)))
        start=max(0,int(np.floor(normal-radius)));end=min(mask.shape[1-axis],int(np.ceil(normal+radius+1)))
        row={'axis':axis,'before':normal,'support_span':[lo,hi],'accepted':False}
        if hi<=lo or end<=start:audit.append(row);continue
        band=mask[lo:hi,start:end] if axis==0 else mask[start:end,lo:hi]
        counts=np.sum(band,axis=0 if axis==0 else 1);peak=int(np.argmax(counts));amount=int(counts[peak])
        row['mask_support_pixels']=amount
        if amount<max(4,(hi-lo)*.04) and colour is not None:
            pixels=rgb[lo:hi,start:end] if axis==0 else rgb[start:end,lo:hi]
            # A thin antialiased stroke is an alpha blend with its background,
            # not necessarily an exact copy of the SAM-supported core RGB.
            direction=colour[2]-colour[0];delta=colour[2]-pixels.astype(float)
            coverage=np.sum(delta*direction,axis=2)/max(1.,float(direction@direction))
            residual=np.linalg.norm(delta-coverage[:,:,None]*direction,axis=2)
            support=(coverage>=.18)&(coverage<=1.3)&(residual<=max(colour[1],np.linalg.norm(direction)*.15))
            counts=np.sum(support,axis=0 if axis==0 else 1);peak=int(np.argmax(counts));amount=int(counts[peak])
            row.update(origin='RGB_with_SAM_colour_anchor',source_colour=colour[0].tolist(),rgb_support_pixels=amount)
        if amount<max(4,(hi-lo)*.04):row['reason']='Insufficient localized stroke support';audit.append(row);continue
        active=counts>=amount*.6;left=right=peak
        while left>0 and active[left-1]:left-=1
        while right+1<len(active) and active[right+1]:right+=1
        if row.get('origin')=='RGB_with_SAM_colour_anchor':
            outside=np.r_[counts[:max(0,left-1)],counts[min(len(counts),right+2):]]
            if len(outside) and np.max(outside)>=amount*.65:
                row['reason']='Ambiguous parallel source-colour strokes';audit.append(row);continue
        if right-left+1>max(3,len(active)*.5):row['reason']='Broad fill is not a stroke centerline';audit.append(row);continue
        positions=np.arange(start+left,start+right+1,dtype=float)
        value=float(np.average(positions,weights=counts[left:right+1]))
        for pid in ids:points[pid][axis]=value
        row.update(accepted=True,after=value,delta=value-normal,normal_width=right-left+1);audit.append(row)
    return out,audit


def main():
    from native_corner_probe import point_scene,export
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--rgb-fallback',action='store_true')
    parser.add_argument('--observation-file',type=Path)
    args=parser.parse_args()
    work=Path(os.environ['PPT_API_WORK_ROOT']);project=Path(read(work/'settings.json')['project_root'])
    base=project/'runs/native_corner_sam7_api_20261021_v40';run=args.run or project/'runs/native_corner_mask_fit_20261021_v41'
    run.mkdir(parents=True,exist_ok=True)
    result=read(base/'sam/predictions.json');selected=read(base/'sam_selection.json')['selected']
    pick=max(selected,key=lambda p:p['score']);prediction=next(p for p in result['predictions'] if p['id']==pick['id'])
    with Image.open(prediction['mask_path']) as im:mask=np.asarray(im)>0
    source=project/'runs/native_corner_points_api_20261021_v37/task_0897/source_crop.png'
    with Image.open(source) as im:rgb=np.asarray(im.convert('RGB')) if args.rgb_fallback else None
    observation_file=args.observation_file or base/'observation.json'
    raw=read(observation_file);after,audit=refine(raw,mask,rgb)
    scene,fit=point_scene(after,[mask.shape[1],mask.shape[0]])
    native,report=export(scene,run/'output',False)
    with Image.open(source) as im:overlay=im.convert('RGB')
    draw=ImageDraw.Draw(overlay);table={p['id']:p['xy'] for p in after['points']}
    for segment in after['parts'][0]['segments']:draw.line([tuple(table[k]) for k in segment['point_ids']],fill='#C000E0',width=2)
    for pid,xy in table.items():draw.ellipse((xy[0]-3,xy[1]-3,xy[0]+3,xy[1]+3),fill='red');draw.text((xy[0]+3,xy[1]+3),pid,fill='black')
    overlay.save(run/'observed_points.png');save(run/'observation.json',after);save(run/'axis_fit.json',audit);save(run/'fit.json',fit)
    compare([source,observation_file.parent/'observed_points.png',run/'observed_points.png'],['CLEAN','Raw GLM points','SAM-aligned native L points'],run/'comparison.png',width_cap=2100)
    report.update(sam_epoch=7,mask_id=prediction['id'],mask_category=prediction['category'],
        accepted_axes=sum(a.get('accepted',False) for a in audit),new_model_calls=0,new_sam_calls=0,
        rgb_fallback=args.rgb_fallback,observation_source=str(observation_file),experimental_only=True,manual_source_review_required=True)
    save(run/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
