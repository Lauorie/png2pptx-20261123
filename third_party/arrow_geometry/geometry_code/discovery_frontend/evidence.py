"""Pixel evidence and original-image coordinates; masks never assign ownership."""
from pathlib import Path
import colorsys
import math
import threading
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pipeline_frames import Crop
from sam_client import SamClient
from transport.common import read_json as read,write_json as write,sha256
from core import gap,intersection

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SAM_SLOTS=threading.BoundedSemaphore(2)

def frame(meta):return Crop(tuple(meta['global_image_size_px']),tuple(meta['crop_bounds_global_half_open']),tuple(meta['actual_crop_input_size_px']))
def save(image,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp');image.save(tmp,format='PNG');tmp.replace(path)
def mask(path):return np.asarray(Image.open(path).convert('L'))>127
def bounds(m):
 y,x=np.nonzero(m)
 return [int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else None
def native_box(local,view):return view.to_global(local[:2])+view.to_global(local[2:])
def local_box(native,view):return view.to_local(native[:2])+view.to_local(native[2:])

def mask_footprint(mrow):
 """SAM bounds describe inclusive pixel centers, not zero-area geometry.

 A one-pixel column occupies one source pixel of width. Keep the stored
 center-coordinate bbox unchanged; use pixel footprints for spatial tests.
 """
 source=frame(mrow['frame']);b=mrow['bbox_mask']
 return native_box([b[0]-.5,b[1]-.5,b[2]+.5,b[3]+.5],source)

def boxes(clean,rows,path,view,context=()):
 image=Image.open(clean).convert('RGB');d=ImageDraw.Draw(image);font=ImageFont.truetype(FONT,max(11,min(18,round(max(image.size)/110))))
 for row,is_context in [(x,True) for x in context]+[(x,False) for x in rows]:
  b=local_box(row['bbox_native'],view);b=[max(0,min(v,(image.width-1,image.height-1)[i%2])) for i,v in enumerate(b)]
  if b[0]>=b[2] or b[1]>=b[3]:continue
  color='#777777' if is_context else '#8e24aa' if row.get('uncertain') or row.get('extent')=='partial' else '#1565c0' if row['kind'] in {'shared_network','dense_network','dense_region'} else '#e65100'
  d.rectangle(b,outline=color,width=max(1,round(max(image.size)/1000)))
  d.text((b[0],max(0,b[1]-font.size)),row.get('display_id',row['id']),font=font,fill=color,stroke_width=1,stroke_fill='white')
 save(image,path)

def check_prediction(folder,image,cfg):
 req=read(folder/'request.json');data=read(folder/'predictions.json');model=data['model']
 if req['image_sha256']!=sha256(image):raise ValueError('SAM input hash mismatch')
 if req['categories']!=cfg['sam_categories'] or data['threshold']!=cfg['score_threshold']:raise ValueError('SAM category/threshold mismatch')
 if model['checkpoint_sha256']!=cfg['sam_checkpoint_sha256'] or model['mask_pixel_threshold']!=cfg['pixel_threshold'] or model['checkpoint']['epoch']!=cfg['sam_epoch']:raise ValueError('SAM checkpoint/pixel threshold mismatch')
 if data['image_size']!=list(Image.open(image).size):raise ValueError('SAM coordinate grid mismatch')
 return data

def get_sam(image,folder,cfg,view,prefix,cache=None):
 if (folder/'evidence.json').exists():return read(folder/'evidence.json')['masks']
 folder.mkdir(parents=True,exist_ok=True);origin=folder/'raw';reused=False
 if cache is not None:
  try:data=check_prediction(cache,image,cfg);origin=cache;reused=True
  except (OSError,ValueError,KeyError) as exc:write(folder/'cache_rejection.json',{'path':str(cache),'reason':str(exc)})
 if not reused:
  with SAM_SLOTS:
   if not (origin/'predictions.json').exists():SamClient(cfg['sam_url'],cfg['sam_timeout_seconds']).segment(image,origin,cfg['sam_categories'],cfg['score_threshold'])
  data=check_prediction(origin,image,cfg)
 rows=[]
 for p in data['predictions']:
  raw=Path(p['mask_path']);m=mask(raw)
  if m.shape!=tuple(view.input_size[::-1]):raise ValueError('SAM mask shape mismatch')
  b=bounds(m)
  if b is None:continue
  rows.append({'id':prefix+'/'+p['id'],'mask_path':str(raw),'mask_sha256':sha256(raw),'frame':view.metadata(view.global_size),
               'bbox_native':native_box(b,view),'bbox_mask':b,'area':int(m.sum()),'category':p['category'],'score':p['score']})
 write(folder/'evidence.json',{'masks':rows,'reused':reused,'prediction_path':str(origin/'predictions.json'),
       'prediction_sha256':sha256(origin/'predictions.json'),'input_sha256':sha256(image),'source_frame':view.metadata(view.global_size)})
 return rows

def project(mrow,target):
 if sha256(mrow['mask_path'])!=mrow['mask_sha256']:raise ValueError('SAM mask changed')
 source=frame(mrow['frame']);m=mask(mrow['mask_path']).astype('uint8')
 a=target.to_global([0,0]);b=target.to_global([1,1]);p=source.to_local(a);q=source.to_local(b)
 matrix=np.array([[q[0]-p[0],0,p[0]],[0,q[1]-p[1],p[1]]],dtype='float64')
 return cv2.warpAffine(m,matrix,target.input_size,flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)

def select_masks(all_masks,focus,view,limit=32):
 candidates=[m for m in all_masks if intersection(mask_footprint(m),view.box)>0
             and any(intersection(mask_footprint(m),f['bbox_native'])>0 or gap(mask_footprint(m),f['bbox_native'])<=3 for f in focus)]
 candidates.sort(key=lambda m:(min(gap(mask_footprint(m),f['bbox_native']) for f in focus),
                                -max(intersection(mask_footprint(m),f['bbox_native'])/max(1,(mask_footprint(m)[2]-mask_footprint(m)[0])*(mask_footprint(m)[3]-mask_footprint(m)[1])) for f in focus),-m['score']))
 selected=[];seen=set()
 for m in candidates:
  # Exact pixel duplicates are redundant evidence, not identity merges.
  projected=project(m,view);b=bounds(projected)
  if b is None:continue
  import hashlib
  digest=hashlib.sha256(projected.tobytes()).hexdigest()
  if digest in seen:continue
  seen.add(digest)
  selected.append({**m,'alias':f'M{len(selected)+1}','bbox_target':b,'projected':projected})
  if len(selected)>=limit:break
 return selected,len(candidates)

def overlay_masks(clean,selected,path):
 # Baseline visual.masks approach: average overlapping colors, alpha=.43, clean image separately supplied.
 image=np.asarray(Image.open(clean).convert('RGB')).astype(float);count=np.zeros(image.shape[:2]);tints=np.zeros_like(image)
 for i,row in enumerate(selected):
  color=np.array(colorsys.hsv_to_rgb(i*.61803398875%1,.8,.85))*255;m=row['projected'];count[m]+=1;tints[m]+=color
 covered=count>0;image[covered]=.57*image[covered]+.43*tints[covered]/count[covered,None]
 image=Image.fromarray(np.rint(image).astype('uint8'));label_rows=[]
 for i,row in enumerate(selected):
  c=tuple(round(x*255) for x in colorsys.hsv_to_rgb(i*.61803398875%1,.8,.85))
  label_rows.append({'id':row['alias'],'kind':'single','bbox_native':row['bbox_target'],'visual_color':'#%02x%02x%02x'%c})
 from visual_labels import render
 render(image,label_rows,path,Crop(image.size,(0,0,*image.size),image.size),
        label_map={r['id']:r['id'] for r in label_rows},draw_box_outlines=False)

def sam_only_observations(all_masks,glm_rows):
 rows=[];audit=[]
 for m in all_masks:
  view=frame(m['frame']);raw=mask(m['mask_path']);covered=np.zeros(raw.shape,dtype=bool)
  for g in glm_rows:
   if intersection(g['bbox_native'],mask_footprint(m))<=0:continue
   b=local_box(g['bbox_native'],view);x0,y0=max(0,math.floor(b[0])),max(0,math.floor(b[1]));x1,y1=min(raw.shape[1],math.ceil(b[2])+1),min(raw.shape[0],math.ceil(b[3])+1)
   if x1>x0 and y1>y0:covered[y0:y1,x0:x1]=True
  fraction=float((raw&covered).sum()/max(1,raw.sum()))
  candidate=fraction<.5
  audit.append({'mask_id':m['id'],'fraction_inside_glm_boxes':fraction,'nominated_as_observation':candidate,
                'inside_box_is_not_verified_ownership':True})
  if candidate:
   b=m['bbox_mask'];partial=(b[0]<=1 and view.box[0]>0) or (b[1]<=1 and view.box[1]>0) or (b[2]>=view.input_size[0]-2 and view.box[2]<view.global_size[0]) or (b[3]>=view.input_size[1]-2 and view.box[3]<view.global_size[1])
   # Zero-width binary strokes get a one-pixel display margin; raw mask remains unchanged.
   bb=m['bbox_native'];bb=[max(0,bb[0]-.5),max(0,bb[1]-.5),min(view.global_size[0]-1,bb[2]+.5),min(view.global_size[1]-1,bb[3]+.5)]
   rows.append({'id':'SAM:'+m['id'],'kind':'sam_proposal','bbox_native':bb,'locator':'Unverified SAM '+m['category']+' evidence; may be non-target border/noise',
                'extent':'partial' if partial else 'complete','source_view':m['id'].split('/')[0],'proposal_mask_id':m['id'],'sam_only':True})
 return rows,audit
