"""Shared experiment I/O, coordinate transport, rendering, and bounded VLM calls."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from png2pptx.frames import scale_box
from .transport import DeadlineDetectorClient, RequestTimeout

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def read(p): return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def valid_box(b,size):
    return isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=size[0]-1 and 0<=b[1]<b[3]<=size[1]-1
def intersect(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def mapped_box(b,source_size,target_size):
    q=scale_box(b,source_size,target_size)
    return [max(0,min(v,target_size[i%2]-1)) for i,v in enumerate(q)]
def geometry(objects,size,uncertain=None):
    return {'schema_version':'figure_elements.1','task':'shape_bbox','coordinate_system':'image_pixel_centers','image_size':list(size),'model_units':'absolute','objects':objects,'uncertain':uncertain or [],'more_pending':bool(uncertain)}

def overlay(source,objects,path,color=None):
    im=source.copy().convert('RGB');d=ImageDraw.Draw(im);f=ImageFont.truetype(FONT,max(12,round(max(im.size)/130)))
    palette=['#e00090','#0080db','#00936c','#b66600','#7138de']
    for i,o in enumerate(objects):
        b=o['bbox_2d'];c=color or palette[i%len(palette)];d.rectangle(b,outline=c,width=2)
        label=o['id'];tb=d.textbbox((0,0),label,font=f);w,h=tb[2]+4,tb[3]+4
        x=max(0,min(b[0],im.width-w));y=max(0,min(b[1]-h,im.height-h))
        d.rectangle((x,y,x+w,y+h),fill='white');d.text((x+2,y),label,font=f,fill=c)
    im.save(path)

def compare(paths,labels,path,width_cap=1000):
    ims=[Image.open(p).convert('RGB') for p in paths];w,h=ims[0].size;scale=min(1,width_cap/w);w,h=round(w*scale),round(h*scale)
    board=Image.new('RGB',(w*len(ims),h+35),'white');d=ImageDraw.Draw(board);f=ImageFont.truetype(FONT,18)
    for i,(im,label) in enumerate(zip(ims,labels)):
        board.paste(im.resize((w,h),Image.Resampling.LANCZOS),(i*w,35));d.text((i*w+7,7),label,font=f,fill='black')
    board.save(path)

class ExperimentDetectorClient(DeadlineDetectorClient):
    """Honor optional effort without changing the linked production client."""
    def __init__(self, *args, reasoning_effort=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort

    def request_parameters(self):
        parameters = super().request_parameters()
        if self.reasoning_effort is not None:
            parameters['reasoning_effort'] = self.reasoning_effort
        return parameters


class ApiSlots(threading.BoundedSemaphore):
    """Shared limiter with actual in-flight request counts for run diagnostics."""
    def __init__(self, limit):
        super().__init__(limit)
        self.stats_lock = threading.Lock()
        self.active = 0; self.peak = 0; self.limit = limit

    def __enter__(self):
        self.acquire()
        with self.stats_lock:
            self.active += 1; self.peak = max(self.peak, self.active)
        return self

    def __exit__(self, *args):
        with self.stats_lock:
            self.active -= 1
        self.release()

    def snapshot(self):
        with self.stats_lock:
            return {'limit': self.limit, 'active': self.active, 'peak': self.peak}


class Runtime:
    def __init__(self,config,base,key):
        self.config=config;self.base=base;self.key=key;self.slots=ApiSlots(config['api_concurrency'])
    def client(self,root,budget):
        deadline = self.config.get('request_timeout_seconds', 600)
        return ExperimentDetectorClient(root,self.config['model'],self.slots,self.base,self.key,max_tokens=budget,max_attempts=1,thinking='on',read_timeout=deadline,request_timeout_seconds=deadline,write_timeout_seconds=self.config.get('request_write_timeout_seconds',30),reasoning_effort=self.config.get('reasoning_effort'))
    def for_stage(self, **overrides):
        stage = copy.copy(self)
        stage.config = {**self.config, **overrides}
        return stage
    def call(self,root,tag,system,prompt,images,validator,budget,*,retry_builder=None):
        c=self.client(root,budget)
        identity={'system':system,'prompt':prompt,'images':[{'label':label,'sha256':sha(path)} for label,path in images],'parameters':c.request_parameters(),'endpoint':self.base,
                  'request_timeout_seconds':self.config.get('request_timeout_seconds',600),'timeout_policy':'discard_no_retry'}
        error=None; previous_folder=None
        for attempt in range(self.config['shape_schema_retries']+1):
            actual=copy.deepcopy(prompt)
            actual_system=system; actual_images=images; actual_validator=validator
            if error:
                if retry_builder is not None:
                    retry=retry_builder(previous_folder,error)
                    actual_system,actual,actual_images,actual_validator=(retry[k] for k in ('system','prompt','images','validator'))
                else:
                    actual.update(validation_error=error,correction='Inspect the same images and return a COMPLETE valid response. Previous response coordinates are not supplied.')
            name=f'{tag}_{attempt:02d}';folder=Path(root)/name;ident={**identity,'actual_prompt':actual}
            if retry_builder is not None:
                ident.update(actual_system=actual_system,actual_images=[{'label':label,'sha256':sha(path)} for label,path in actual_images])
            if folder.exists():
                if not (folder/'identity.json').exists() or read(folder/'identity.json')!=ident:raise ValueError('Request identity mismatch: '+name)
                if (folder/'result.json').exists():return actual_validator(read(folder/'result.json'))
                saved_error=read(folder/'error_0.json') if (folder/'error_0.json').exists() else {}
                error=saved_error.get('error','Interrupted prior call')
                if saved_error.get('type') == 'RequestTimeout':raise RequestTimeout(error)
                previous_folder=folder
                continue
            try:
                return c.call(name,actual_system,actual,actual_images,actual_validator)
            except RequestTimeout:
                raise
            except (RuntimeError,ValueError) as exc:error=c.redact(exc)
            finally:
                if folder.exists():save(folder/'identity.json',ident)
            previous_folder=folder
        raise RuntimeError(error or 'VLM call failed')

def usage(root):
    rows=[];missing=[];responses=[];requests=[]
    # Inputs are atomically published by renaming .preparing directories while
    # other targets run. Do not recurse into those transient trees for accounting.
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if not d.endswith('.preparing') and d!='code_snapshot']
        folder=Path(directory)
        if 'response_0.json' in files:responses.append(folder/'response_0.json')
        if 'request.json' in files:requests.append(folder/'request.json')
    for f in sorted(responses):
        raw=read(f);u=raw.get('usage')
        if not u:missing.append(str(f.relative_to(root)));continue
        rows.append({'call':str(f.parent.relative_to(root)),'input_tokens':u.get('prompt_tokens',0),'output_tokens':u.get('completion_tokens',0),'reasoning_tokens':(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0),'total_tokens':u.get('total_tokens',0)})
    incomplete=[str(f.parent.relative_to(root)) for f in requests if not (f.parent/'response_0.json').exists()]
    return {'calls':rows,'totals':{k:sum(r[k] for r in rows) for k in ['input_tokens','output_tokens','reasoning_tokens','total_tokens']},'calls_without_usage':missing,'calls_without_response':incomplete,'reasoning_included_in_output':True}
