"""Recover wrapped ordinary baselines by matching existing transcription to pixels.

This is not OCR: characters and their order are immutable. Candidate word breaks
must account for every source baseline with one consistent typeface and size.
Math, ambiguous partitions, clipped/mismatched text and identity collisions are
left unchanged with an audit record.
"""
from copy import deepcopy
from functools import lru_cache
import math
import re

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ..common import save
from .foreground import contrast_ink


def bands(row,image,complete=True):
    b=row['bbox_2d']
    x0=max(0,math.floor(b[0])-2);y0=max(0,math.floor(b[1])-2)
    x1=min(image.width,math.ceil(b[2])+3);y1=min(image.height,math.ceil(b[3])+3)
    if x1-x0<12 or y1-y0<20:return [],None,(x0,y0)
    rgb=np.array(image.crop((x0,y0,x1,y1)).convert('RGB'),dtype=np.float32)
    ink=contrast_ink(rgb,[0,0,x1-x0,y1-y0]);mask=(ink>.20).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    heights=[s[3] for s in stats[1:] if s[3]>=4 and s[4]>=6]
    character_height=float(np.percentile(heights,65)) if len(heights)>=10 else mask.shape[0]
    # Remove frame edges, underlines and isolated single pixels, not characters.
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if (area<3 or (w>mask.shape[1]*.65 and h<=3) or h>mask.shape[0]*.65
                or h>character_height*2.4):
            mask[labels==i]=0;ink[labels==i]=0
        else:
            # Geometry must not penalize a green word simply because adjacent
            # black words have stronger contrast. Normalize each ink component,
            # retaining its antialias shape rather than adopting the model hue.
            selected=labels==i
            strength=float(np.percentile(ink[selected],90))
            ink[selected]=np.clip(ink[selected]/max(strength,.20),0,1)
    mask=(ink>.38).astype(np.uint8)
    ys=np.flatnonzero(mask.sum(axis=1)>=max(2,min(6,round(mask.shape[1]*.02))))
    if not len(ys):return [],ink,(x0,y0)
    groups=np.split(ys,np.flatnonzero(np.diff(ys)>max(2,round(mask.shape[0]*.02)))+1)
    tallest=max(g[-1]-g[0]+1 for g in groups)
    valid=[g for g in groups if g[-1]-g[0]+1>=max(4,tallest*.35)]
    out=[]
    for g in valid:
        top,bottom=int(g[0]),int(g[-1])+1
        if not complete:
            yy,xx=np.where(mask[top:bottom])
            if len(xx)>=8 and xx.max()-xx.min()>=3:
                out.append([int(xx.min()),top,int(xx.max()+1),bottom])
            continue
        # Dense projection rows locate a baseline, but do not bound its ink:
        # sparse descenders and accents belong to the same glyph component.
        ids=np.unique(labels[top:bottom][mask[top:bottom]>0]);ids=ids[ids!=0]
        if not len(ids):continue
        parts=stats[ids]
        left=int(parts[:,0].min());right=int((parts[:,0]+parts[:,2]).max())
        top=int(parts[:,1].min());bottom=int((parts[:,1]+parts[:,3]).max())
        if int(parts[:,4].sum())<8 or right-left<3:continue
        out.append([left,top,right,bottom])
    if not 2<=len(out)<=16:return [],ink,(x0,y0)
    typical_height=float(np.median([b[3]-b[1] for b in out]))
    if any(out[i+1][1]-out[i][3]<max(3,typical_height*.20) for i in range(len(out)-1)):
        return [],ink,(x0,y0)
    return out,ink,(x0,y0)


@lru_cache(maxsize=4096)
def template(text,path):
    font=ImageFont.truetype(path,64);b=font.getbbox(text)
    im=Image.new('L',(max(1,b[2]-b[0]),max(1,b[3]-b[1])))
    ImageDraw.Draw(im).text((-b[0],-b[1]),text,font=font,fill=255)
    box=im.getbbox()
    return im.crop(box) if box else im


def partition(text,words,boxes,path,size):
    """Geometry proposes breaks; pixel verification below decides acceptance."""
    n=len(words);k=len(boxes);font=ImageFont.truetype(path,64)
    states={(0,0):(0.,[])}
    for line,box in enumerate(boxes):
        width=box[2]-box[0];height=box[3]-box[1]
        for start in range(n):
            if (line,start) not in states:continue
            previous,parts=states[(line,start)]
            for end in range(start+1,n-(k-line-1)+1):
                span=(words[start].start(),words[end-1].end())
                b=font.getbbox(text[span[0]:span[1]])
                w=(b[2]-b[0])*size/64;h=(b[3]-b[1])*size/64
                cost=previous+((w-width)/max(width,8))**2+.35*((h-height)/max(height,4))**2
                key=(line+1,end)
                if key not in states or cost<states[key][0]:states[key]=(cost,[*parts,span])
    return states.get((k,n))


def match(text,path,box,ink):
    im=template(text,path);l,t,r,b=box
    x0=max(0,l-2);y0=max(0,t-2);x1=min(ink.shape[1],r+2);y1=min(ink.shape[0],b+2)
    target=ink[y0:y1,x0:x1];mass=float((target*target).sum());best=None
    base=min((r-l+1)/im.width,(b-t+1)/im.height)
    for multiplier in np.linspace(.85,1.15,13):
        scale=base*multiplier;w=max(2,round(im.width*scale));h=max(2,round(im.height*scale))
        if w>target.shape[1] or h>target.shape[0]:continue
        values=np.array(im.resize((w,h),Image.Resampling.LANCZOS),dtype=np.float32)/255
        dot=cv2.matchTemplate(target,values,cv2.TM_CCORR)
        y,x=np.unravel_index(dot.argmax(),dot.shape)
        score=2*max(0.,float(dot[y,x]))/(mass+float((values*values).sum())+1e-6)
        if best is None or score>best['score']:
            best={'score':score,'size':scale*64,'bbox':[x0+int(x),y0+int(y),x0+int(x)+w-1,y0+int(y)+h-1]}
    return best


def slice_runs(runs,start,end):
    result=[];offset=0
    for run in runs:
        lo=max(start,offset);hi=min(end,offset+len(run['text']))
        if hi>lo:result.append({**run,'text':run['text'][lo-offset:hi-offset]})
        offset+=len(run['text'])
    return result


def _propose(row,image,complete):
    if row.get('kind')=='formula' or any(r['script']!='normal' for r in row['runs']):
        return None,{'status':'ineligible_math'}
    words=list(re.finditer(r'\S+',row['text']))
    if not 2<=len(words)<=80:return None,{'status':'ineligible_word_count'}
    boxes,ink,offset=bands(row,image,complete)
    if not boxes or len(words)<len(boxes):return None,{'status':'no_supported_baselines'}
    from . import FONTS
    from .layout import resolve_font
    proposals={};seen=set()
    typical=float(np.median([b[3]-b[1] for b in boxes]))/.72
    for family in (row['font'],*FONTS):
        path=resolve_font(family,row['bold'],row['italic'])['path']
        if path in seen:continue
        seen.add(path)
        for factor in (.8,1.,1.2,1.4):
            candidate=partition(row['text'],words,boxes,path,typical*factor)
            if candidate is None:continue
            cost,spans=candidate;key=(family,tuple(spans))
            proposals[key]=(cost,spans,family,path)
    verified=[]
    for cost,spans,family,path in sorted(proposals.values())[:8]:
        matches=[match(row['text'][a:b],path,box,ink) for (a,b),box in zip(spans,boxes)]
        if any(m is None for m in matches):continue
        sizes=[m['size'] for m in matches];scores=[m['score'] for m in matches]
        verified.append({'score':float(np.mean(scores)),'minimum':min(scores),
                         'size_ratio':max(sizes)/min(sizes),'spans':spans,'font':family,'matches':matches})
    audit={'bands':[[b[0]+offset[0],b[1]+offset[1],b[2]+offset[0],b[3]+offset[1]] for b in boxes],
           'candidates':verified,'status':'unverified_partition'}
    if not verified:return None,audit
    best=max(verified,key=lambda p:p['score'])
    rivals=[v for v in verified if v['spans']!=best['spans']]
    if best['minimum']<.70 or best['score']<.80 or best['size_ratio']>1.3:return None,audit
    if rivals and best['score']-max(v['score'] for v in rivals)<.04:
        return None,{**audit,'status':'ambiguous_partition'}
    children=[]
    for index,((start,end),m) in enumerate(zip(best['spans'],best['matches']),1):
        child=deepcopy(row)
        for key in ('geometry_prior_bbox','glyph_registration'):child.pop(key,None)
        child.update(id=f'{row["id"]}.B{index:02d}',parent_id=row.get('parent_id',row['id']),
                     text=row['text'][start:end],runs=slice_runs(row['runs'],start,end),font=best['font'],
                     bbox_2d=[v+offset[i%2] for i,v in enumerate(m['bbox'])],rotation_deg=0,
                     pixel_reflow={'parent':row['id'],'character_span':[start,end],'score':m['score']})
        children.append(child)
    assert ''.join(''.join(c['text'].split()) for c in children)==''.join(row['text'].split())
    return children,{**audit,'status':'accepted','selected':best}


def propose(row,image):
    children,audit=_propose(row,image,True)
    audit['band_policy']='complete_components'
    if children or audit['status'].startswith('ineligible'):return children,audit
    # A glyph component can touch a neighboring graphic or colored fill.
    # Dense baseline cores offer a second hypothesis, never weaker acceptance
    # thresholds: the entire transcribed line must still match rendered ink.
    alternative,second=_propose(row,image,False)
    second['band_policy']='dense_core_verified_by_complete_template'
    if alternative:
        second['rejected_complete_components']=audit
        return alternative,second
    audit['dense_core_status']=second['status']
    return None,audit


def refine(rows,source,folder):
    from .local_content import collisions
    output=[];audit=[];cv2.setNumThreads(1);groups={};consumed=set()
    for row in rows:
        if row.get('parent_id') and row.get('parent_geometry_prior_bbox'):
            groups.setdefault(row['parent_id'],[]).append(row)
    with Image.open(source) as image:
        for row in rows:
            if row['id'] in consumed:continue
            members=groups.get(row.get('parent_id'),[row])
            if len(members)>1 and all(r.get('kind')!='formula' for r in members):
                members=sorted(members,key=lambda r:r['id'])
                candidate=deepcopy(members[0]);runs=[]
                for member in members:
                    defaults={k:member.get(k,False) for k in ('bold','italic','underline')}
                    if runs:runs.append({'text':' ','script':'normal','color':member['color'],**defaults})
                    # Reparenting a run must not change its inherited style.
                    runs.extend({**defaults,**deepcopy(run)} for run in member['runs'])
                candidate.update(id=row['parent_id'],runs=runs,text=''.join(r['text'] for r in runs),
                                 bbox_2d=row['parent_geometry_prior_bbox'])
            else:members=[row];candidate=row
            member_ids={r['id'] for r in members}
            children,record=propose(candidate,image)
            if children:
                others=[r for r in rows if r['id'] not in member_ids and r['id'] not in consumed]
                new_ids={r['id'] for r in children}
                if any(new_ids.intersection(pair) for pair in collisions([*output,*others,*children])):
                    children=None;record['status']='rejected_identity_collision'
            if children:
                if [r['text'] for r in children]==[r['text'] for r in members]:
                    for child,member in zip(children,members):child['id']=member['id']
                consumed.update(member_ids);output.extend(children)
            else:
                # A failed parent-level repair must not suppress valid children.
                consumed.update(member_ids);output.extend(members)
            audit.append({'id':candidate['id'],'input_ids':sorted(member_ids),**record})
    save(folder/'pixel_reflow.json',audit)
    return output
