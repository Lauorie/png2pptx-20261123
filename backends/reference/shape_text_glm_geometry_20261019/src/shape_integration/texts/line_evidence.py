"""Conservative pixel evidence for multiple baselines inside a plain-text box."""
import math
import numpy as np


def line_bands(row, image):
    if row.get('kind')=='formula' or abs(row.get('rotation_deg',0))>5:return []
    if len({r.get('color',row['color']) for r in row.get('runs',[])})>1:return []
    b=row['bbox_2d'];x0=max(0,math.floor(b[0]));y0=max(0,math.floor(b[1]))
    x1=min(image.width,math.ceil(b[2])+1);y1=min(image.height,math.ceil(b[3])+1)
    if x1-x0<12 or y1-y0<16:return []
    rgb=np.asarray(image.crop((x0,y0,x1,y1)).convert('RGB'),dtype=np.float32)
    fg=np.array([int(row['color'][i:i+2],16) for i in (1,3,5)],dtype=np.float32)
    bg=np.percentile(rgb.reshape(-1,3),15 if fg.mean()>180 else 85,axis=0)
    v=bg-fg;den=float(v@v)
    if den<2500:return []
    mask=((bg-rgb)@v)/den>.45
    ys=np.flatnonzero(mask.sum(axis=1)>=max(3,.12*mask.shape[1]))
    if not len(ys):return []
    groups=np.split(ys,np.flatnonzero(np.diff(ys)>max(1,round(mask.shape[0]*.04))+1)+1)
    tallest=max(g[-1]-g[0]+1 for g in groups)
    bands=[]
    for g in groups:
        a,z=int(g[0]),int(g[-1])+1
        if z-a<max(4,tallest*.4):continue  # dots, accents and underlines are not lines
        yy,xx=np.where(mask[a:z])
        if len(xx)<max(10,mask.sum()*.15) or xx.max()-xx.min()<mask.shape[1]*.30:continue
        bands.append([int(x0+xx.min()),y0+a,int(x0+xx.max()),y0+z-1])
    if len(bands)<2:return []
    if any(bands[i+1][1]-bands[i][3]<max(3,.15*tallest) for i in range(len(bands)-1)):return []
    return bands
