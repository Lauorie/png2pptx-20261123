"""Recognize a complete dashed shaft and its stroked heads before filtering.

The axis is evidence indexing only; export always retains the original native
path, including every dash and head. Networks and internal grids abstain.
"""
import numpy as np


def terminal_groups(lines,width=1.):
    """Disjoint shaft/head pairs inside one multi-route native path.

    Only unique exact terminal attachments bind a pair. Independent paths keep
    their own subpath indices; no proximity grouping or whole-network merge.
    """
    matches=[]
    for i,a in enumerate(lines):
        for j,b in enumerate(lines[i+1:],i+1):
            if len(a)<3 and len(b)<3:continue
            found=recognize([a,b],width)
            if found is not None:matches.append((i,j,found))
    counts={i:sum(i in (a,b) for a,b,_ in matches) for i in range(len(lines))}
    return [(a,b,found) for a,b,found in matches if counts[a]==counts[b]==1]


def terminal(lines, width):
    """Index a native shaft whose two short wings share its terminal tip.

    One wing may continue the shaft's subpath and the other start after M.
    Only the evidence axis is trimmed; the native object stays indivisible.
    """
    if len(lines)!=2:return None
    found=[];tolerance=max(.25,.05*width)
    # Encoding order and total V-head length do not identify the shaft.
    for body,head in ((0,1),(1,0)):
        short=np.asarray(lines[head],float)
        for long in (np.asarray(lines[body],float),np.asarray(lines[body],float)[::-1]):
            prefix=np.r_[0,np.cumsum(np.linalg.norm(np.diff(long,axis=0),axis=1))]
            for i in range(1,len(long)):
                if prefix[i]<=0:continue
                tip=long[i];j=int(np.argmin(np.linalg.norm(short-tip,axis=1)))
                error=float(np.linalg.norm(short[j]-tip))
                if error>tolerance:continue
                # Attachment follows the local incoming tangent of a straight,
                # bent or curved shaft; its native geometry is never changed.
                axis=tip-long[i-1];norm=float(np.linalg.norm(axis));length=prefix[i]
                if norm<=0:continue
                axis/=norm
                arms=([long[i+1]-tip]if i+1<len(long)else [])
                arms.extend(short[k]-tip for k in (j-1,j+1)if 0<=k<len(short))
                if len(arms)!=2 or any(float(v@axis)>=0 for v in arms):continue
                if float(np.cross(axis,arms[0])*np.cross(axis,arms[1]))>=0:continue
                if any(np.linalg.norm(v)>=length for v in arms):continue
                found.append((error,-prefix[i],long[:i+1].copy()))
    if not found:return None
    points=min(found,key=lambda item:item[:2])[2]
    return {'points':points,'heads':['none','open'],
            'attachment_bounds':[np.concatenate(lines).tolist()],
            'compound_stroked_route':True,'attached_stroked_head':True,'encoded_dash':False}


def recognize(lines, width=1.):
    if len(lines)==2:
        found=terminal(lines,width)
        if found is not None:return found
        # Some path exporters encode two diverging terminal wings from the
        # same neck, with one appended to the shaft. The shared neck and
        # opposite local arms bind ownership even though no tip is shared.
        for body,wing in ((lines[0],lines[1]),(lines[1],lines[0])):
            if len(body)<3 or len(wing)!=2:continue
            for long in (np.asarray(body,float),np.asarray(body,float)[::-1]):
                neck=long[-2];axis=neck-long[-3];norm=np.linalg.norm(axis)
                if norm<=0:continue
                axis/=norm;length=np.linalg.norm(np.diff(long[:-1],axis=0),axis=1).sum()
                nearest=int(np.argmin(np.linalg.norm(np.asarray(wing)-neck,axis=1)))
                if np.linalg.norm(np.asarray(wing)[nearest]-neck)>max(.25,.05*width):continue
                arms=[long[-1]-neck,np.asarray(wing)[1-nearest]-neck]
                along=np.array([v@axis for v in arms]);cross=np.array([np.cross(axis,v) for v in arms])
                if (np.any(along<=0) or cross.prod()>=0 or
                    max(np.linalg.norm(v) for v in arms)>.2*length or
                    max(along)/min(along)>1.5 or max(abs(cross))/min(abs(cross))>2):continue
                tip=neck+along.mean()*axis
                return {'points':np.vstack([long[:-1],tip]),'heads':['none','open'],
                    'attachment_bounds':[np.concatenate(lines).tolist()],
                    'compound_stroked_route':True,'attached_stroked_head':True,'encoded_dash':False}
        return None
    if len(lines)<3 or any(len(p)!=2 for p in lines):
        return None
    xy=np.concatenate(lines); center=xy.mean(0)
    vals, axes=np.linalg.eigh(np.cov(xy.T))
    if vals[-1]<=0 or vals[0]>.035*vals[-1]:return None
    axis=axes[:,-1];normal=np.array([-axis[1],axis[0]])
    along=(xy-center)@axis;cross=(xy-center)@normal
    lo,hi=float(along.min()),float(along.max());span=hi-lo
    if span<4 or np.ptp(cross)>max(3*width,.22*span):return None
    projected=[(np.asarray(p)-center)@axis for p in lines]
    heads=[]
    for side,tip in enumerate((lo,hi)):
        candidates=[]
        for line,t in zip(lines,projected):
            near=int(np.argmin(abs(t-tip)));other=1-near
            if abs(t[near]-tip)>max(.2,.03*span):continue
            across=(np.asarray(line)-center)@normal
            if abs(across[near])>max(width,.06*span):continue
            if abs(across[other]-across[near])<.04*span:continue
            if abs(t[other]-t[near])>.25*span:continue
            candidates.append((np.asarray(line)[near],float(across[other]-across[near])))
        found=any(np.linalg.norm(a-c)<=max(.25,.05*width) and b*d<0
                  for a,b in candidates for c,d in candidates)
        heads.append('open' if found else 'none')
    if all(h=='none' for h in heads):return None
    axial=[]
    for line,t in zip(lines,projected):
        q=(np.asarray(line)-center)@normal
        if np.ptp(q)<=max(.15,.2*width) and abs(q.mean())<=max(.2,.4*width):
            axial.append(sorted(t))
    if not axial:return None
    return {'points':center+np.outer([lo,hi],axis),'heads':heads,
            'attachment_bounds':[xy.tolist()], 'compound_stroked_route':True,
            'encoded_dash':len(axial)>1}
