"""Backend frames, bounded work packets, explicit identity/ownership ledger."""
from collections import defaultdict
import copy
import math
from protocol import NETWORKS


def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def overlap(a,b):return intersection(a,b)/max(1e-6,min(area(a),area(b)))
def inside(child,parent):return intersection(child,parent)/max(1e-6,area(child))
def gap(a,b):return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))
def union(boxes):return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]


def crop_box(boxes,size,pad=48):
 b=union(boxes)
 return [max(0,math.floor(b[0]-pad)),max(0,math.floor(b[1]-pad)),min(size[0],math.ceil(b[2]+pad)+1),min(size[1],math.ceil(b[3]+pad)+1)]


def batches(rows,size,cfg):
 index={r['id']:r for r in rows}
 if len(index)!=len(rows):raise ValueError('Observation IDs must be unique')
 remaining=set(index);out=[]
 def hard(cid):
  r=index[cid];return r['kind'] in NETWORKS|{'dense_region'} or area(r['bbox_native'])>size[0]*size[1]*.08
 def order(cid):
  b=index[cid]['bbox_native'];return ((b[1]+b[3])/2,(b[0]+b[2])/2,cid)
 while remaining:
  first=min(remaining,key=order);focus=[first];remaining.remove(first)
  limit=cfg['complex_batch_max'] if hard(first) else cfg['simple_batch_max']
  while remaining and len(focus)<limit:
   current=union([index[k]['bbox_native'] for k in focus]);candidates=[k for k in remaining if hard(k)==hard(first)]
   if not candidates:break
   nxt=min(candidates,key=lambda k:(gap(current,index[k]['bbox_native']),area(union([current,index[k]['bbox_native']])),order(k)))
   b=union([current,index[nxt]['bbox_native']])
   if gap(current,index[nxt]['bbox_native'])>cfg['neighbor_gap_native_px']:break
   if not hard(first) and max(b[2]-b[0],b[3]-b[1])>cfg['simple_batch_span_native_px']:break
   if hard(first) and overlap(current,index[nxt]['bbox_native'])<.3:break
   focus.append(nxt);remaining.remove(nxt)
  box=crop_box([index[k]['bbox_native'] for k in focus],size,cfg['context_padding_native_px'])
  others=[k for k in index if k not in focus and gap(index[k]['bbox_native'],box)==0]
  def context_rank(k):
   parent=index[k]['kind'] in NETWORKS and any(inside(index[f]['bbox_native'],index[k]['bbox_native'])>.5 for f in focus)
   return (not parent,min(gap(index[k]['bbox_native'],index[f]['bbox_native']) for f in focus),order(k))
  others.sort(key=context_rank)
  out.append({'batch_id':f'B{len(out)+1:04d}','focus_ids':focus,'context_ids':others[:cfg['context_candidate_max']],'crop':box})
 assert sorted(x for b in out for x in b['focus_ids'])==sorted(index)
 return out


def pending_object(row,reason,prefix='PENDING'):
 return {**row,'id':prefix+'/'+row['id'],'source_ids':row.get('source_ids',[row['id']]),'bbox_source':'original',
         'uncertain':True,'reason':reason,'sam_ids':[],'needs_split':row['kind']=='dense_region'}


def make_pairs(objects):
 pairs=[]
 for i,a in enumerate(objects):
  for b in objects[i+1:]:
   x,y=a['bbox_native'],b['bbox_native']
   # Selection only: no identity is merged by these numerical predicates.
   if overlap(x,y)<.15 and gap(x,y)>8:continue
   if set(a['source_ids'])&set(b['source_ids']):continue # preserve an explicit split
   pairs.append({'id':f'P{len(pairs)+1:05d}','a':a['id'],'b':b['id']})
 return pairs


def pair_batches(pairs,index,max_pairs=6,max_objects=10):
 remaining=list(pairs);out=[]
 while remaining:
  current=[remaining.pop(0)];ids={current[0]['a'],current[0]['b']}
  while remaining and len(current)<max_pairs:
   box=union([index[k]['bbox_native'] for k in ids])
   candidates=[(i,p) for i,p in enumerate(remaining) if len(ids|{p['a'],p['b']})<=max_objects]
   if not candidates:break
   i,p=min(candidates,key=lambda ip:gap(box,union([index[ip[1]['a']]['bbox_native'],index[ip[1]['b']]['bbox_native']])))
   if gap(box,union([index[p['a']]['bbox_native'],index[p['b']]['bbox_native']]))>96:break
   current.append(remaining.pop(i));ids|={p['a'],p['b']}
  out.append(current)
 return out


def resolve(objects,pairs,decisions):
 """Only explicit same/member relations change responsibility, with conflict guards."""
 objects=copy.deepcopy(objects);index={r['id']:r for r in objects};parent={k:k for k in index};groups={k:{k} for k in index}
 pairs={p['id']:p for p in pairs};by_pair={d['pair_id']:d for d in decisions}
 if len(by_pair)!=len(decisions):raise ValueError('Conflicting duplicate pair decisions')
 distinct={frozenset([p['a'],p['b']]) for pid,p in pairs.items() if by_pair.get(pid,{}).get('relation')=='distinct'}
 conflicts=[];merged=[];claims=[]
 def find(k):
  while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
  return k
 def uncertain(a,b,why):
  index[a]['uncertain']=index[b]['uncertain']=True;conflicts.append({'a':a,'b':b,'reason':why})
 for pid,p in pairs.items():
  a,b=p['a'],p['b'];d=by_pair.get(pid)
  if d is None:uncertain(a,b,'Ownership request absent/failed');continue
  rel=d['relation']
  if rel=='distinct':continue
  if rel=='uncertain':uncertain(a,b,d['reason']);continue
  if rel in {'a_in_b','b_in_a'}:
   claims.append((a,b,d['reason']) if rel=='a_in_b' else (b,a,d['reason']));continue
  ra,rb=find(a),find(b)
  if ra==rb:continue
  combined=groups[ra]|groups[rb]
  sources=[sid for oid in combined for sid in index[oid]['source_ids']]
  if len(sources)!=len(set(sources)):
   uncertain(a,b,'Would undo explicit split');continue
  if index[a]['kind']!=index[b]['kind'] or overlap(index[a]['bbox_native'],index[b]['bbox_native'])<.2:
   uncertain(a,b,'Same-object claim contradicts refined kind/location');continue
  if any(frozenset([x,y]) in distinct for x in groups[ra] for y in groups[rb]):
   uncertain(a,b,'Transitive merge contradicts explicit distinct relation');continue
  parent[rb]=ra;groups[ra]|=groups.pop(rb);merged.append({'a':a,'b':b,'reason':d['reason']})
 final={};alias={}
 def rank(oid):
  r=index[oid];return (r['extent']=='complete',not r['uncertain'],r['bbox_source']=='sam',r['bbox_source']=='model',oid)
 for gid,ids in groups.items():
  best=max(ids,key=rank);r=copy.deepcopy(index[best]);r['id']=best
  r['source_ids']=sorted({sid for oid in ids for sid in index[oid]['source_ids']});r['observation_object_ids']=sorted(ids)
  r['uncertain']=any(index[oid]['uncertain'] for oid in ids);r['source_reasons']={sid:[index[oid]['reason'] for oid in ids if sid in index[oid]['source_ids']] for sid in r['source_ids']}
  final[best]=r
  for oid in ids:alias[oid]=best
 candidates=defaultdict(list)
 for child,owner,reason in claims:
  a,b=alias[child],alias[owner]
  if a==b:continue
  ca,pa=final[a],final[b]
  if pa['kind'] not in NETWORKS or pa['extent']!='complete' or pa['uncertain'] or ca['kind']!='single' or inside(ca['bbox_native'],pa['bbox_native'])<.6:
   final[a]['uncertain']=True;conflicts.append({'a':child,'b':owner,'reason':'Member claim lacks a complete unambiguous network owner'});continue
  candidates[a].append((b,reason))
 absorbed=[]
 for a,owners in candidates.items():
  unique={b for b,_ in owners}
  if len(unique)!=1:
   final[a]['uncertain']=True;conflicts.append({'a':a,'reason':'Multiple parent network claims'});continue
  b=next(iter(unique));child=final[a];owner=final[b]
  absorbed.append({'member':copy.deepcopy(child),'owner_id':b,'reasons':[r for _,r in owners]})
  owner['source_ids']=sorted(set(owner['source_ids'])|set(child['source_ids']))
  owner.setdefault('member_ids',[]).append(a);owner['source_reasons'].update(child['source_reasons'])
 for row in absorbed:del final[row['member']['id']]
 result=sorted(final.values(),key=lambda o:(o['bbox_native'][1],o['bbox_native'][0],o['id']))
 for i,row in enumerate(result,1):row['display_id']=f'I{i:04d}'
 assert {s for r in objects for s in r['source_ids']}=={s for r in result for s in r['source_ids']}
 return {'objects':result,'merges':merged,'absorbed_members':absorbed,'conflicts':conflicts,'decisions':decisions}


def ledger(observations,result,removed):
 removed={r['source_id']:r for r in removed};out=[]
 for old in observations:
  sid=old['id'];targets=[o for o in result['objects'] if sid in o['source_ids']]
  if targets:
   out.append({'source_id':sid,'status':'split' if len(targets)>1 else 'pending' if targets[0]['uncertain'] or targets[0]['extent']=='partial' or targets[0]['kind']=='dense_region' else 'assigned',
               'owner_ids':[o['id'] for o in targets],'before_bbox':old['bbox_native'],'after_bboxes':[o['bbox_native'] for o in targets],
               'reasons':[reason for o in targets for reason in o.get('source_reasons',{}).get(sid,[o['reason']])]})
  elif sid in removed:out.append({'source_id':sid,'status':'excluded','owner_ids':[],'before_bbox':old['bbox_native'],'reasons':[removed[sid]['reason']]})
  else:raise AssertionError('Silently lost observation: '+sid)
 return out
