"""Small semantic contracts for discovery, bbox refinement and ownership."""
import math

KINDS={'single','serial_chain','shared_network','dense_network','dense_region'}
NETWORKS={'serial_chain','shared_network','dense_network'}
SYSTEM='Inspect actual scientific-figure pixels. Resolve physical arrow identity before geometry. All coordinates are absolute pixels of TARGET; backend performs coordinate transforms. Return compact JSON only.'
RULES=[
 '普通箭头/独立无头线一支一身份single，包含整条直线/曲线/折线及箭尖。彼此独立但相似、同色、同起终模块或排列紧密不自动合组。可分开的重复箭头逐支定位，不推算数量和间距。',
 '连续多头连接允许作为serial_chain用一个框包住整条杆、所有头、转弯和末端无头尾段；发现阶段只负责完整范围，后续几何再按头分段。不同并行/嵌套连接仍分别框出。',
 'shared_network只用于实际共享可见主干/分叉/连接点的网络，整网负责所有分支且共享段只恢复一次。仅交叉、碰到同一个模块或端点接近不证明共享。',
 'dense_network用于神经网络的一层间连线/明确子网络，以及密集平行示意线束。一个有意义的层/一束作为整体，不按内部单边拆开。内部水平边、斜边均归同一网络，不再另发独立重建任务；网络外部的输入/输出箭头独立。',
 '大量相似但方向/路径复杂的箭头，尽量分为single。实在无法分清才暂记dense_region，框全并留待拆分；它不是已确定的一个箭头或已完成网络，不猜成员数量。',
 '分子结构整体处理，仅发现其外部连线。真实框内连接、坐标轴、标注线等按scope保留；普通封闭形状的轮廓、单元格边、文字笔画不属于箭头。',
 'bbox只表示该身份可见范围；context裁剪只是观察窗口，不等于重建责任。不能把一小段/一支箭头因看见整排而扩成整排任务；若实际是残缺的同一箭头，可补全并明确依据。',
 'TARGET含任意真实笔画即应发现；若完整对象跨出TARGET，extent=partial并只框可见部分，后台扩图。不要编造图外坐标。只做身份与bbox，不输出路径点、线宽、颜色或头型测量。',
]

def discovery(scope,size,view):
 return {'task':'发现TARGET中的全部箭头/独立线条及合理网络组；先空间扫描，再反向检查边界和末端遗漏。GLOBAL只帮助语义理解。',
         'scope':scope,'rules':RULES,'target_size_px':list(size),'view':view,
         'output_schema':{'objects':[{'id':'A1','kind':'single|serial_chain|shared_network|dense_network|dense_region',
                                    'bbox':[0,0,10,10],'locator':'短位置与连接描述','extent':'complete|partial'}]}}

def refinement(scope,size,focus,context,evidence,region,expansion=False):
 return {'task':('扩大的观察窗口中补全focus的同一物理对象；保持原身份含义。' if expansion else '校正focus候选的身份与bbox，合并重复观察，拆开可区分箭头，排除非范围内容。'),
         'scope':scope,'rules':RULES+[
          'GLOBAL和TARGET是干净原图。BOXES、SAM图中的标号/颜色只是证据。旧描述和旧框可能偏移。只编辑focus，每个focus必须有objects或removed或unresolved去向。context只读。',
          'objects.source_ids只引用focus：多个来源可合为一身份，同一来源可拆成多个真实身份。即使框相交也可能是两条不同线。',
          'bbox_source=sam时必须确认所选mask完整且仅支持本对象，complete_support=true；后台用mask真实像素计算bbox。混合、共享、不完整mask只能作参考，不能直接套用为多个完整single。',
          'bbox_source=model输出TARGET绝对像素bbox；bbox_source=original指定original_id保留该候选框。没有SAM不能据此删除。',
          '如果原框偏移到空白/文字但其描述的真实箭头可在附近定位，应纠正而非仅删除。确为形状边框/字形/没有对应真实对象才removed。',
          '不能仅用被另一大框包含作为删除理由。若网络成员归属涉及context，先保留本对象与具体说明，后续统一归属审核；不把独立连接吞入网络。',
          'SAM未被GLM发现的候选只是线状证据，可能是框边或噪声；要对照完整原图判断。不要为了采用SAM而虚构连接。',
          '不新增与focus无关的目标。复杂同一网络可使用完整model框，不要求用所有内部mask列举细线。理由一短句即可。'],
         'target_size_px':list(size),'target_region_in_global_px':region,
         'focus':focus,'context':context,'sam_evidence':evidence,
         'output_schema':{'objects':[{'id':'O1','source_ids':['focus ID'],'kind':'single|serial_chain|shared_network|dense_network|dense_region',
           'locator':'短身份描述','extent':'complete|partial','bbox_source':'sam|model|original','bbox':None,'original_id':None,
           'sam_ids':[],'complete_support':False,'uncertain':False,'reason':'短视觉依据'}],
           'removed':[{'source_id':'focus ID','reason':'明确非范围理由'}], 'unresolved':[{'source_id':'focus ID','reason':'歧义依据'}]}}

def relation_prompt(scope,objects,pairs):
 return {'task':'仅审核给定pair的身份归属，不重新定位，不生成路径；不同ID不一定不同对象，相交框不一定重复。',
         'scope':scope,'rules':RULES+[
          'same表示同一支完整箭头或同一个网络的重复观察，可选择较完整代表；相邻重复箭头是distinct。部分观察与其对应完整对象可same。',
          'a_in_b表示A的实际线条全属于B负责的真实shared_network/dense_network；b_in_a反之。网络内部水平边与斜边都属成员，不能只把斜线算网络。',
          '仅bbox包含/交叉/同色不能判断归属。外接输出箭头、独立引线及穿过网络的独立连接须distinct。dense_region尚未拆清，不能作为确定归属的父对象。',
          'same与网络成员关系不同；两个只有部分共享路径但范围不同的网络若无法确认一致，给uncertain，避免把路径拼成大网络。',
          '每个pair必须回答一次；无法确认用uncertain，后台保留。reason只写一短句可见依据。'],
         'objects':objects,'pairs':pairs,
         'output_schema':{'decisions':[{'pair_id':'P1','relation':'same|a_in_b|b_in_a|distinct|uncertain','reason':'短依据'}]}}

def box(value,size):
 if not isinstance(value,list) or len(value)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):raise ValueError('Four finite absolute pixel coordinates required')
 if not (-1<=value[0]<value[2]<=size[0] and -1<=value[1]<value[3]<=size[1]):raise ValueError('BBox outside TARGET')
 return value

def validate_discovery(answer,size):
 if not isinstance(answer,dict) or not isinstance(answer.get('objects'),list):raise ValueError('objects array required')
 seen=set()
 for row in answer['objects']:
  if not isinstance(row.get('id'),str) or not row['id'] or row['id'] in seen:raise ValueError('Unique object IDs required')
  seen.add(row['id']);box(row.get('bbox'),size)
  if row.get('kind') not in KINDS or row.get('extent') not in {'complete','partial'} or not str(row.get('locator','')).strip():raise ValueError('kind, extent, locator required')
 return answer

def validate_refine(answer,size,focus,sam_ids):
 if not isinstance(answer,dict) or any(not isinstance(answer.get(k),list) for k in ['objects','removed','unresolved']):raise ValueError('objects/removed/unresolved required')
 focus=set(focus);sam_ids=set(sam_ids);seen=set();ids=set();mask_owners=set()
 for row in answer['objects']:
  oid=row.get('id');refs=row.get('source_ids')
  if not isinstance(oid,str) or not oid or oid in ids:raise ValueError('Unique output IDs required')
  ids.add(oid)
  if not isinstance(refs,list) or not refs or len(refs)!=len(set(refs)) or not set(refs)<=focus:raise ValueError('Actual focus source_ids required')
  seen.update(refs)
  if row.get('kind') not in KINDS or row.get('extent') not in {'complete','partial'} or type(row.get('uncertain')) is not bool:raise ValueError('kind/extent/uncertain required')
  if not str(row.get('locator','')).strip() or not str(row.get('reason','')).strip():raise ValueError('locator/reason required')
  masks=row.get('sam_ids')
  if not isinstance(masks,list) or not set(masks)<=sam_ids:raise ValueError('Actual SAM IDs required')
  source=row.get('bbox_source')
  if source=='sam':
   if not masks or row.get('complete_support') is not True:raise ValueError('SAM extent needs complete unmixed support')
   if set(masks)&mask_owners:raise ValueError('A complete mask cannot own two objects')
   mask_owners.update(masks)
  elif source=='model':box(row.get('bbox'),size)
  elif source=='original':
   if row.get('original_id') not in refs:raise ValueError('original_id must be a source')
  else:raise ValueError('Unknown bbox_source')
 for key in ['removed','unresolved']:
  for row in answer[key]:
   sid=row.get('source_id')
   if sid not in focus or sid in seen or not str(row.get('reason','')).strip():raise ValueError('Exclusive source disposition required')
   seen.add(sid)
 if seen!=focus:raise ValueError('Unaccounted focus IDs: '+str(sorted(focus-seen)))
 return answer

def validate_relations(answer,pairs):
 if not isinstance(answer,dict) or not isinstance(answer.get('decisions'),list):raise ValueError('decisions array required')
 expected={p['id'] for p in pairs};seen=set()
 for row in answer['decisions']:
  if row.get('pair_id') not in expected or row['pair_id'] in seen:raise ValueError('Unique supplied pair ID required')
  seen.add(row['pair_id'])
  if row.get('relation') not in {'same','a_in_b','b_in_a','distinct','uncertain'} or not str(row.get('reason','')).strip():raise ValueError('relation/reason required')
 if seen!=expected:raise ValueError('Every supplied pair needs a decision')
 return answer
