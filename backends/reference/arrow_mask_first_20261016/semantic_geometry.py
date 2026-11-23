"""One identity/structure request per observation group; masks are evidence."""
import copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import contracts,protocol,visual,mask_vectors
from transport.common import read_json as read,write_json as write,sha256 as sha


def prepare(crop,group,candidates,out):
    proposals={};info=[]
    for candidate in candidates:
        diagnostic=None
        try:
            # A mixed/bent mask must not become a fictitious continuous route
            # before the model decides identity. Only direct straight estimates
            # are offered as optional measurements.
            row=mask_vectors.straight_mask.estimate(candidate['mask'])
            if row is not None:row.pop('route_path',None)
            if row is not None:
                p=np.asarray([row['start'],*row.get('via',[]),row['end']])
                if not np.isfinite(p).all() or (p<-.5).any() or (p>np.asarray(crop.size)-.5).any():
                    diagnostic='Numerical proposal outside image; raw mask retained';row=None
        except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
            row=None;diagnostic='Numerical proposal unavailable: '+str(exc)
        pid=None
        if row is not None:
            pid=f'D{len(proposals)+1}';row.update(id=pid,masks=[candidate['id']]);proposals[pid]=row
        y,x=np.nonzero(candidate['mask'])
        from scipy.ndimage import label
        components=int(label(candidate['mask'],np.ones((3,3)))[1])
        info.append(dict(id=candidate['id'],bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None,
                         components=components,
                         global_overlap=[k for k,v in candidate.get('scores',{}).items() if v>0],proposal=pid))
        if diagnostic:info[-1]['measurement_note']=diagnostic
    visual.masks(crop,[(r['id'],r['mask']) for r in candidates],out/'input/masks.png')
    image=Image.open(out/'input/masks.png').convert('RGB');draw=ImageDraw.Draw(image)
    for i,(key,row) in enumerate(proposals.items()):
        p=np.asarray([row['start'],*row.get('via',[]),row['end']]);c=visual.color(i)
        draw.line([tuple(q) for q in p],fill=c,width=2)
        for label,q in [('a',p[0]),('b',p[-1])]:visual.label(draw,q,key+label,c,image)
    image.save(out/'input/proposals.png');write(out/'proposals.json',proposals)
    return proposals,info


def prompt(labels,size,proposals,candidates,description,context_size=None,context_bbox=None):
    hints=[dict(id=k,mask=r['masks'][0],kind=r['kind'],points=[
        [round(float(x),1),round(float(y),1)] for x,y in [r['start'],*r.get('via',[]),r['end']]]) for k,r in proposals.items()]
    return dict(task='Identify the visible connectors represented by the supplied discovery evidence, then choose mask or GLM geometry.',
        sources=labels,size=list(size),context_size=list(context_size) if context_size else None,
        description=description,crop_in_context=context_bbox,masks=candidates,proposals=hints,head_types=protocol.HEADS,
        instructions=[protocol.SCOPE,
            'CLEAN is the local pixel frame; CONTEXT is the whole figure. crop_in_context gives the exact CLEAN rectangle in CONTEXT pixels; its magenta outline is a locator annotation. Use visible strokes and attachments without explaining the figure architecture. G names coarse discovery evidence, not confirmed arrow identities. A mixed G may contain several independent connectors: emit each as a separate route and repeat that G in sources where appropriate. Several G may also support one route. The counts of G, M and D never prescribe the number of physical routes. Account for every G as used, excluded or uncertain, but accounting for all G does not prove all their visible connectors were recovered. M masks and D straight estimates are optional measurements, not a required one-to-one correspondence; inspect CLEAN first. Keep visible eligible connectors in the supplied target extents even without a clean single-line M/D: use geometry=glm with masks=[] if necessary. Ignore unrelated crop neighbors.',
            'geometry=mask: use good evidence for an independent straight line, broad smooth arc or filled block arrow. Use a matching D with nodes=[] only when it covers the complete target. Otherwise supply the complete target endpoints and a few on-curve guide nodes, even when a mask supports only the head or a fragment. A D spanning extra objects is not a match. Supporting masks measure geometry and style; their bounds do not define the target extent.',
            'geometry=glm: use for sharp multi-leg routes, mixed straight/curved parts, shared networks, or unusable masks. Set proposal=null,nodes=[]; the next request fits geometry. locator is one short visual identifier using appearance and visible attachment objects, so the next request can find the intended ink. Do not put estimated coordinates, turn sequences, panel-border alignment, or inferred architecture in locator. Geometry is determined from CLEAN in the next request. Sharp corners stay sharp; visible rounded corners stay curved.',
            'A filled block arrow without a usable mask may use geometry=points with its ordered closed contour nodes and heads none/none; keep its filled identity. An independent simple line may also use points when you can directly give its complete geometry and masks cannot measure it. Those points are not subsequently fitted to the mask.',
            'Related routes with a true junction use the same network string and geometry=glm; crossings use network=null. Draw a shared trunk once. kind is straight/polyline/curve/block_arrow. An ordinary shaft with a head is not a filled block arrow.',
            'Heads are [first,last] using head_types. For a D proposal this means [Da,Db] in the shown/numeric point order, NOT the logical source-to-destination order: a leftward or upward arrow may have its head at Da. For your own nodes use their first and last IDs. Judge the actual rear edge of a head in CLEAN: flat=triangle, concave filled notch=stealth, two unfilled strokes=arrow. Internal junctions have none; unclear ends may be unknown. line_style is solid/dashed/dotted/unknown. Do not estimate width or dash lengths. Mask-mode nodes use local pixels, one decimal; never one node per dash.',
            'Only a continuation of the SAME physical connector or a branch at a TRUE shared junction belongs in followups; crossing lines and unrelated neighbors do not. If such a related branch or continuation is outside this G target or CLEAN, do not assume another group handles it. Give a tight bbox around the suspected missing portion and its attachments in CONTEXT pixels, plus a short identifying description. Later repair expands this bbox for context and checks all groups actual paths before editing; it may find nothing missing. Do not guess branch counts or coordinates in the description. followups=[] for complete isolated targets or unrelated neighbors.'],
        output_schema=dict(nodes='[[id,x,y],...] or []',routes='[{id,sources:[G IDs],masks:[M IDs],kind,proposal:D ID or null,nodes:[IDs],geometry:mask|points|glm,network:null or ID,locator:short visual identifier,heads:[first,last],line_style}]',
            excluded='[G IDs]',uncertain='[G IDs]',followups='[{context_bbox:[x0,y0,x1,y1],description}] or []'))


def validate(raw,size,labels,proposals,mids,candidates=(),context_size=None):
    answer=copy.deepcopy(raw)
    if not isinstance(answer,dict) or not isinstance(answer.get('nodes'),list) or not isinstance(answer.get('routes'),list):raise ValueError('nodes and routes lists required')
    followups=answer.setdefault('followups',[])
    if not isinstance(followups,list):raise ValueError('followups must be a list')
    for item in followups:
        if context_size is None:raise ValueError('Followup needs CONTEXT coordinates')
        contracts.bbox(item.get('context_bbox'),context_size)
        if not isinstance(item.get('description'),str) or not item['description'].strip():raise ValueError('Describe the related incomplete connection')
    names={str(n[0]) for n in answer['nodes']}
    for i,r in enumerate(answer['routes']):
        if 'locator' in r and not isinstance(r['locator'],str):raise ValueError('locator must be a short visual identifier string')
        if 'shaft_sample' in r:
            import shaft_sampling
            shaft_sampling.validate(r['shaft_sample'],size)
        if r.get('geometry') not in ('mask','points','glm'):raise ValueError('geometry must be mask, glm or legacy points')
        network=r.get('network')
        if network is not None and (not isinstance(network,str) or not network):raise ValueError('network must be null or a nonempty ID string')
        masks=contracts.ids(r.get('masks',[]))
        if not set(masks)<=set(mids):raise ValueError('Unknown M mask')
        r['masks']=masks
        if r['geometry']=='glm':
            if r['kind']=='block_arrow':raise ValueError('A filled block contour uses geometry=mask')
            if r.get('proposal'):raise ValueError('Deferred GLM geometry does not use a D proposal')
            continue
        if r['kind']=='block_arrow':
            heads=r.get('heads')
            if not isinstance(heads,list) or len(heads)!=2 or any(h not in protocol.HEADS for h in heads):raise ValueError('Two valid head enums required')
            r.setdefault('semantic_heads',heads)
            # A filled contour already includes its head pixels. Keep the model's
            # semantic heads separately; do not draw a second marker on the fill.
            r['heads']=['none','none']
            if r['geometry']=='mask' and masks:
                index={c['id']:c['mask'] for c in candidates}
                if not set(masks)<=set(index):raise ValueError('Selected contour mask data unavailable')
                shape=mask_vectors.contour(np.logical_or.reduce([index[m] for m in masks]))
                p=[shape['start'],*shape['via'],shape['end']]
                generated=[f'_M{i}_{j}' for j in range(len(p))]
                existing={str(n):[x,y] for n,x,y in answer['nodes']}
                if r.get('nodes')!=generated:
                    if names.intersection(generated):raise ValueError('Contour node name collision')
                    answer['nodes'] += [[n,*q] for n,q in zip(generated,p)];names.update(generated)
                elif any(not np.allclose(existing[n],q,rtol=0,atol=1e-8) for n,q in zip(generated,p)):
                    raise ValueError('Cached contour evidence changed')
                r['nodes']=generated;r['proposal']=None
                r.setdefault('line_style','solid')
                continue
        if r.get('proposal'):
            key=r['proposal']
            if key not in proposals:raise ValueError('Unknown D proposal')
            p=proposals[key]
            if r.get('nodes') and r['geometry']=='points':
                # Explicit model coordinates take precedence over an incidental
                # D reference; retain its mask as associated evidence.
                r['proposal']=None
                if not r['masks']:r['masks']=p['masks']
                continue
            if r['geometry']!='mask':raise ValueError('A D proposal uses geometry=mask; points mode needs your own nodes')
            points=[p['start'],*p.get('via',[]),p['end']]
            if r['kind']!=p['kind']:
                # The model owns the geometric family. A declared straight
                # route only needs the displayed two ends, even if the noisy
                # numerical proposal was curved. Other kinds need actual bends.
                if r.get('nodes'):
                    r['proposal']=None
                    if not r['masks']:r['masks']=p['masks']
                    continue
                if r['kind']=='straight':
                    points=[p['start'],p['end']];r['proposal']=None
                    r['measurement_seed']=key
                else:raise ValueError('The requested kind needs bend/curve nodes; keep your kind, set proposal=null and supply those nodes')
            indices=np.linspace(0,len(points)-1,min(7,len(points))).round().astype(int)
            expected=[f'_D{i}_{j}' for j in range(len(indices))]
            existing={str(n):[x,y] for n,x,y in answer['nodes']}
            if r.get('nodes')==expected and all(n in existing and np.allclose(existing[n],points[k],rtol=0,atol=1e-8)
                                               for n,k in zip(expected,indices)):
                # Cached results have already expanded D references. Validation
                # must be idempotent; do not manufacture a name collision.
                if not r['masks']:r['masks']=p['masks']
                continue
            r['nodes']=[]
            for j,k in enumerate(indices):
                name=f'_D{i}_{j}'
                if name in names:raise ValueError('Reserved proposal node name collision')
                names.add(name);r['nodes'].append(name);answer['nodes'].append([name,*points[k]])
            if not r['masks']:r['masks']=p['masks']
    return contracts.graph(answer,size,labels,True,require_line_styles=True,allow_deferred=True)
