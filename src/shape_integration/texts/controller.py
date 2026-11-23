"""One full-image OCR pass and optional parallel enlarged-region corrections."""
import copy
import math
import unicodedata
from concurrent.futures import as_completed
from PIL import Image
from ..common import save, sha, now, usage, intersect, overlay
from ..styling.views import crop_transform, resample_crop
from . import PROTOCOL
from . import prompts, contracts, model, layout, export


def frame(source, folder, opt, region=None):
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:
        if region is None:box=[0,0,*im.size];edge=opt['global_long_edge']
        else:
            pad=max(opt['crop_min_padding_px'],max(region[2]-region[0],region[3]-region[1])*opt['crop_padding'])
            box=[max(0,math.floor(region[0]-pad)),max(0,math.floor(region[1]-pad)),
                 min(im.width,math.ceil(region[2]+pad)+1),min(im.height,math.ceil(region[3]+pad)+1)]
            global_scale=opt['global_long_edge']/max(im.size)
            edge=min(opt['crop_max_long_edge'],max(box[2]-box[0],box[3]-box[1])*global_scale*opt['crop_scale'])
        extent=[box[2]-box[0],box[3]-box[1]]
        size=[max(2,round(v*edge/max(extent))) for v in extent]
        scale,offset=crop_transform(box,size)
        resample_crop(im.convert('RGB'),box,size,scale,offset).save(folder/'clean.png')
        result={'box':box,'size':size,'scale':scale,'offset':offset,'source_size':list(im.size),
                'convention':'local=(source-origin+.5)*scale-.5+offset',
                'sizing_basis':'global input pixels; local crop uses crop_scale capped by crop_max_long_edge'}
    save(folder/'frame.json',result);return result


def mapped_box(box, f, inverse=False):
    if inverse:return [(v+.5-f['offset'][i%2])/f['scale']+f['box'][i%2]-.5 for i,v in enumerate(box)]
    return [(v-f['box'][i%2]+.5)*f['scale']-.5+f['offset'][i%2] for i,v in enumerate(box)]


def local_hints(clean, rows, f, path):
    values=[]
    for r in rows:
        if intersect(r['bbox_2d'],f['box'])<=0:continue
        b=mapped_box(r['bbox_2d'],f)
        b=[max(0,min(v,f['size'][i%2]-1)) for i,v in enumerate(b)]
        values.append({'id':r['id'],'bbox_2d':b})
    with Image.open(clean) as im:overlay(im,values,path)
    return [v['id'] for v in values]


def duplicate(a,b,threshold):
    def key(r):return ''.join(unicodedata.normalize('NFKC',r['text']).split())
    if not key(a) or key(a)!=key(b):return False
    inter=intersect(a['bbox_2d'],b['bbox_2d'])
    area=lambda r:(r['bbox_2d'][2]-r['bbox_2d'][0])*(r['bbox_2d'][3]-r['bbox_2d'][1])
    return inter/max(1e-9,area(a)+area(b)-inter)>=threshold


def merge(rows, incoming, f, origin, history, opt):
    for value in incoming:
        b=mapped_box(value['bbox_2d'],f,True)
        b=[max(0,min(v,f['source_size'][i%2]-1)) for i,v in enumerate(b)]
        row={**copy.deepcopy(value),'bbox_2d':b,'origin':origin,'observations':[{'origin':origin,'local':value}]}
        matches=[i for i,old in enumerate(rows) if old['id'] in row['replaces']]
        if not matches:matches=[i for i,old in enumerate(rows) if duplicate(old,row,opt['duplicate_iou'])]
        if matches:
            index=matches[0];old=rows[index]
            # A lower-quality partial crop must not destroy a readable whole line.
            if old['legibility']=='clear' and row['legibility']!='clear':
                history.append({'action':'keep_clear','id':old['id'],'origin':origin,'candidate':row});continue
            row['id']=old['id'];row['observations']=old['observations']+row['observations'];rows[index]=row
            history.append({'action':'replace','id':row['id'],'origin':origin,'before':old,'after':row})
        else:
            row['id']=f'T{len(rows)+1:04d}';rows.append(row)
            history.append({'action':'add','id':row['id'],'origin':origin})
    return rows


def run_case(source,folder,runtime,opt,pool,slots,base=None,progress=lambda *a:None):
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:size=list(im.size)
    assets=export.retained_images(base,size);save(folder/'retained_images.json',assets)
    errors=[];concerns=[];rows=[];history=[]
    def status(stage,done=0,total=1):
        save(folder/'progress.json',{'stage':stage,'done':done,'total':total,'updated_utc':now()});progress(stage,done,total)
    def call(root,region=None,existing=()):
        f=frame(source,root/'input',opt,region);clean=root/'input/clean.png'
        images=[('CLEAN source text',clean)]
        known=[]
        if existing or assets:
            local_hints(clean,[*existing,*assets],f,root/'input/annotations.png')
            known=[r['id'] for r in existing if intersect(r['bbox_2d'],f['box'])>0]
            images.append(('Existing T text IDs and retained IMG assets; annotations only',root/'input/annotations.png'))
        prompt=prompts.recognize(f['size'],local=region is not None,known=known,assets=bool(assets),
                                 max_regions=opt['max_regions'],default_font=opt['default_font'])
        value=model.ask(runtime,root,prompt,images,
                        lambda v:contracts.recognition(v,f['size'],known=known,local=region is not None,default_font=opt['default_font']),opt)
        save(root/'recognition.json',value);return value,f
    status('recognition')
    try:
        value,gf=call(folder/'detection/global');merge(rows,value['texts'],gf,'global',history,opt)
        concerns.extend(value['uncertain'])
    except Exception as exc:
        errors.append('Global recognition failed: '+str(exc));value={'regions':[]}
    save(folder/'detection/global_source.json',{'texts':rows,'errors':errors})
    regions=value['regions']
    if len(regions)>opt['max_regions']:concerns.append('Region budget exceeded; some proposed regions were not reviewed')
    selected=regions[:opt['max_regions']] if opt['local_enabled'] else []
    if regions and not opt['local_enabled']:concerns.append('Enlarged text review disabled with pending regions')
    save(folder/'detection/regions.json',{'proposed':regions,'selected':selected})
    snapshot=copy.deepcopy(rows)
    jobs={pool.submit(call,folder/'detection/regions'/f'R{i:02d}',mapped_box(r['bbox_2d'],gf,True),snapshot):i
          for i,r in enumerate(selected,1)}
    results={};status('local_recognition',0,len(jobs))
    for fut in as_completed(jobs):
        i=jobs[fut]
        try:results[i]=fut.result()
        except Exception as exc:errors.append(f'Region R{i:02d}: {exc}')
        status('local_recognition',len(results)+sum(e.startswith('Region ') for e in errors),len(jobs))
    for i,(value,f) in sorted(results.items()):
        merge(rows,value['texts'],f,f'R{i:02d}',history,opt);concerns.extend(value['uncertain'])
    save(folder/'detection/merge_history.json',history)
    save(folder/'recognition.json',{'texts':rows,'uncertain':concerns,'errors':errors})
    with Image.open(source) as im:overlay(im,rows,folder/'overlay.png')
    status('font_sizing',len(rows),len(rows))
    fitted,layout_errors=layout.fit_all(rows,folder/'typography',slots,opt);errors.extend(layout_errors)
    unresolved=[r['id'] for r in rows if r['legibility']!='clear']
    scene={'protocol':PROTOCOL,'source_sha256':sha(source),'source_size':size,'texts':fitted,
           'unreadable':[r for r in rows if not r['text'].strip()],'uncertain':concerns,'errors':errors}
    save(folder/'scene.json',scene)
    status('export',len(fitted),len(rows))
    export_failed=False
    try:counts=export.final(fitted,source,folder,slots,opt,base)
    except Exception as exc:
        export_failed=True
        errors.append('Native export/render failed: '+str(exc));counts={'textboxes':len(fitted)}
        save(folder/'export_error.json',{'error':str(exc)})
    state='failed' if export_failed or (not rows and errors) else 'completed_with_unresolved' if errors or concerns or unresolved else 'completed'
    result={'state':state,'text_lines':len(rows),'exported_lines':len(fitted),'unresolved_ids':unresolved,
            'errors':errors,'uncertain':concerns,'local_regions':len(selected),'counts':counts,'visual_accuracy_verified':False}
    save(folder/'summary.json',result);save(folder/'usage.json',usage(folder));scene['errors']=errors;save(folder/'scene.json',scene)
    lines=['# 文字恢复','',f'状态：{state}；识别 {len(rows)} 行，导出 {len(fitted)} 个可编辑文本框。',
           '', '[逐行内容与字号](scene.json) · [原始识别](recognition.json) · [文字 PPT](text.pptx)',
           '', '![文字定位](overlay.png)', '', '![文字渲染](comparison.png)']
    if base:lines+=['','[完整组合 PPT](combined.pptx)','','![完整组合](combined_comparison.png)']
    (folder/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');status(state,len(rows),len(rows))
    return result
