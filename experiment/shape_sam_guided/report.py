"""Raw paired results; evaluation does not modify or tune reconstructed geometry."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import io
import json
from pathlib import Path
import statistics
import time
import zipfile
import cairosvg
from PIL import Image,ImageDraw
from core import read,save,sha
from models import Models
from evidence import edge_distance,score


def render(scene, crop=None, long_edge=2048):
    x0,y0,x1,y1=crop or [0,0,*scene['source_size']]
    w,h=x1-x0,y1-y0;scale=long_edge/max(w,h)
    size=(max(2,round(w*scale)),max(2,round(h*scale)))
    rows=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{x0-.5} {y0-.5} {w} {h}">',
          f'<rect x="{x0-.5}" y="{y0-.5}" width="{w}" height="{h}" fill="white"/>']
    for p in scene['parts'].values():
        if p['status']!='ok':continue
        path=' '.join(c[0]+' '+' '.join(format(v,'.12g') for v in c[1:]) for c in p['commands'])
        rows.append(f'<path d="{path}" fill="none" stroke="#006bea" stroke-width="1.2" stroke-linejoin="round"/>')
    rows.append('</svg>')
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=''.join(rows).encode(),output_width=size[0],output_height=size[1]))).convert('RGB')


def scene_path(root,item,config,method):
    if method=='historical':return Path(config['baseline_root'])/'runs'/config['baseline_run']/'cases'/item['id']/'fitting/scene.json'
    return root/method/'cases'/item['id']/'fitting/scene.json'


def audit_scene(path):
    scene=read(path)
    for key,p in scene['parts'].items():
        if p['object_id'] not in scene['objects']:raise ValueError('Dangling object reference')
        if any(k not in scene['points'] for k in p.get('point_keys',[])):raise ValueError('Dangling point reference')
        if key not in scene['paths'] and p['status']=='ok':raise ValueError('Missing path')
    for obj in scene['objects'].values():
        if any(k not in scene['parts'] for k in obj['part_keys']):raise ValueError('Dangling part reference')
    ppt=path.with_name('geometry_final.pptx')
    with zipfile.ZipFile(ppt) as archive:
        xml=archive.read('ppt/slides/slide1.xml').decode()
        if '<p:pic>' in xml:raise ValueError('Unexpected bitmap in geometry PPT')
    return {'scene_sha256':sha(path),'ppt_sha256':sha(ppt),'objects':len(scene['objects']),
            'parts':sum(p['status']=='ok' for p in scene['parts'].values()),'picture_shapes':0}


def usage_stats(root):
    totals=Counter();by_stage=Counter();service=0.;queue=0.;revisions=Counter()
    for p in root.rglob('response_0.json'):
        r=read(p);u=r.get('usage') or {};totals['calls']+=1
        for name in ['prompt_tokens','completion_tokens','total_tokens']:totals[name]+=u.get(name,0)
        by_stage[p.parent.name.rsplit('_',1)[0]]+=1
        timing=p.with_name('timing.json')
        if timing.exists():
            t=read(timing);service+=t.get('service_seconds',0);queue+=t.get('queue_seconds',0)
    for p in (root/'cases').glob('*/fitting/objects/*/record.json'):
        row=read(p);revisions['corrections']+=row.get('corrections',0);revisions['context_rounds']+=row.get('context_rounds',0)
        revisions['objects']+=1
    replayed=list(root.rglob('replay.json'))
    replay_tokens=sum(read(p).get('reference_usage',{}).get('total_tokens',0) for p in replayed)
    return {'totals':dict(totals),'calls_by_stage':dict(by_stage),'summed_service_seconds':service,
            'summed_queue_seconds':queue,**dict(revisions), 'replayed_calls':len(replayed),
            'replayed_total_tokens':replay_tokens, 'logical_calls':totals['calls']+len(replayed),
            'logical_total_tokens':totals['total_tokens']+replay_tokens}


def evaluate_pair(root,item,config,models,left,right):
    folder=root/'evaluation'/item['id']/(right+'_vs_'+left);folder.mkdir(parents=True,exist_ok=True)
    paths={k:scene_path(root,item,config,k) for k in [left,right]}
    identity={k:sha(p) for k,p in paths.items()}
    if (folder/'verdict.json').exists():
        cached=read(folder/'verdict.json')
        if cached['scene_hashes']!=identity:raise ValueError('Evaluated scenes changed')
        return cached
    scenes={k:read(p) for k,p in paths.items()}
    with Image.open(Path(config['baseline_root'])/item['original']) as im:source=im.convert('RGB')
    full=render(scenes[left]);source.resize(full.size,Image.Resampling.LANCZOS).save(folder/'source.png')
    for name,scene in scenes.items():render(scene).save(folder/(name+'.png'))
    plan=read(Path(config['baseline_root'])/'runs'/config['baseline_run']/'cases'/item['id']/'shapes/plan.json')
    crops=[]
    for region in plan['regions']:
        b=[v*item['source_size'][i%2]/item['input_size'][i%2] for i,v in enumerate(region['bbox_2d'])]
        crops.append([max(0,int(b[0])),max(0,int(b[1])),min(source.width,int(b[2])+1),min(source.height,int(b[3])+1)])
    reviews=[]
    for order,names in enumerate([[left,right],[right,left]]):
        target=folder/f'order_{order}';target.mkdir(exist_ok=True)
        panels=[]
        # One sheet per region avoids shrinking unrelated regions to unreadable thumbnails.
        for i,crop in enumerate(crops):
            a,b=render(scenes[names[0]],crop,1100),render(scenes[names[1]],crop,1100)
            original=source.crop(crop).resize(a.size,Image.Resampling.LANCZOS)
            w,h=a.size;board=Image.new('RGB',(w*3,h+30),'white');draw=ImageDraw.Draw(board)
            for j,(label,panel) in enumerate(zip(['Source','A','B'],[original,a,b])):
                board.paste(panel,(j*w,30));draw.text((j*w+8,8),label,fill='black')
            path=target/f'region_{i}.png';board.save(path);panels.append((f'Aligned region {i}: Source / A / B',path))
        prompt={'version':'sam_guided.geometry_review.2',
            'task':'Compare scientific-diagram geometry A and B against the clean source. Method identity is hidden.',
            'rules':['Assess completeness, false extra elements, position, dimensions, corner curvature, equal-size repeated shapes, gaps and face junctions.',
                     'Source and A/B region panels have EXACTLY the same crop and scale. Use source-visible evidence. A mask or object count is not accuracy ground truth.',
                     'Ignore diagnostic colors, fills, line width and dashed-versus-solid style. Text, photos and standalone connector arrows are outside this geometry-only experiment.',
                     'Connecting arrows are excluded whether thin, thick, filled or block-shaped. Their presence, absence or attached fragments must not decide the preference. This does not exclude intrinsic operator strokes or the actual contours of processing blocks.',
                     'For stroked shapes evaluate stroke centerlines, excluding cast shadows. A rounded rectangle need not be a semicircular-ended pill.',
                     'Inspect all regions. Prefer tie for equivalent geometry and uncertain when resolution or evidence is insufficient. Do not invent defects.',
                     'An error common to A and B is a residual error, not a regression. Record errors separately for each alternative.'],
            'output':{'preferred':'A|B|tie|uncertain','regressions':{'A':['specific source-supported disadvantages relative to B'],'B':['specific source-supported disadvantages relative to A']},
                      'residual_errors':{'A':['remaining errors'],'B':['remaining errors']},'evidence':'concrete comparison'}}
        assets=read(Path(config['baseline_root'])/'runs'/config['baseline_run']/'cases'/item['id']/'dedup/image_boxes_original.json')['assets']
        prompt['source_size_original']=item['source_size']
        prompt['excluded_image_regions_original']=[{'id':a['id'],'bbox':a['bbox_2d']} for a in assets]
        prompt['rules'].append('Contents of declared image-asset regions are handled outside shape reconstruction; do not report them as missing shapes. Preserve evaluation of any enclosing geometric frame. Inspect actual corner curvature rather than assuming a named box is a pill.')
        if config.get('evaluate_background_panels'):
            prompt['version'] = 'sam_fixed.geometry_review.1'
            prompt['rules'].append('Check enclosing panels and background regions before small details. '
                'A source-visible fill boundary defines shape geometry even without a stroke. Ignore fill '
                'color differences, but assess whether that panel exists and has the correct extent. '
                'A panel missing from BOTH alternatives remains a residual error for both. '
                'Do not invent a container from a group bbox without a source-visible boundary.')
        def validate(raw):
            if raw.get('preferred') not in {'A','B','tie','uncertain'} or not raw.get('evidence'):raise ValueError('Invalid evaluation')
            for field in ['regressions','residual_errors']:
                if not isinstance(raw.get(field),dict) or set(raw[field])!={'A','B'}:raise ValueError('Missing separate error inventory')
                if any(not isinstance(v,list) for v in raw[field].values()):raise ValueError('Error inventory must be lists')
            return raw
        review=models.ask(target/'call','Evaluate geometric reconstruction from image evidence. Return JSON only.',prompt,
            [('Clean source',folder/'source.png'),('A: geometry only',folder/(names[0]+'.png')),
             ('B: geometry only',folder/(names[1]+'.png')),*panels],validate,16000)
        mapping=dict(zip(['A','B'],names));proposal='A' if names[0]==right else 'B'
        reviews.append({'mapping':mapping,'preference':mapping.get(review['preferred'],review['preferred']),
                        'proposed_regressions':review['regressions'][proposal],**review})
    preferred=[r['preference'] for r in reviews]
    consensus=preferred[0] if preferred[0]==preferred[1] else 'inconsistent'
    result={'id':item['id'],'left':left,'right':right,'scene_hashes':identity,'consensus':consensus,
            'eligible_for_delivery':consensus==right and not any(r['proposed_regressions'] for r in reviews),
            'reviews':reviews,'evaluation_is_human_ground_truth':False}
    save(folder/'verdict.json',result);return result


def case_report(root,item,config,models):
    folder=root/'evaluation'/item['id'];folder.mkdir(parents=True,exist_ok=True)
    methods=['historical',*config.get('arms',['glm_control','sam_forward'])]
    audits={k:audit_scene(scene_path(root,item,config,k)) for k in methods}
    with Image.open(Path(config['baseline_root'])/item['original']) as im:source=im.convert('RGB')
    distance=edge_distance(source);metrics={}
    for name in methods:
        scene=read(scene_path(root,item,config,name))
        values=[score(p['commands'],distance)['mean_clipped_px'] for p in scene['parts'].values() if p['status']=='ok']
        metrics[name]={'edge_mean_px':statistics.mean(values) if values else None,**audits[name]}
    pairs=[evaluate_pair(root,item,config,models,left,right) for left,right in
           config.get('evaluation_pairs',[['historical','sam_forward'],['glm_control','sam_forward']])]
    result={'id':item['id'],'metrics':metrics,'pairs':pairs}
    save(folder/'summary.json',result)
    print('EVALUATED',item['id'],[p['consensus'] for p in pairs],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir;config=read(args.config);items=read(root/'manifest.json')['images'];models=Models(config);results=[]
    lock=(root/'.evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    errors=[]
    save(root/'evaluation_errors.json',errors)
    with ThreadPoolExecutor(max_workers=config['glm_concurrency']) as pool:
        pending={pool.submit(case_report,root,i,config,models):i for i in items}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                error={'id':pending[future]['id'],'type':type(exc).__name__,'error':str(exc)}
                errors.append(error);save(root/'evaluation_errors.json',errors)
                print('EVALUATION_FAILED',json.dumps(error),flush=True)
            save(root/'evaluation_progress.json',{'completed':len(results),'failed':len(errors),
                 'total':len(items),'updated_at':time.time()})
    if errors:
        raise RuntimeError(f'{len(errors)} evaluations failed; completed comparisons retained. See evaluation_errors.json.')
    results.sort(key=lambda r:r['id'])
    pairs=config.get('evaluation_pairs',[['historical','sam_forward'],['glm_control','sam_forward']])
    arms=config.get('arms',['glm_control','sam_forward'])
    methods=['historical',*arms]
    generated=read(root/'manifest.json')['config'].get('arms',arms)
    aggregate={'cases':len(results),'preferences':{right+'_vs_'+left:dict(Counter(next(p['consensus'] for p in r['pairs'] if p['left']==left and p['right']==right) for r in results))
              for left,right in pairs},'usage':{k:usage_stats(root/k) for k in arms},
              'generation_arms':generated,'reference_methods':[k for k in methods if k not in generated],
              'geometry_audits_passed':True,'results':results,'completed_at':time.time(),
              'evaluation_protocol':'sam_guided.geometry_review.2',
              'limitations':['27 previously inspected regression cases, not an unseen test set.',
                 'Reviewer is same model family; even order-consistent votes are not human annotations.',
                 'Historical run used a different serving endpoint/token ceiling and changing revision budget.',
                 'Paired arms hold initial discovery/layout, weights and revision budgets fixed; generation remains stochastic.',
                 'Edge distance is a diagnostic, not ground-truth geometry accuracy.']}
    if config.get('reuse_request_root') or config.get('reuse_request_roots') or config.get('reuse_request_map'):
        aggregate['limitations'].append(
            'Controlled incremental ablation: unchanged effective requests replay frozen reference responses. '
            'This is conditional on those reference outputs, not an independent fresh generation. '
            'Actual calls/tokens exclude replays; logical calls/tokens include them for cold-run accounting.')
    save(root/'aggregate.json',aggregate)
    with (root/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['id',*[k+'_parts' for k in methods],*[k+'_edge_px' for k in methods],*[right+'_vs_'+left for left,right in pairs]])
        for r in results:
            writer.writerow([r['id'],*[r['metrics'][k]['parts'] for k in methods],
               *[r['metrics'][k]['edge_mean_px'] for k in methods],*[p['consensus'] for p in r['pairs']]])
    lines=['# SAM bbox / mask / critic 消融','',
        '生成组按配置执行区域复查、逐对象观察/拟合、原有 critic/refine 和 coverage；共同使用裁剪不足恢复。SAM 按配置补充范围、尺度与轮廓，critic 信息独立开关。',
        '',f'完成 {len(results)} 图；本目录生成组：'+', '.join(generated)+'。只读对照：'+', '.join(aggregate['reference_methods'])+'。双顺序模型评价分布：',
        '',json.dumps(aggregate['preferences'],ensure_ascii=False),
        '', '[逐图指标](metrics.csv) · [完整记录](aggregate.json) · [实际 PPT 对比总览](native_review/overview.pdf)',
        '', '| 图 | '+' | '.join(right+' vs '+left for left,right in pairs)+' | '+' | '.join(arms)+' |',
        '|---|'+'---|'*(len(pairs)+len(arms))]
    for r in results:
        ident=r['id'];lines.append('| '+ident+' | '+' | '.join(p['consensus'] for p in r['pairs'])+' | '+' | '.join(f'[PPT]({arm}/cases/{ident}/fitting/geometry_final.pptx)' for arm in arms)+' |')
    lines+=['','## 效率','', '调用统计包括正向区域复查、对象观察、critic、coverage 和图像冲突检查，不含离线评估。SAM 新推理成本单列在 sam_cache/*/index.json。并发共享服务下的单图墙钟时间包含竞争，不能直接解释为独立部署提速。',
            '',json.dumps(aggregate['usage'],ensure_ascii=False,indent=2),'','## 限制','',*['- '+s for s in aggregate['limitations']]]
    (root/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
