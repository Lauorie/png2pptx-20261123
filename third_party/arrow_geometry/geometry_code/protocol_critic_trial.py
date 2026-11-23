"""Frozen, request-auditable pilot for target/path protocol and local refinement."""
import argparse,copy,json,shutil,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from PIL import Image,ImageDraw,ImageFont
import arrow_path_protocol as pp
import arrow_local_critic as critic
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model


def image_grid(items,dest):
    cards=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for label,file in items:
        im=Image.open(file).convert('RGB');im.thumbnail((1000,1100))
        card=Image.new('RGB',(max(im.width,220),im.height+32),'white');card.paste(im,(0,32))
        ImageDraw.Draw(card).text((5,3),label,font=font,fill='black');cards.append(card)
    result=Image.new('RGB',(sum(i.width for i in cards),max(i.height for i in cards)),'#eeeeee');x=0
    for c in cards:result.paste(c,(x,0));x+=c.width
    result.save(dest)


def prepare(run,source):
    jobs=[]
    for cid,name,oldjob,clean_name,context_name in [
        ('ds012','upload','ds012_direct_geometry','02.png','01.png'),
        ('ds025','good_geometry','ds025_direct_geometry','02.png','01.png')]:
        old=source/'cases'/cid/oldjob;out=run/'cases'/cid/name;out.mkdir(parents=True,exist_ok=True)
        raw=read(old/'answer.json');prompt=read(old/'modified_prompt.json')
        paths=[]
        for p in raw['paths']:
            paths.append(dict(id=p['id'],kind='stroke',parts=p['parts'],style={k:p[k] for k in ('color','width_px','line_style','heads')}))
        doc=pp.validate(dict(version=1,frame='CLEAN',size=prompt['size'],targets=[dict(id='T1',paths=paths)]))
        for dest,src in [('clean.png',old/'input'/clean_name),('context.png',old/'input'/context_name)]:shutil.copy2(src,out/dest)
        write(out/'seed.json',doc)
        jobs.append(dict(id=cid+'_'+name,case=cid,folder=str(out),source=str(old),source_answer_sha256=sha256(old/'answer.json'),
            crop_in_context=prompt['crop_in_context'],kind='critic',seed_origin='previous direct geometry request'))
    cid='063';old=source/'cases/063/063_style';out=run/'cases'/cid/'brace';out.mkdir(parents=True,exist_ok=True)
    raw=read(old/'original_answer.json');sty=read(old/'answer.json')['styles'];prompt=read(old/'original_prompt.json')
    styles={s['id']:{k:s[k] for k in ('color','width_px','line_style','heads')} for s in sty}
    doc,audit=pp.from_legacy(raw,prompt['size'],styles);write(out/'seed.json',doc);write(out/'import_audit.json',audit)
    shutil.copy2(old/'input/01.png',out/'clean.png');shutil.copy2(old/'input/04.png',out/'context.png')
    jobs.append(dict(id='063_brace',case=cid,folder=str(out),source=str(old),source_answer_sha256=sha256(old/'original_answer.json'),
        style_answer_sha256=sha256(old/'answer.json'),crop_in_context=prompt['crop_in_context'],kind='critic',seed_origin='unchanged original GLM geometry plus previous GLM black 2.5px style'))
    # Real rejected branch output: validate/import/render/export without a new call.
    old=source/'offline/ds012_download_branch_contract';out=run/'cases/ds012/download_protocol';out.mkdir(parents=True,exist_ok=True)
    raw=read(old/'raw_answer.json');size=list(Image.open(old/'clean.png').size)
    styles={r['id']:dict(color='#707070',width_px=2,line_style='solid',heads=['none','none']) for r in raw['routes']}
    doc,audit=pp.from_legacy(raw,size,styles);write(out/'seed.json',doc);write(out/'import_audit.json',audit);shutil.copy2(old/'clean.png',out/'clean.png')
    original_parts=[[raw['nodes'][next(i for i,n in enumerate(raw['nodes']) if n[0]==ref)][1:] for ref in s['nodes']] for t in raw['routes'] for s in t['segments']]
    new_parts=[part['points'] for t in doc['targets'] for p in t['paths'] for part in p['parts']]
    assert original_parts==new_parts
    critic.render(doc,out/'clean.png',out/'result',export_native=True)
    from lib import native
    native.render(out/'result/arrows.pptx',out/'native',size)
    image_grid([('Original',out/'clean.png'),('Imported paths',out/'result/styled.png'),('Native PPT render',out/'native/styled.png'),('Geometry overlay',out/'result/overlap.png')],out/'comparison.png')
    write(out/'state.json',dict(state='completed',mode='offline_protocol',api_calls=0,coordinates_changed=False,
          parts=len(new_parts),paths=sum(len(t['paths']) for t in doc['targets']),native_rendered=True,diagnostic_style=True))
    jobs.append(dict(id='ds012_download_protocol',case='ds012',folder=str(out),source=str(old),kind='offline',
        source_answer_sha256=sha256(old/'raw_answer.json'),seed_origin='previous rejected geometry, no point changes; diagnostic style only'))
    for j in jobs:
        out=Path(j['folder']);write(out/'provenance.json',{**j,'seed_sha256':sha256(out/'seed.json'),'clean_sha256':sha256(out/'clean.png'),
              'sam_calls':0,'fit_from_mask':False,'coordinate_sampler':False})
        if j['kind']=='critic':critic.render(read(out/'seed.json'),out/'clean.png',out/'baseline',export_native=True)
    write(run/'manifest.json',dict(ids=['ds012','063','ds025'],jobs=jobs,source_run=str(source),critic_rounds=1,code_snapshot=str(run/'code')))
    report(run)


def changed_views(out,before,after):
    import numpy as np
    old=pp.path_index(before);new=pp.path_index(after);clean=Image.open(out/'clean.png').convert('RGB')
    root=out/'changed_paths';root.mkdir(exist_ok=True);notes=[]
    for i,key in enumerate(sorted(set(old)|set(new))):
        if old.get(key)==new.get(key):continue
        pts=[]
        for p in [old.get(key),new.get(key)]:
            if p:
                pts.extend(p.get('contour',[]) if p['kind']=='block_arrow' else [xy for s in p['parts'] for xy in s['points']])
        if not pts:continue
        a=np.array(pts,float);lo=a.min(axis=0);hi=a.max(axis=0);margin=max(24,float((hi-lo).max())*.1)
        box=[max(0,int(lo[0]-margin)),max(0,int(lo[1]-margin)),min(clean.width,int(hi[0]+margin+1)),min(clean.height,int(hi[1]+margin+1))]
        assets=[]
        for title,src in [('Original',out/'clean.png'),('Before',out/'critic/before/styled.png'),('After',out/'critic/after/styled.png'),('Overlay',out/'critic/after/overlap.png')]:
            dest=root/f'{i}_{title}.png';Image.open(src).crop(box).save(dest);assets.append((title,dest))
        image_grid(assets,root/f'{i}_comparison.png');notes.append(dict(path=key,bbox=box,image=f'changed_paths/{i}_comparison.png'))
    write(out/'changed_paths.json',notes)


def run_job(job,model):
    out=Path(job['folder']);start=time.time();write(out/'state.json',dict(state='running',started=start))
    try:
        seed=read(out/'seed.json')
        after,state=critic.refine(seed,out/'clean.png',out/'critic',model,out/'context.png',job['crop_in_context'])
        state.update(seconds=time.time()-start,paths_before=len(pp.path_index(seed)),paths_after=len(pp.path_index(after)),
                     changed_paths=[k for k in sorted(set(pp.path_index(seed))|set(pp.path_index(after))) if pp.path_index(seed).get(k)!=pp.path_index(after).get(k)])
        write(out/'after.json',after)
        image_grid([('Original',out/'clean.png'),('Before',out/'critic/before/styled.png'),('After',out/'critic/after/styled.png'),('Overlay',out/'critic/after/overlap.png')],out/'comparison.png')
        changed_views(out,seed,after)
    except Exception as e:
        state=dict(state='failed',error=str(e),seconds=time.time()-start,retained_previous=True);write(out/'error.json',dict(error=str(e),traceback=traceback.format_exc()))
    write(out/'state.json',state);return {'id':job['id'],**state}


def report(run):
    manifest=read(run/'manifest.json');lines=['# 多路径协议 + 局部 critic/refine 小实验','',
        '预算不变：113000 输出上限、1200 秒总超时、120 秒空闲超时、thinking enabled。使用已保存的输入与几何，不重新 SAM；本轮不运行全量。','',
        '先确认协议能保留分支，再检查 critic 是否改善坏例、是否损害良例。以下完成状态不等于视觉通过。','',
        '| 局部任务 | 状态 | 决定 | 路径数（前→后） | 请求 |','|---|---|---|---|---|']
    states=[]
    for j in manifest['jobs']:
        out=Path(j['folder']);s=read(out/'state.json') if (out/'state.json').exists() else {'state':'prepared'};states.append({'id':j['id'],**s})
        rel=str(out.relative_to(run));lines.append(f"| {j['id']} | {s['state']} | {s.get('decision',j['kind'])} | {s.get('paths_before','—')} → {s.get('paths_after',s.get('paths','—'))} | [输入输出]({rel}/critic/request) · [状态]({rel}/state.json) |")
    lines+=['','## 原图 / 修改前 / 修改后 / 叠图','']
    for j in manifest['jobs']:
        out=Path(j['folder']);rel=out.relative_to(run)
        if (out/'comparison.png').exists():lines += [f"### {j['id']}",'',f'![comparison]({rel}/comparison.png)','']
        if (out/'changed_paths.json').exists():
            for v in read(out/'changed_paths.json'):lines += [f"局部变化：{v['path']}",'',f"![changed path]({rel}/{v['image']})",'']
    (run/'RESULTS.md').write_text('\n'.join(lines));write(run/'progress.json',{'updated':time.time(),'states':states})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--source-run',type=Path);ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--report',action='store_true');a=ap.parse_args();run=a.run_dir.resolve()
    if a.prepare:prepare(run,a.source_run.resolve())
    if a.run:
        config=read(run/'config.json');assert config['max_tokens']==113000 and config['request_timeout_seconds']==1200
        model=Model(config,run_dir=run);jobs=[j for j in read(run/'manifest.json')['jobs'] if j['kind']=='critic']
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(run_job,j,model) for j in jobs]):print(json.dumps(f.result(),ensure_ascii=False),flush=True);report(run)
        report(run);write(run/'finished.json',{'state':'finished','critic_tasks':len(jobs),'all_cases_full_pipeline':False})
    if a.report:report(run)

if __name__=='__main__':main()
