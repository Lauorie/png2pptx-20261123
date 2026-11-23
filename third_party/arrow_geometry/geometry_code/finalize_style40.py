"""Verify completed style replay and render its actual editable PPTs."""
import argparse,collections,fcntl,json,re,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import numpy as np
from PIL import Image
import pipeline
from transport.common import read_json as read,write_json as write,sha256 as sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    root=a.run.resolve();source=a.source.resolve()
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    finished=read(root/'finished.json');ids=read(root/'manifest.json')['cases'];assert len(ids)==40
    geometry=[];pending=[];exports=[];errors=[];calls=[]
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for cid in ids:
        folder=root/'cases'/cid;rows=read(folder/'scene.json')['arrows'];old=read(source/'cases'/cid/'scene.json')['arrows']
        before={r['id']:r for r in old};after={r['id']:r for r in rows}
        assert before.keys()==after.keys(),cid
        for key,r in after.items():
            for field in ('parent_id','start','via','end','beziers','contours','heads','node_ids'):
                assert r.get(field)==before[key].get(field),(cid,key,field)
            if pipeline.style.pending(r):pending.append(dict(case=cid,id=key,label=r['label'],kind=r['kind']))
            if r['kind']!='block_arrow' and r.get('line_style') in ('solid','dashed','dotted'):
                assert r['style']['dash']['type']==r['line_style'],(cid,key,'model category lost')
        geometry.append(dict(case=cid,rows=len(rows),identities_geometry_heads_unchanged=True))
        with ZipFile(folder/'arrows.pptx') as z:xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
        shapes={}
        for tag in ('sp','cxnSp'):
            for s in xml.findall('.//p:'+tag,ns):shapes[s.find('.//p:cNvPr',ns).get('name')]=s
        expected={r['id'] for r in rows if not pipeline.style.pending(r)}
        assert set(shapes)==expected,(cid,'PPT geometry accounting')
        for key,s in shapes.items():
            row=after[key]
            if row['kind']=='block_arrow':continue
            pattern=row['style']['dash']
            if pattern.get('mark_px') and pattern.get('gap_px'):assert s.find('.//a:custDash',ns) is not None
            else:assert s.find('.//a:prstDash',ns).get('val')==pattern['preset']
        assert not xml.findall('.//p:pic',ns)
        exports.append(dict(case=cid,editable_shapes=len(shapes),pending_not_committed=len(rows)-len(shapes),raster_images=0))
    def render(cid):
        folder=root/'cases'/cid
        pipeline.native.render(folder/'arrows.pptx',folder/'native',Image.open(folder/'clean.png').size)
        write(folder/'native/source.json',dict(ppt_sha256=sha(folder/'arrows.pptx')))
        page=folder/'RESULTS.md';text=page.read_text()
        if '![](native/styled.png)' not in text:page.write_text(text+'\n实际 LibreOffice 渲染：\n\n![](native/styled.png)\n')
        return cid
    with ThreadPoolExecutor(max_workers=3) as pool:
        for cid in pool.map(render,ids):print(cid,'PPT verified',flush=True)
    for f in root.glob('cases/*/*/*/classify/*/request/call_*/status.json'):
        d=read(f);calls.append(dict(path=str(f.parent.relative_to(root)),state=d['state'],seconds=d.get('seconds'),usage=d.get('usage'),reasoning_chars=d.get('reasoning_chars')))
    latency=[c['seconds'] for c in calls if c['state']=='transport_complete']
    request_stats=dict(calls=calls,states=dict(collections.Counter(c['state'] for c in calls)),
        completed_latency_p50_p90_p99=np.percentile(latency,[50,90,99]).tolist(),
        note='Old interrupted streams remain archived; their logical requests may have a validated complete answer from another call.')
    write(root/'request_statistics.json',request_stats)
    result=dict(cases=40,groups=finished['groups'],stroke_rows=finished['stroke_rows'],classified=finished['classified'],measured=finished['measured'],
        geometry_checks=geometry,ppt_checks=exports,pending=pending,group_failures=finished['failures'],request_failures=finished['request_failures'],
        original_frozen_code_unchanged=all(sha(root/'code'/k)==v for k,v in read(root/'code_manifest.json').items()),accuracy_measured=False)
    write(root/'VALIDATION.json',result)
    text=(root/'RESULTS.md').read_text();first,rest=text.split('\n',1)
    note=f'''\n\n已完成 **40/40 张、{finished['groups']} 个已有分组、{finished['classified']}/{finished['stroke_rows']} 段线几何的类别判断与样式测量**。填充块状箭头沿用原轮廓。\n\n普通 mask、mask 重选、GLM 点修复、已有补漏输出，以及直线/折线/曲线均已覆盖。身份、几何和端点头型核对不变；40 份 PPT 已实际渲染，并验证虚实线编码。\n\n**这是样式阶段回放，不是重新发现/重新拟合全图。** 原来未发现、未完成或几何不正确的对象仍保留相应局限。模型和像素同时不确定的 {len(pending)} 段保留几何待复查，不提交为实线。\n\n[完整验证](VALIDATION.json) · [耗时与请求记录](request_statistics.json) · [格式解析修复重放](format_replay.json) · [31 项代码检查](postfix_tests.json)\n\n'''
    if pending:
        note+='待复查：'+ '；'.join(f"{r['case']} {r['label']}（{r['id']}）" for r in pending)+'。\n\n'
    (root/'RESULTS.md').write_text(first+note+rest)
    missing=[]
    for page in [root/'RESULTS.md',*root.glob('cases/*/RESULTS.md')]:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',page.read_text()):
            if '://' not in target and not (page.parent/target).exists():missing.append((str(page),target))
    assert not missing,missing
    print(json.dumps({k:result[k] for k in ('cases','groups','stroke_rows','classified','measured','pending','group_failures','request_failures')},indent=2),flush=True)


if __name__=='__main__':main()
