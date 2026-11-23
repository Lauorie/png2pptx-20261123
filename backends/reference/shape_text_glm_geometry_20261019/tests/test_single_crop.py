from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image,ImageDraw

from shape_integration.common import save,read,sha
from shape_integration.texts import single_crop as s,local_content


def row(ident,box):
    return {'id':ident,'text':ident,'kind':'text','latex':'',
            'bbox_2d':box,'rotation_deg':0,'font':'Arial','color':'#000000',
            'bold':False,'italic':False,'underline':False,'visual_line_count':1,
            'legibility':'clear','line_layout_status':'single_baseline',
            'runs':[{'text':ident,'script':'normal','color':'#000000'}]}


def draw_rows(rows,path):
    image=Image.new('RGB',(200,80),'white');draw=ImageDraw.Draw(image)
    for r in rows:draw.rectangle(r['bbox_2d'],fill='black')
    image.save(path)


def fixture(tmp_path,high_only=False):
    rows=[row('Low',[10,20,35,35]),row('High',[80,20,110,35])]
    source=tmp_path/'source.png';truth=deepcopy(rows);truth[0]['bbox_2d']=[20,20,45,35]
    draw_rows(truth,source)
    if high_only:rows=deepcopy(truth)
    base=tmp_path/'baseline';base.mkdir();draw_rows(rows,base/'native.png')
    for name in ('text.pptx','native.pdf'):(base/name).write_bytes(b'synthetic test asset')
    save(base/'scene.json',{'texts':rows,'source_size':[200,80]})
    output=tmp_path/'output';output.mkdir()
    return source,base,output,rows


@pytest.mark.parametrize('regress',[False,True])
def test_only_low_score_targets_get_one_joint_review_and_cpu_rollback(tmp_path,monkeypatch,regress):
    source,base,out,rows=fixture(tmp_path);calls=[]
    def refine(original,image,folder,runtime,opt,pool,progress,**kw):
        calls.append(kw)
        assert kw['target_ids']==['Low'] and kw['collision_passes']==0
        assert kw['rendered']==base/'native.png'
        assert kw['scores']['High']['before']['score']>.99
        changed=deepcopy(original)
        changed[0].update(text='Corrected',bold=True,bbox_2d=[50,20,75,35] if regress else [20,20,45,35])
        changed[0]['runs']=[{'text':'Corrected','script':'normal','color':'#000000','bold':True}]
        save(folder/'summary.json',{'groups':1,'unresolved_ids':[]})
        return changed,[]
    monkeypatch.setattr(s.local_content,'refine',refine)
    monkeypatch.setattr(s.layout,'fit_all',lambda r,*a,**k:(r,[]))
    monkeypatch.setattr(s.colors,'recover',lambda r,*a:(r,[]))
    def export(rows,image,folder,*args):
        folder.mkdir(parents=True,exist_ok=True);draw_rows(rows,folder/'native.png')
        for name in ('text.pptx','native.pdf'):(folder/name).write_bytes(b'synthetic test asset')
    monkeypatch.setattr(s.export,'final',export)
    summary=s.finish(source,out,base,{},object())
    final=read(out/'final/scene.json')['texts']
    assert len(calls)==1 and summary['review_rounds']==1
    assert final[1]==rows[1]
    if regress:
        assert final==rows and summary['accepted_ids']==[]
        assert len(read(out/'cpu_guard/history.json'))==2
    else:
        assert summary['accepted_ids']==['Low']
        assert final[0]['text']=='Corrected' and final[0]['bold'] is True
        assert final[0]['bbox_2d']==[20,20,45,35]


def test_no_low_scores_makes_no_model_call_or_extra_render(tmp_path,monkeypatch):
    source,base,out,rows=fixture(tmp_path,high_only=True)
    def forbidden(*a,**k):raise AssertionError('Unexpected review/render')
    monkeypatch.setattr(s.local_content,'refine',forbidden)
    monkeypatch.setattr(s.export,'final',forbidden)
    result=s.finish(source,out,base,{},None)
    assert result['review_rounds']==0 and result['groups']==0
    assert read(out/'final/scene.json')['texts']==rows


def test_equivalent_targets_envelope_keeps_strict_occurrence_validation():
    original=row('One',[10,10,60,25])
    unit={**original,'direction_status':'confirmed','reason':'Visible source text'}
    item={'id':'One','status':'resolved','units':[unit]}
    checked=local_content.validate({'targets':[item]},[original],[200,80])
    assert checked['items'][0]['id']=='One'
    with pytest.raises(ValueError,match='Every target'):
        local_content.validate({'targets':[item,item]},[original],[200,80])
    with pytest.raises(ValueError,match='Expected items'):
        local_content.validate({'targets':[item],'unexpected':True},[original],[200,80])


def test_upscaled_edge_box_clips_only_half_pixel_support():
    from shape_integration.texts.controller import mapped_box
    frame={'box':[0,0,100,50],'size':[1000,500],'scale':10.,'offset':[0,0],
           'source_size':[100,50]}
    box=[0,0,999,499]
    assert mapped_box(box,frame,True)[0]<0
    assert local_content.source_box(box,frame)==[0,0,99,49]
    box=[100,100,900,400]
    assert local_content.source_box(box,frame)==mapped_box(box,frame,True)
    with pytest.raises(ValueError,match='Invalid local'):
        local_content.source_box([-1,0,999,499],frame)
    with pytest.raises(ValueError,match='outside source'):
        local_content.source_box([0,0,999,499],{**frame,'box':[10,0,110,50]})


def test_three_aligned_images_include_bbox_highlights_and_readonly_context(tmp_path):
    source,base,out,rows=fixture(tmp_path)
    rendered=tmp_path/'render.png';draw_rows(rows,rendered)
    source_hash=sha(source);calls=[]
    class Runtime:
        def for_stage(self,**kwargs):return self
        def call(self,folder,name,system,prompt,images,validator,max_tokens,**kwargs):
            calls.append(prompt)
            assert [t['id'] for t in prompt['targets']]==['Low']
            assert any(c['id']=='High' for c in prompt['context_read_only'])
            assert len(images)==3
            sizes=[Image.open(path).size for _,path in images]
            assert len(set(sizes))==1
            clean=np.asarray(Image.open(images[0][1]).convert('RGB'))
            hint=np.asarray(Image.open(images[1][1]).convert('RGB'))
            assert np.any(clean!=hint)
            assert np.any(np.all(hint==[224,32,64],axis=2))
            assert np.any(np.all(hint==[136,136,136],axis=2))
            target=prompt['targets'][0]
            unit={**deepcopy(rows[0]),'bbox_2d':target['rough_bbox_2d'],
                  'direction_status':'confirmed','reason':'Complete visible baseline'}
            return validator({'items':[{'id':'Low','status':'resolved','units':[unit]}]})
    opt={'geometry_max_targets':6,'crop_min_padding_px':8,'crop_padding':.04,
         'crop_scale':3.,'global_long_edge':200,'crop_max_long_edge':400,'max_tokens':1000}
    with ThreadPoolExecutor(max_workers=1) as pool:
        revised,errors=local_content.refine(rows,source,out/'review',Runtime(),opt,pool,
            target_ids=['Low'],rendered=rendered,
            scores={'Low':{'before':{'score':.4}}},collision_passes=0)
    assert len(calls)==1 and not errors
    assert sha(source)==source_hash and revised[1]==rows[1]
    assert read(out/'review/plan.json')['max_collision_passes']==0


def test_neighbor_regression_rejects_candidate_without_falsifying_its_score():
    from shape_integration.texts.bbox_color_review import nonregression_gate
    metric={'before':{'score':.4},'after':{'score':.9,'render_mass':20},
            'reliable':True,'neighbor_regressions':['High']}
    assert nonregression_gate(metric)==(False,'protected_neighbor_cpu_score_decreased')


def test_global_inventory_defers_ordinary_crops_and_multiline_retry(tmp_path,monkeypatch):
    from shape_integration.texts import staged_formula,multiline_retry,options
    source=tmp_path/'source.png';Image.new('RGB',(200,80),'white').save(source)
    unit=row('Low',[10,10,80,25]);unit['replaces']=[]
    calls=[]
    def ask(runtime,folder,prompt,*args):
        calls.append(prompt)
        return {'texts':[unit],'regions':[{'bbox_2d':[5,5,100,40],'reason':'Dense region'}],'uncertain':[]}
    def forbidden(*a,**k):raise AssertionError('Unexpected pre-baseline multiline request')
    monkeypatch.setattr(staged_formula.model,'ask',ask)
    monkeypatch.setattr(multiline_retry,'repair',forbidden)
    opt=options({});opt.update(global_long_edge=200,inventory_local_enabled=False,multiline_before_baseline=False)
    result=staged_formula.recognize(source,tmp_path/'staged',None,opt)
    assert len(calls)==1 and len(result['texts'])==1
    assert result['regions_needing_review']
    assert not (tmp_path/'staged/regions').exists()
    assert not (tmp_path/'staged/multiline_retry').exists()
    assert any('regions=[]' in rule for rule in calls[0]['rules'])
