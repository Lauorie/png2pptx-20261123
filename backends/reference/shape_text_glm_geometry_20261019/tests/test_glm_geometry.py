from copy import deepcopy
import json

from PIL import Image
import pytest

from shape_integration.texts import glm_geometry as g
from shape_integration.common import save, read


def row(ident='T1',text='Hello World',**updates):
    return dict(id=ident,text=text,kind='text',latex='',runs=[{'text':text,'script':'normal','bold':True}],
                bbox_2d=[10,10,95,40],rotation_deg=0,visual_line_count=1,**updates)


def response(text='Hello World',box=None):
    return {'items':[{'id':'T1','status':'located','reason':'Source glyph bounds',
                     'lines':[{'text':text,'bbox_2d':box or [10,10,95,40]}]}]}


def frame():
    return {'box':[0,0,200,100],'size':[400,200],'source_size':[200,100],
            'scale':2,'offset':[0,0]}


def test_checked_bbox_maps_back_without_changing_text_or_styles():
    r=row();before=deepcopy(r)
    checked=g.validate(response(box=[20.5,20.5,190.5,80.5]),[r],[400,200])
    result=g.mapped_detections(checked,[r],frame())[0]
    assert result['target_id']=='T1' and result['source_bbox']==[10,10,95,40]
    assert result['text']=='Hello World' and r==before
    assert 'confidence' not in result  # Never pretend a geometry proposal is calibrated OCR.


@pytest.mark.parametrize('mutation',['text','style','missing','duplicate','outside','nan','empty'])
def test_invalid_replies_cannot_overwrite_frozen_targets(mutation):
    raw=response()
    if mutation=='text':raw['items'][0]['lines'][0]['text']='Hello Word'
    if mutation=='style':raw['items'][0]['bold']=False
    if mutation=='missing':raw['items']=[]
    if mutation=='duplicate':raw['items']*=2
    if mutation=='outside':raw['items'][0]['lines'][0]['bbox_2d']=[-1,0,40,50]
    if mutation=='nan':raw['items'][0]['lines'][0]['bbox_2d']=[0,0,float('nan'),50]
    if mutation=='empty':raw['items'][0]['lines']=[]
    with pytest.raises(ValueError):g.validate(raw,[row()],[200,100])


def test_ambiguous_has_no_geometry_and_formula_cannot_split():
    raw={'items':[{'id':'T1','status':'ambiguous','reason':'Unclear occurrence','lines':[]}]}
    assert g.mapped_detections(g.validate(raw,[row()],[200,100]),[row()],frame())==[]
    r=row();r.update(kind='formula',visual_line_count=2,latex=r'\frac{x}{y}')
    raw=response();raw['items'][0]['lines']=[{'text':'Hello','bbox_2d':[10,10,50,25]},
                                         {'text':'World','bbox_2d':[10,40,50,65]}]
    with pytest.raises(ValueError,match='never split formulas'):g.validate(raw,[r],[200,100])


def test_multiline_spans_partition_content_and_reject_word_split():
    r=row();r['visual_line_count']=2
    raw=response();raw['items'][0]['lines']=[{'text':'Hello','bbox_2d':[10,10,50,25]},
                                         {'text':'World','bbox_2d':[10,40,50,65]}]
    assert g.validate(raw,[r],[200,100])['items'][0]['spans']==[[0,5],[6,11]]
    raw['items'][0]['lines'][0]['text']='Hel'
    raw['items'][0]['lines'][1]['text']='lo World'
    with pytest.raises(ValueError,match='word'):g.validate(raw,[r],[200,100])


def test_glm_geometry_worker_saves_failures_without_dropping_targets(tmp_path,monkeypatch):
    crop=tmp_path/'input';crop.mkdir()
    Image.new('RGB',(400,200),'white').save(crop/'clean.png');save(crop/'frame.json',frame())
    image=tmp_path/'source.png';Image.new('RGB',(200,100),'white').save(image)
    class Runtime:
        def for_stage(self,**kwargs):return self
    def fail(*args,**kwargs):raise RuntimeError('simulated request failure')
    monkeypatch.setattr(g.model,'ask',fail)
    job={'group':'G000','ids':['T1'],'crop_dir':str(crop),'output':str(tmp_path/'glm')}
    result=g.run_jobs([job],[row()],image,image,{'T1':{'before':{'score':.4}}},
                      Runtime(),{},lambda *args,**kwargs:None)
    report=read(tmp_path/'glm/report.json')
    assert result[0]['state']=='failed' and report['detections']==[]
    assert report['geometry_failed'] and report['backend']=='glm'
    assert set(report['input_sha256'])=={'clean.png','frame.json'}


def test_prompt_freezes_styles_and_uses_cpu_score():
    p=g.geometry_prompt([row()],frame(),{'T1':{'before':{'score':.45}}})
    assert p['targets'][0]['cpu_similarity_percent']==45
    assert 'bold/italic/underline' in ' '.join(p['rules'])
    assert 'CPU' in json.dumps(p)


def test_single_box_keeps_model_coordinates_exactly_without_pixel_or_template_search():
    r=row();original=deepcopy(r)
    d={'index':0,'target_id':'T1','source_bbox':[12.25,11.5,92.75,39.5],
       'text':r['text'],'character_span':[0,len(r['text'])]}
    selected=g.select_geometry(r,[d],[200,100])
    assert selected['bbox_2d']==d['source_bbox']
    assert selected['bbox_policy']=='model_bbox_without_cpu_tightening_v1'
    assert 'template_score' not in selected
    assert r==original
    selected['bbox_2d'][0]=30
    assert d['source_bbox'][0]==12.25


@pytest.mark.parametrize('mode',['missing','duplicate','partial','wrong_owner','outside'])
def test_bad_model_geometry_leaves_prior_for_caller(mode):
    r=row();d={'index':0,'target_id':'T1','source_bbox':[12,11,92,39],
              'text':r['text'],'character_span':[0,len(r['text'])]}
    items=[d];neighbors=[]
    if mode=='missing':items=[]
    if mode=='duplicate':items=[d,d]
    if mode=='partial':d['text']='Hello'
    if mode=='wrong_owner':
        d['source_bbox']=[120,10,190,40]
        other=row('T2');other['bbox_2d']=d['source_bbox'];neighbors=[other]
    if mode=='outside':d['source_bbox']=[-1,10,90,40]
    before=deepcopy(r)
    with pytest.raises(ValueError):g.select_geometry(r,items,[200,100],neighbors)
    assert r==before


def test_multiline_retains_each_model_box_without_tightening_source_ink():
    r=row();r['visual_line_count']=2
    parts=[{'index':0,'target_id':'T1','character_span':[0,5],'source_bbox':[10,10,95,24]},
           {'index':1,'target_id':'T1','character_span':[6,11],'source_bbox':[10,27,95,40]}]
    # A blank source provides no ink boundaries to shrink to.
    children,_=g.propose_lines(r,parts,Image.new('RGB',(200,100),'white'))
    assert [c['bbox_2d'] for c in children]==[p['source_bbox'] for p in parts]
    assert [c['text'] for c in children]==['Hello','World']
