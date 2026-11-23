from copy import deepcopy

import pytest
from PIL import Image

from shape_integration.texts import options, multiline_retry as m


def row():
    return {'id':'T1','text':'Hello World','kind':'text','visual_line_count':2,
        'line_layout_status':'requires_split','rotation_deg':0,'bbox_2d':[5,5,90,65],
        'font':'Arial','color':'#CC0000','runs':[
            {'text':'Hello ','color':'#CC0000','script':'normal'},
            {'text':'World','color':'#0000CC','script':'normal'}]}


def response():
    return {'parent_id':'T1','lines':[{'text':'Hello','bbox_2d':[10,10,70,30]},
                                     {'text':'World','bbox_2d':[10,40,70,60]}]}


def test_only_malformed_ordinary_target_is_requested_and_colors_preserved(tmp_path,monkeypatch):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    single={**row(),'id':'T2','visual_line_count':1}
    formula={**row(),'id':'T3','kind':'formula','latex':r'\frac{a}{b}'}
    original={'texts':[row(),single,formula]};before=deepcopy(original);calls=[]
    def ask(runtime,folder,prompt,images,validator,opt):
        calls.append(prompt['parent_id']);return validator(response())
    monkeypatch.setattr(m.model,'ask',ask)
    result=m.repair(original,source,tmp_path/'retry',None,options({}))
    assert calls==['T1'] and original==before
    a,b,*rest=result['texts']
    assert (a['text'],b['text'])==('Hello','World')
    assert a['runs'][0]['color']=='#CC0000' and b['runs'][0]['color']=='#0000CC'
    assert a['visual_line_count']==b['visual_line_count']==1
    assert rest==[single,formula]


@pytest.mark.parametrize('error',[ValueError('bad response'),RuntimeError('timeout')])
def test_failure_keeps_exact_original_and_does_not_abort(tmp_path,monkeypatch,error):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    def ask(*args,**kwargs):raise error
    monkeypatch.setattr(m.model,'ask',ask)
    original={'texts':[row()]}
    result=m.repair(original,source,tmp_path/'retry',None,options({}))
    assert result['texts']==original['texts']
    assert result['regions_needing_review'][0]['status']=='kept_original'


@pytest.mark.parametrize('kind',['missing','changed','overlap','one_box'])
def test_invalid_split_rejected(kind):
    raw=response()
    if kind=='missing':raw['lines'][1]['text']='Worl'
    if kind=='changed':raw['lines'][1]['text']='Earth'
    if kind=='overlap':raw['lines'][1]['bbox_2d']=raw['lines'][0]['bbox_2d']
    if kind=='one_box':raw['lines']=raw['lines'][:1]
    with pytest.raises(ValueError):m.validate(raw,row(),[100,100])
