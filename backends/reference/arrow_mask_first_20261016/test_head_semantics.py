import copy
import pytest
from PIL import Image
import head_semantics as hs
from transport.common import read_json


def row():
    return dict(id='G:R',kind='straight',start=[15,50],end=[85,50],via=[],heads=['none','triangle'],width_estimate=4)


class Model:
    def __init__(self,error=False):self.error=error;self.calls=0
    def ask(self,path,prompt,images,validator):
        self.calls+=1
        if self.error:raise RuntimeError('Unavailable')
        assert all(p.exists() for _,p in images)
        return validator(dict(heads={'R1':['stealth','none']}))


def test_head_binding_changes_no_geometry_and_keeps_old_hints(tmp_path):
    original=row();model=Model()
    result,issues,calls=hs.recover(Image.new('RGB',(100,100),'white'),[original],{},tmp_path,model)
    assert not issues and calls==model.calls==1
    for k in ['start','end','via','kind','id','width_estimate']:assert result[0][k]==original[k]
    assert result[0]['heads']==['stealth','none'] and original['heads']==['none','triangle']
    assert result[0]['start_head_type']=='stealth' and not result[0]['end_head']


def test_failed_head_request_keeps_path_and_reports_unknown(tmp_path):
    original=row();result,issues,calls=hs.recover(Image.new('RGB',(100,100),'white'),[original],{},tmp_path,Model(True))
    assert result[0]['start']==original['start'] and result[0]['end']==original['end']
    assert result[0]['heads']==['unknown','unknown'] and issues and calls==1


def test_known_offcrop_endpoint_uses_original_global_pixels(tmp_path):
    r=row();r['end']=[150,50]
    view=dict(view_to_global=[[.5,0,20],[0,.5,20],[0,0,1]])
    result,issues,_=hs.recover(Image.new('RGB',(100,100),'white'),[r],{},tmp_path,Model(),
        full_image=Image.new('RGB',(200,100),'white'),view=view)
    ends=read_json(tmp_path/'sources.json')[0]['ends']
    assert ends[0]['frame']=='local' and ends[1]['frame']=='global'
    assert result[0]['end']==[150,50] and not issues


def test_equivalent_flat_head_map_needs_no_model_retry():
    raw={'R1':['stealth','none']}
    assert hs.validate(raw,['R1'])=={'heads':raw}
    with pytest.raises(ValueError):hs.validate({'R2':['none','none']},['R1'])


def test_complete_head_answer_with_trailing_prose_is_unambiguous():
    from transport.model import parse
    raw='{"heads":{"R1":["stealth","none"]}}'
    assert parse(raw+'\n\nThe rear edge is notched.')=={'heads':{'R1':['stealth','none']}}
    with pytest.raises(ValueError):parse(raw+'\n'+raw)
