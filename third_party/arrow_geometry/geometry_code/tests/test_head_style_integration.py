"""Integration boundaries: new heads cannot move geometry or rewrite shafts."""
import copy
import importlib.util
import json
import sys
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]

import pytest
from PIL import Image

import evidence
import head_style_recovery
import style_pipeline
from simple_style_contract import Contract, merge_style


def line(ident='L1'):
    return dict(id=ident, kind='polyline', start=[20.125,40.25], via=[[90.5,40.25]],
                end=[180.875,80.5], beziers=[], heads=['none','triangle'],
                style=dict(color='#123456',width_px=3.,dash=dict(type='solid',preset='solid',pattern_px=[]),
                           start=None,end=dict(type='triangle',width='med',length='med')))


def test_old_review_then_heads_then_final_render(tmp_path,monkeypatch):
    row=line(); calls=[]; render_heads=[]
    def render(rows,out,*args):
        render_heads.append(copy.deepcopy(rows[0]['heads']))
        out.mkdir(parents=True,exist_ok=True);p=out/'render.png';Image.new('RGB',(220,120),'white').save(p);return p
    monkeypatch.setattr(style_pipeline,'render',render)
    class Model:
        def ask(self,folder,prompt,images,validator,**kwargs):
            if folder.name=='critic':
                calls.append('critic')
                assert prompt['targets'][0]['locator_tags']=={'start':'1S','end':'1E'}
                return validator({'checks':[{'id':row['id'],'verdict':'accept','reason':''}]})
            calls.append('heads')
            assert set(prompt['target'])=={'id','start','end','locator_tags'}
            assert 'current_styles' not in prompt and 'feedback' not in prompt
            assert len(images)==3 and 'REFERENCE' in images[-1][0]
            assert all('RENDER' not in name and 'detail' not in str(path) for name,path in images)
            assert kwargs['system'].startswith('Recover visible line-end styles')
            return validator({'id':row['id'],'start':'circle','end':'none'})
    cfg=dict(style_profile='three_heads_mid_min2',style_entry='review_existing',style_refine_rounds=1,
             style_locator_layout='adaptive_short_tags',head_recovery_enabled=True,head_recovery_reference=True)
    original=copy.deepcopy(row)
    result,issues=style_pipeline.recover(Image.new('RGB',(220,120),'white'),[row],{},tmp_path,cfg,Model(),{})
    assert calls==['critic','heads'] and render_heads[-1]==['oval','none'] and not issues
    assert row==original
    assert (tmp_path/'final').resolve()==(tmp_path/'head_recovery/render').resolve()
    new=Contract().enforce_rows([merge_style(row,result[0])])[0]
    assert new['head_source'].startswith('head_prompt_v004') and new['head_recovery']['critic_run'] is False
    assert new['pre_head_style_review']['state']=='reviewed' and 'review' not in new['style']
    assert new['start']==row['start'] and new['via']==row['via'] and new['end']==row['end']
    assert all(new['style'][k]==row['style'][k] for k in ['color','width_px','dash'])


def test_failed_heads_retain_previous_and_blocks_are_unchanged(tmp_path):
    a=line();b=line('block');b['kind']='block_arrow'
    class Model:
        def ask(self,*args,**kwargs):raise RuntimeError('simulated request failure')
    rows=[a,b];original=copy.deepcopy(rows)
    result,issues=head_style_recovery.recover(rows,tmp_path,{'locator_paths':{'1':'L1'}},tmp_path/'heads',{},Model(),(220,120))
    assert result==original and rows==original
    assert len(issues)==1 and issues[0]['id']=='L1'


def test_freeze_reviewed_handoff_preserves_deferred_and_source(tmp_path):
    path=Path(__file__).resolve().parents[1]/'checkpoint/runners/full_pipeline/prepare.py'
    spec=importlib.util.spec_from_file_location('prepare_head_pipeline_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';root=tmp_path/'new'
    cfg={k:True for k in ['review_inherit_existing_type','review_omit_candidate_labels','review_defer_uncertain_to_missing']}
    cfg['post_geometry_missing_enabled']=True
    module.write(source/'config.json',cfg)
    case=source/'cases/task_0001'
    module.write(case/'status.json',{'state':'completed_with_issues'})
    module.write(case/'result.json',{'regions':[]})
    handoff={'tasks':[{'id':'initial'}],'deferred_candidates':[{'id':'uncertain'}]}
    module.write(case/'reconstruction_handoff.json',handoff)
    image=source/'inputs/task_0001/source.png';image.parent.mkdir(parents=True)
    Image.new('RGB',(40,30),'white').save(image)
    old={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    module.freeze_localization(root,source,['task_0001'],cfg)
    assert module.read(root/'localization/cases/task_0001/reconstruction_handoff.json')==handoff
    assert module.read(root/'localization/config.json')['new_local_review_calls'] is False
    assert old=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='Deferred'):
        module.freeze_localization(tmp_path/'invalid',source,['task_0001'],{**cfg,'post_geometry_missing_enabled':False})
