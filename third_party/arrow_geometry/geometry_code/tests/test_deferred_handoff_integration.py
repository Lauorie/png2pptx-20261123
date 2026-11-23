"""Offline checks across handoff, initial geometry, SAM filtering and missing review."""
import copy
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

import bbox_inputs
import bbox_pipeline
import bbox_geometry
import pipeline
import recall_review
import reconcile_stage
import style_pipeline
from scene_snapshot import canonical_rows
from test_bbox_pipeline import BASE, locks
from test_post_geometry_missing import row

ROOT = Path(__file__).resolve().parents[1]


def handoff(active=True):
    common = dict(identity_hint='locator', render_type='linework', grouping='single',
                  readonly_identities=[], requires_review=False)
    pending = dict(common, target_id='U', region_id='uncertain', bbox_native=[40,35,80,60],
                   task_role='missing_candidate', render_owner=False, extent='uncertain', requires_review=True)
    tasks = [dict(common, target_id='A', region_id='confirmed', bbox_native=[10,10,30,20],
                  render_owner=True, extent='complete')] if active else []
    return dict(case_id='case', tasks=tasks, deferred_candidates=[pending])


@pytest.mark.parametrize('active',[True,False])
def test_initial_geometry_then_deferred_check_then_missing_geometry(tmp_path,monkeypatch,active):
    source=Image.new('RGB',(100,80),'white')
    plan=bbox_inputs.normalize(handoff(active),(100,80),'case')
    order=[]
    monkeypatch.setattr(bbox_inputs,'load_case',lambda *args:(source,copy.deepcopy(plan)))
    def recover(clean,task,*args,**kwargs):
        order.append(('geometry',task['id']))
        assert task['id']!='U', 'Uncertain task must never enter initial geometry'
        if task['id'].startswith('MISSING_'):
            assert task['missing_only'] and ('check','LOCALIZATION_UNCERTAIN_U') in order
        geometry=row();geometry.update(id=task['id'],parent_id=task['id'])
        return dict(group=task['id'],state='completed',rows=[geometry],issues=[])
    monkeypatch.setattr(bbox_geometry,'recover_target',recover)
    monkeypatch.setattr(pipeline,'publish',lambda clean,results,*args,**kw:canonical_rows(results))
    monkeypatch.setattr(recall_review,'snapshot',lambda *args:None)
    monkeypatch.setattr(style_pipeline,'prepare_case_geometry',lambda clean,results,*args:(results,[]))
    monkeypatch.setattr(reconcile_stage,'run',lambda clean,results,*args:(results,[]))
    class Model:
        def ask(self,folder,prompt,images,validate,**kwargs):
            if 'candidate' not in prompt:
                order.append(('global','nominate'));return validate(dict(regions=[]))
            order.append(('check',prompt['candidate']['id']))
            assert (('geometry','A') in order)==active
            return validate(dict(decision='missing',reason='visible omission',regions=[dict(
                bbox=prompt['candidate']['bbox'],render_type='linework',grouping='single',label='confirmed')]))
    cfg={**BASE,'native_render':False,'post_geometry_missing_enabled':True,'display_side':300}
    result=bbox_pipeline.run_case('case',cfg,tmp_path,Model(),locks(),lambda *args:None)
    assert result['state']=='completed'
    assert order==([('geometry','A')] if active else [])+[
        ('global','nominate'),('check','LOCALIZATION_UNCERTAIN_U'),('geometry','MISSING_LOCALIZATION_UNCERTAIN_U_001')]


def test_disabled_missing_stage_cannot_silently_drop_deferred_candidates(tmp_path,monkeypatch):
    plan=bbox_inputs.normalize(handoff(),(100,80),'case')
    monkeypatch.setattr(bbox_inputs,'load_case',lambda *args:(Image.new('RGB',(100,80)),plan))
    with pytest.raises(ValueError,match='require post_geometry_missing_enabled'):
        bbox_pipeline.run_case('case',{**BASE,'post_geometry_missing_enabled':False},tmp_path,None,locks(),lambda *args:None)


def test_deferred_bbox_does_not_suppress_sam_candidate():
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    stage=load('full_pipeline_stage',ROOT/'checkpoint/runners/full_pipeline/stage.py')
    filtering=load('sam_coverage_filter',ROOT/'checkpoint/upstream/sam_supplement/code/containment_filter.py')
    raw=handoff()
    result=dict(regions=[dict(id='confirmed',extent='complete'),dict(id='uncertain',extent='uncertain')])
    baseline=stage.baseline_rows(raw,result)
    assert [r['original_target_id'] for r in baseline]==['A']
    sam=[dict(id='S',bbox=raw['deferred_candidates'][0]['bbox_native'],member_ids=[])]
    retained,excluded,_=filtering.filter_candidates(sam,baseline,.9)
    assert len(retained)==1 and not excluded
