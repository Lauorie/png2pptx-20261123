"""No service calls: omission ordering, single-target inputs and coordinate safety."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import post_geometry_missing as missing
import coverage_visual
from transport.common import read_json as read, write_json as write
from test_bbox_pipeline import BASE, locks

spec = importlib.util.spec_from_file_location('omission_filter', ROOT/'checkpoint/upstream/sam_supplement/code/containment_filter.py')
filter_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(filter_module)


def row():
    return dict(id='old', parent_id='old', kind='straight', start=[15., 15.],
                end=[85., 15.], via=[], heads=['none', 'none'],
                style=dict(width_px=1., color='#000000', dash={'preset': 'solid'},
                           start={'type': 'none'}, end={'type': 'none'}))


def test_coverage_uses_candidate_area_and_no_old_box_union():
    groups = [dict(id='S', bbox=[9.9, 10, 40, 30], member_ids=['M'])]
    baseline = [dict(id='B', bbox=[10, 5, 90, 40])]
    before = copy.deepcopy((groups, baseline))
    assert len(filter_module.filter_candidates(groups, baseline)[0]) == 1
    kept, removed, audit = filter_module.filter_candidates(groups, baseline, .9)
    assert not kept and removed[0]['coverage_matches'][0]['candidate_coverage'] > .99
    assert (groups, baseline) == before and audit['geometry_coverage_verified'] is False
    split = [dict(id='B1', bbox=[0, 0, 25, 40]), dict(id='B2', bbox=[25, 0, 45, 40])]
    assert len(filter_module.filter_candidates(groups, split, .9)[0]) == 1


def test_queue_dedups_near_identical_only_and_keeps_global_inside_big_box():
    global_regions = [dict(bbox=[10, 10, 40, 40]), dict(bbox=[30, 10, 60, 40])]
    sam = [dict(id='S1', bbox=[10.1, 10, 40, 40], member_ids=['M1'])]
    candidates, aliases = missing.candidate_queue(global_regions, sam, (100, 100))
    assert len(candidates) == 2  # overlapping neighbors remain two requests
    assert candidates[0]['source'] == 'GLOBAL' and aliases[0]['candidate'] == 'SAM_S1'
    assert candidates[0]['bbox_native'] == [10, 10, 40, 40]


def test_deferred_sources_join_checks_and_near_identical_aliases_keep_provenance():
    deferred=[dict(target_id='R0002',region_id='uncertain_region',bbox_native=[10,10,40,40],
                   extent='uncertain',observation_ids=['G_R2']),
              dict(target_id='R0003',region_id='other_region',bbox_native=[50,50,80,70],extent='uncertain')]
    before=copy.deepcopy(deferred)
    sam=[dict(id='S1',bbox=[50,50,80,70],member_ids=['mask1'])]
    tasks,aliases=missing.candidate_queue([dict(bbox=[10,10,40,40])],sam,(100,80),deferred)
    assert len(tasks)==2 and len(aliases)==2
    assert tasks[0]['aliases'][0]['origin']['observation_ids']==['G_R2']
    assert tasks[1]['source']=='LOCALIZATION_UNCERTAIN'
    assert tasks[1]['aliases'][0]['source']=='SAM'
    assert tasks[1]['origin']['bbox_native']==[50,50,80,70] and deferred==before


def test_geometry_projection_curves_shapes_and_dashes_applied_once():
    original = row()
    original.update(beziers=[[[15, 15], [25, 10], [70, 10], [85, 15]]],
                    contours=[[[15, 15], [25, 10], [35, 15]]],
                    shape_parts=[dict(id='body', commands=[['M',15,15],['L',25,15],['Z']], bbox_2d=[15,10,25,15])])
    original['style']['dash'].update(pattern_px=[4, 2], mark_px=4, gap_px=2)
    before = copy.deepcopy(original)
    frame = dict(global_to_view=[[2,0,-10],[0,2,-20],[0,0,1]])
    result = coverage_visual.project([original], frame)[0]
    assert result['start'] == [20,10] and result['beziers'][0][1] == [40,0]
    assert result['contours'][0][1] == [40,0]
    assert result['shape_parts'][0]['commands'][1] == ['L',40,10]
    assert result['style']['width_px'] == 2 and result['style']['dash']['mark_px'] == 8
    assert result['style']['dash']['pattern_px'] == [8,4] and original == before


def test_visible_blue_geometry_and_gray_source_are_separate(tmp_path):
    source = Image.new('RGB', (100,80), 'white')
    source.putpixel((50,50), (0,0,0))
    identity = dict(global_to_view=[[1,0,0],[0,1,0],[0,0,1]])
    coverage_visual.render(source,[row()],identity,tmp_path/'current.png',labels=False)
    current = Image.open(tmp_path/'current.png')
    assert current.getpixel((50,50))[0] in range(157,160)
    r,g,b = current.getpixel((50,15)); assert b > r and b > g
    assert source.getpixel((50,50)) == (0,0,0)


def test_reply_clamps_coordinates_but_forbids_editing_old_geometry():
    reply = dict(decision='missing', regions=[dict(bbox=[-1,2,102,50],render_type='linework')])
    assert missing.check_reply(reply,(100,80))['regions'][0]['bbox'] == [0,2,99,50]
    assert reply['regions'][0]['bbox'][0] == -1
    with pytest.raises(ValueError): missing.check_reply(dict(reply,replace=['old']),(100,80))
    with pytest.raises(ValueError): missing.check_reply(dict(reply,decision='covered'),(100,80))
    with pytest.raises(ValueError): missing.check_reply(dict(decision='existing_geometry_issue',regions=[]),(100,80))


def test_reviewed_regions_consolidated_before_geometry_and_keep_existing_readonly(tmp_path, monkeypatch):
    source = Image.new('RGB', (100, 80), 'white')
    old = [dict(group='old', state='completed', rows=[row()])]
    before = copy.deepcopy(old)
    def inspect(clean, rows, candidate, folder, config, model, locks, sam_case=None):
        write(Path(folder)/'resolution.json', dict(decision='missing', unresolved=False))
        return [dict(bbox=[30, 30, 70, 60], render_type='linework', label='first diagonal'),
                dict(bbox=[31, 30, 71, 60], render_type='linework', label='other diagonal')], []
    received = []
    def recover(clean, target, *args):
        received.append(copy.deepcopy(target))
        return dict(group=target['id'], rows=[], issues=[])
    monkeypatch.setattr(missing, 'inspect_candidate', inspect)
    monkeypatch.setattr(missing.bbox_geometry, 'recover_target', recover)
    cfg = {**BASE, 'post_geometry_global_discovery_enabled': False,
           'missing_box_consolidation_enabled': True}
    final, _ = missing.run(source, old, tmp_path/'missing', cfg, FakeModel(), locks(), 'case',
                          deferred_candidates=[dict(target_id='R', bbox_native=[20, 25, 80, 65])])
    assert len(received) == 1 and received[0]['locator'] == ''
    assert received[0]['bbox_native'] == [30, 30, 71, 60]
    assert len(received[0]['members']) == 2 and received[0]['existing_geometry'][0]['id'] == 'old'
    assert final[0] == before[0] and old == before
    assert len(read(tmp_path/'missing/additions_before_consolidation.json')) == 2
    assert 'existing_geometry' not in read(tmp_path/'missing/additions.json')[0]


class FakeModel:
    def __init__(self, expand=False): self.calls=[]; self.expand=expand
    def ask(self, folder, prompt, images, validate, **kwargs):
        self.calls.append((Path(folder),copy.deepcopy(prompt),images,kwargs))
        for _,path in images: assert Path(path).is_file()
        if 'candidate' in prompt:
            if self.expand: return validate(dict(decision='needs_context',reason='crop boundary',regions=[]))
            if prompt['candidate']['id'].startswith('SAM'):
                return validate(dict(decision='not_arrow',reason='rectangle border',regions=[]))
            return validate(dict(decision='missing',reason='visible connector absent',regions=[
                dict(bbox=prompt['candidate']['bbox'],render_type='linework',grouping='single',label='visible line')]))
        if 'targets' in prompt:
            b=prompt['targets'][0]['bbox']; y=(b[1]+b[3])/2
            assert any(label.startswith('CURRENT') for label,_ in images)
            return validate(dict(status='draw',nodes=[['a',b[0],y],['b',b[2],y]],routes=[dict(
                id='R1',paths=[dict(id='p',heads=['none','triangle'],segments=[dict(kind='line',nodes=['a','b'])])])]))
        # The global view must be made from actual geometry, with no bbox inventory.
        assert len(images)==2 and 'already_reconstructed' not in prompt
        return validate(dict(regions=[dict(bbox=[30,180,180,230],description='visible omission')]))


def test_two_sources_single_checks_append_only_and_real_geometry_path(tmp_path):
    source = Image.new('RGB',(100,80),'white')
    old = [dict(group='old', state='completed', rows=[row()])]
    before = copy.deepcopy(old)
    sam = tmp_path/'sam/cases/case'
    write(sam/'retained_candidates.json',[dict(id='S1',bbox=[70,40,90,60],member_ids=[])])
    write(sam/'status.json',dict(state='completed',issues=[]))
    model = FakeModel()
    cfg={**BASE,'display_side':300,'post_geometry_sam_run':str(tmp_path/'sam')}
    events=[]
    final,issues=missing.run(source,old,tmp_path/'missing',cfg,model,locks(),'case',events.append)
    checks=[c for c in model.calls if 'candidate' in c[1]]
    assert len(checks)==2 and all(len(c[2])==3 for c in checks)
    assert {c[1]['candidate']['id'] for c in checks}=={'GLOBAL_001','SAM_S1'}
    assert model.calls[0][0].parent.name=='global'
    assert events==['global_missing_discovery','single_bbox_missing_check','missing_geometry_and_critic']
    assert not issues and len(final)==2 and old==before and final[0]==before[0]
    assert len(final[1]['rows'])==1 and final[1]['rows'][0]['parent_id'].startswith('MISSING_GLOBAL')
    assert 'existing_geometry' not in final[1]['discovery_task']
    assert read(tmp_path/'missing/candidates.json')['global_proposals_filtered_by_localization'] is False


def test_unresolved_context_does_not_add_guess(tmp_path):
    model=FakeModel(expand=True)
    candidate=dict(id='GLOBAL_001',source='GLOBAL',bbox_native=[40,35,60,45],origin={})
    additions,issues=missing.inspect_candidate(Image.new('RGB',(160,120),'white'),[row()],candidate,
        tmp_path,{**BASE,'context_padding':0},model,locks())
    assert not additions and issues[0]['unresolved'] and len(model.calls)==2
    first=read(tmp_path/'view_00/transform.json')['bbox_global']
    second=read(tmp_path/'view_01/transform.json')['bbox_global']
    assert second[2]-second[0] > first[2]-first[0]


@pytest.mark.parametrize('decision',['covered','not_arrow','uncertain','missing'])
def test_deferred_candidate_requires_visual_missing_decision_before_geometry(tmp_path,decision):
    class DeferredModel(FakeModel):
        def ask(self,folder,prompt,images,validate,**kwargs):
            if 'candidate' in prompt:
                self.calls.append((Path(folder),copy.deepcopy(prompt),images,kwargs))
                assert set(prompt['candidate'])=={'id','bbox'}
                assert prompt['candidate']['id']=='LOCALIZATION_UNCERTAIN_R0002'
                assert any(label=='CURRENT' for label,_ in images)
                regions=[] if decision!='missing' else [dict(bbox=prompt['candidate']['bbox'],
                            render_type='linework',grouping='single',label='visible missing connection')]
                return validate(dict(decision=decision,reason='source versus current evidence',regions=regions))
            if 'targets' in prompt:return super().ask(folder,prompt,images,validate,**kwargs)
            self.calls.append((Path(folder),copy.deepcopy(prompt),images,kwargs))
            return validate(dict(regions=[]))
    source=Image.new('RGB',(100,80),'white')
    old=[dict(group='old',state='completed',rows=[row()])];before=copy.deepcopy(old)
    pending=[dict(target_id='R0002',bbox_native=[25,35,75,60],extent='uncertain',
                  render_type='block_arrow',identity_hint='untrusted old description',observation_ids=['G_R2'])]
    model=DeferredModel()
    final,issues=missing.run(source,old,tmp_path/'missing',{**BASE,'display_side':300},model,locks(),
                             'case',deferred_candidates=pending)
    assert old==before and final[:len(old)]==before
    geometry_calls=[c for c in model.calls if 'targets' in c[1]]
    assert bool(geometry_calls)==(decision=='missing')
    assert len(final)==len(old)+(decision=='missing')
    candidates=read(tmp_path/'missing/candidates.json')
    assert candidates['deferred_localization_candidates']==1
    assert candidates['tasks'][0]['origin']['observation_ids']==['G_R2']
    resolutions=read(tmp_path/'missing/check_resolutions.json')
    assert resolutions['LOCALIZATION_UNCERTAIN_R0002']['decision']==decision
    unresolved=read(tmp_path/'missing/unresolved_candidates.json')
    assert bool(unresolved)==(decision == 'uncertain')
    assert bool(issues)==(decision == 'uncertain')
    pending_repairs=read(tmp_path/'missing/pending_existing_repairs.json')
    assert not pending_repairs


def test_sam_mask_projection_only_selected_candidate(tmp_path):
    from transport.common import sha256
    mask=Image.new('L',(100,80));mask.putpixel((50,30),255);mask.save(tmp_path/'mask.png')
    write(tmp_path/'raw_candidates.json',[dict(id='G_M1',view='G',mask_path=str(tmp_path/'mask.png'),mask_sha256=sha256(tmp_path/'mask.png'))])
    write(tmp_path/'sam/G/frame.json',dict(crop_bounds_global_half_open=[0,0,100,80],actual_crop_input_size_px=[100,80]))
    frame=dict(size=[100,80],global_to_view=[[2,0,-19.5],[0,2,-19.5],[0,0,1]],
               view_to_global=[[.5,0,9.75],[0,.5,9.75],[0,0,1]])
    candidate=dict(source='SAM',origin=dict(member_ids=['G_M1']))
    image=missing._sam_overlay(Image.new('RGB',(100,80),'white'),frame,candidate,tmp_path)
    assert image.getpixel((80,40))!=(255,255,255) and image.getpixel((30,30))==(255,255,255)
    candidate['origin']['member_ids']=[]
    assert missing._sam_overlay(Image.new('RGB',(100,80),'white'),frame,candidate,tmp_path).getpixel((80,40))==(255,255,255)


def test_system_override_participates_in_request_identity(tmp_path, monkeypatch):
    import transport.model as module
    config=dict(base_url='http://offline.invalid',model='offline',temperature=.1,max_tokens=113000,
                thinking={'type':'enabled'},api_concurrency=1,format_retries=0,transport_retries=0)
    systems=[]
    def request(cfg,system,prompt,images,folder):
        folder.mkdir(parents=True,exist_ok=True); systems.append(system)
        return '{"regions":[]}'
    monkeypatch.setattr(module,'request',request)
    model=module.Model(config,system='GEOMETRY')
    folder=tmp_path/'request'
    model.ask(folder,{},[],lambda a:a,system='MISSING')
    model.ask(folder,{},[],lambda a:a,system='MISSING')
    model.ask(folder,{},[],lambda a:a)
    assert systems==['MISSING','GEOMETRY']
    assert read(folder/'identity.json')['system']=='MISSING' and model.system=='GEOMETRY'


def test_core_integration_geometry_before_missing_before_dedup_style(tmp_path, monkeypatch):
    import pipeline, reconcile_stage, style_pipeline
    from test_bbox_pipeline import PipelineContract, Model
    root=PipelineContract().fixture(tmp_path)
    order=[]
    def supplement(clean,results,*args,**kwargs):
        assert results and results[0]['rows']
        order.append('missing');return results,[]
    def dedup(clean,results,*args,**kwargs): order.append('dedup');return results,[]
    def style(clean,results,*args,**kwargs): order.append('style');return results,[]
    monkeypatch.setattr(missing,'run',supplement)
    monkeypatch.setattr(reconcile_stage,'run',dedup)
    monkeypatch.setattr(style_pipeline,'finish_case',style)
    cfg={**BASE,'post_geometry_missing_enabled':True,'discovery_provider':'bbox_handoff',
         'pipeline_mode':'full','coverage_rounds':0,'native_render':False}
    pipeline.run_case('x',cfg,root/'cases/x',Model(),locks(),lambda *args:None)
    assert order==['missing','dedup','style']
