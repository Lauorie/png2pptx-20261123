import copy
import threading

import pytest
from PIL import Image

import coverage_stage as coverage
import pipeline
from transport.common import read_json, write_json


def old_row():
    return dict(id='old',parent_id='old',label='A1',kind='straight',start=[10,20],end=[80,20],via=[],heads=['none','none'],
        style=dict(color='#000000',width_px=1,dash=dict(type='solid')))


def setup_case(tmp_path,monkeypatch):
    image=Image.new('RGB',(100,80),'white');image.save(tmp_path/'clean.png')
    (tmp_path/'result').mkdir();image.save(tmp_path/'result/boxes.png')
    monkeypatch.setattr(coverage.shared_repair,'draw_current',lambda crop,rows,view,path:crop.save(path))
    return image,dict(cpu=threading.Semaphore(1))


@pytest.mark.parametrize('mode,count,has_inventory',[
    ('two_images',2,False),('boxes_and_inventory',3,True)])
def test_actual_request_matches_selected_input_protocol(tmp_path,monkeypatch,mode,count,has_inventory):
    image,locks=setup_case(tmp_path,monkeypatch);rows=[old_row()];before=copy.deepcopy(rows)
    class Model:
        def ask(self,folder,prompt,images,validator):
            assert len(images)==count
            assert ('already_reconstructed' in prompt)==has_inventory
            assert all(path.exists() for _,path in images)
            if not has_inventory:
                assert all('BOXES' not in label for label,_ in images)
                assert set(prompt)=={'task','size','instructions','output_schema'}
            return validator(dict(regions=[dict(bbox=[10,30,60,60],description='visible short connector')]))
    answer=coverage.discover(image,rows,tmp_path,dict(coverage_input_mode=mode),Model(),locks)
    assert answer['regions'][0]['bbox']==[10,30,60,60]
    assert rows==before
    assert len(read_json(tmp_path/'coverage/input/inventory.json'))==1
    policy=read_json(tmp_path/'coverage/input_policy.json')
    assert policy['images_sent']==count and policy['numeric_inventory_sent']==has_inventory


def test_invalid_mode_fails_before_request(tmp_path,monkeypatch):
    image,locks=setup_case(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='coverage_input_mode'):
        coverage.discover(image,[old_row()],tmp_path,dict(coverage_input_mode='typo'),None,locks)


def test_discovery_failure_preserves_reconstruction(tmp_path,monkeypatch):
    image,locks=setup_case(tmp_path,monkeypatch);results=[dict(group='base',rows=[old_row()])];saved=copy.deepcopy(results)
    class Model:
        def ask(self,*args):raise TimeoutError('deadline')
    after,issues,changes=coverage.run(image,results,tmp_path,{},Model(),locks)
    assert after==saved and results==saved and changes['added']==[]
    assert issues==[dict(stage='coverage',error='deadline')]
    assert read_json(tmp_path/'coverage_status.json')['state']=='failed'


def test_commits_explicit_patches_and_surfaces_local_failures(tmp_path,monkeypatch):
    image,locks=setup_case(tmp_path,monkeypatch)
    original=[dict(group='base',rows=[old_row(),dict(old_row(),id='kept',parent_id='kept')])];saved=copy.deepcopy(original)
    tasks=[dict(bbox=[10,30,60,60],description='visible short connector')]
    monkeypatch.setattr(coverage,'discover',lambda *a:dict(regions=tasks))
    patch=dict(group='missing1',replace=['old'],rows=[dict(old_row(),id='new')],issues=[],status='repair')
    def execute(clean,regions,current,phase,out,config,model,locks,pool,notify):
        assert regions==tasks and [r['id'] for r in current]==['old','kept']
        notify(1,1)
        return [patch],[dict(stage='missing',error='another local request failed')]
    monkeypatch.setattr(coverage.parallel_repairs,'execute',execute)
    published=[];monkeypatch.setattr(pipeline,'publish',lambda clean,results,out:published.append(copy.deepcopy(results)))
    after,issues,changes=coverage.run(image,original,tmp_path,dict(parallel_repairs=True),None,locks)
    assert original==saved and [r['id'] for g in after for r in g['rows']]==['kept','new']
    assert changes['replaced']==['old'] and changes['added']==['new'] and len(issues)==1 and published


def test_main_entry_can_resume_into_same_coverage_stage(tmp_path,monkeypatch):
    source=tmp_path/'baseline/cases/sample';stage=source/'stages/continuation';stage.mkdir(parents=True)
    image=Image.new('RGB',(100,80),'white');image.save(source/'clean.png')
    write_json(stage/'scene.json',dict(size=[100,80],arrows=[old_row()]))
    out=tmp_path/'run/cases/sample';calls=[]
    def publish(clean,results,dest):
        (dest/'result').mkdir(exist_ok=True)
        for name in ['styled','instances','overlap','boxes']:clean.save(dest/'result'/f'{name}.png')
        (dest/'arrows.pptx').write_bytes(b'export fixture')
        return [copy.deepcopy(r) for g in results for r in g['rows']]
    def run(clean,results,*args,**kwargs):
        calls.append('shared coverage stage')
        return results,[],dict(added=[],replaced=[],regions=0)
    monkeypatch.setattr(pipeline,'publish',publish)
    monkeypatch.setattr(coverage,'run',run)
    monkeypatch.setattr(pipeline.native,'write',lambda size,rows,styles,path:path.write_bytes(b'export fixture'))
    def render(ppt,out,size):out.mkdir(parents=True,exist_ok=True);Image.new('RGB',size).save(out/'styled.png')
    monkeypatch.setattr(pipeline.native,'render',render)
    cfg=dict(coverage_resume_from=str(tmp_path/'baseline'),native_render=True)
    result=pipeline.run_case('sample',cfg,out,None,dict(cpu=threading.Semaphore(1)),lambda *a:None)
    assert calls==['shared coverage stage'] and result['native_rendered'] and result['added_paths']==0
    assert not result['identity_rerun'] and read_json(out/'change_audit.json')['unmodified_baseline_preserved']
