import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]
import geometry_critic as gc


def row(source='glm_segmented_points'):
    return dict(id='G1:T1',geometry_owner_id='G1:T1',kind='polyline',start=[20.,30.],
                via=[[40.,30.]],end=[60.,50.],beziers=[],heads=['unknown','unknown'],
                coordinate_source=source,style={'test':'preserve until style stage'})


class Model:
    def __init__(self, decisions):self.decisions=list(decisions);self.calls=[]
    def ask(self,folder,q,images,validator):
        self.calls.append((q,images))
        action=self.decisions.pop(0)
        if action=='fail':raise RuntimeError('injected request failure')
        if action=='refine':
            answer=copy.deepcopy(q['current'])
            answer['nodes'][1][2]+=5
            return validator(dict(decision='refine',reason='visible bend offset',geometry=answer))
        return validator(dict(decision=action,**({'reason':'clipped'} if action=='uncertain' else {})))


def config(**kw):
    return dict(geometry_critic_enabled=True,geometry_critic_max_refinements=2,
                geometry_critic_long_side=160,**kw)


def test_crop_roundtrip_and_degenerate_extent():
    d=gc.as_points([row()]);f=gc.frame_for(d,(100,80),long_side=160)
    restored=gc.transform(gc.transform(d,f),f,True)
    np.testing.assert_allclose([n[1:] for n in restored['nodes']],[n[1:] for n in d['nodes']],atol=1e-12)
    r=row();r.update(via=[],start=[0,10],end=[70,10])
    f=gc.frame_for(gc.as_points([r]),(100,80))
    assert f['bbox'][0]==0 and f['bbox'][3]>f['bbox'][1]


def test_accept_keeps_exact_geometry_and_metadata(tmp_path):
    original=row();model=Model(['accept'])
    after,state=gc.refine_target(Image.new('RGB',(100,80),'white'),[original],tmp_path,model,config())
    assert after==[original] and state['accepted'] and state['revisions']==0
    q,images=model.calls[0]
    assert len(images)==2
    assert set(q)=={'task','size','current','instructions','output_schema'}
    assert 'source_masks' not in str(q) and 'locator' not in str(q)


def test_refine_then_accept_maps_to_original_coordinates(tmp_path):
    original=row();m=Model(['refine','accept'])
    after,state=gc.refine_target(Image.new('RGB',(100,80)),[original],tmp_path,m,config())
    assert state['revisions']==1 and state['accepted'] and state['calls']==2
    assert after[0]['id']==original['id'] and after[0]['via']!=original['via']
    assert after[0]['start']==original['start'] and after[0]['end']==original['end']
    assert 'style' not in after[0] and original['style']=={'test':'preserve until style stage'}


def test_failure_retains_last_complete_revision(tmp_path):
    image=Image.new('RGB',(100,80));m=Model(['refine','fail'])
    after,state=gc.refine_target(image,[row()],tmp_path,m,config())
    assert state['state']=='failed' and state['revisions']==1
    assert after[0]['via']!=row()['via'] and state['retained_previous']


def test_two_refinements_are_the_limit(tmp_path):
    m=Model(['refine','refine','refine'])
    after,state=gc.refine_target(Image.new('RGB',(100,80)),[row()],tmp_path,m,config())
    assert len(m.calls)==2 and state['revisions']==2
    assert state['state']=='budget_exhausted' and not state['accepted']


def test_uncertain_is_not_accept_and_does_not_mutate(tmp_path):
    original=row();after,state=gc.refine_target(Image.new('RGB',(100,80)),[original],tmp_path,Model(['uncertain']),config())
    assert after==[original] and not state['accepted'] and state['state']=='uncertain'


def test_mask_gate_and_shared_prompt(tmp_path):
    r=row('mask_no_skeleton');m=Model([]);image=Image.new('RGB',(100,80))
    after,support,issues,calls=gc.recover(image,[r],{r['id']:None},tmp_path/'off',m,config())
    assert after==[r] and calls==0 and not m.calls
    m=Model(['accept'])
    after,support,issues,calls=gc.recover(image,[r],{r['id']:None},tmp_path/'on',m,config(geometry_critic_mask_enabled=True))
    assert calls==1 and after==[r]
    m2=Model(['accept']);gc.refine_target(image,[row()],tmp_path/'glm',m2,config())
    assert m.calls[0][0]==m2.calls[0][0]


def test_accept_preserves_mask_bezier_exactly(tmp_path):
    r=row('mask_no_skeleton');r.update(kind='curve',via=[],beziers=[[[20,30],[25,5],[55,70],[60,50]]])
    after,state=gc.refine_target(Image.new('RGB',(100,80)),[r],tmp_path,Model(['accept']),config())
    assert after==[r] and state['accepted']
    doc=gc.as_points([r]);assert doc['routes'][0]['paths'][0]['segments'][0]['kind']=='curve'


def test_invalid_extra_target_or_style_is_rejected():
    d=gc.as_points([row()]);d['routes'][0]['id']='R2'
    with pytest.raises(ValueError):gc.validate(dict(decision='refine',reason='x',geometry=d),(100,80))
    d=gc.as_points([row()]);d['routes'][0]['heads']=['none','triangle']
    with pytest.raises(ValueError):gc.validate(dict(decision='refine',reason='x',geometry=d),(100,80))


def scaled_geometry(orientation, factors):
    native_size=np.array([400,400]);factors=np.asarray(factors,float)
    pts=np.array([[200.,150.],[200.,250.]]) if orientation=='vertical' else np.array([[150.,200.],[250.,200.]])
    local=(pts+.5)*factors-.5
    r=row();r.update(start=local[0].tolist(),end=local[1].tolist(),via=[])
    size=np.rint(native_size*factors).astype(int).tolist()
    view=dict(size=size,global_size=native_size.tolist(),view_to_global=[
        [1/factors[0],0,.5/factors[0]-.5],
        [0,1/factors[1],.5/factors[1]-.5],[0,0,1]])
    return r,size,view


@pytest.mark.parametrize('orientation',['vertical','horizontal'])
@pytest.mark.parametrize('factors',[(1.,1.),(3.375,3.375),(2.2,5.5)])
def test_minimum_original_span_is_independent_of_parent_upsampling(orientation,factors):
    r,size,view=scaled_geometry(orientation,factors)
    frame=gc.frame_for(gc.as_points([r]),size,source_view=view)
    axis=0 if orientation=='vertical' else 1
    # floor(left), ceil(right) and the inclusive center add fewer than three
    # parent pixels; the 50-original-pixel minimum is never rescaled away.
    assert 50<=frame['crop_extent_original_px'][axis]<50+3/factors[axis]+1e-9
    assert frame['min_extent_unit']=='original_image_pixels'
    assert frame['min_extent']==25.
    np.testing.assert_allclose(frame['min_extent_local'],np.asarray(factors)*25)
    alternate=gc.frame_for(gc.as_points([r]),size,source_view=view,long_side=2048)
    assert alternate['bbox']==frame['bbox']
    restored=gc.transform(gc.transform(gc.as_points([r]),frame),frame,True)
    np.testing.assert_allclose([n[1:] for n in restored['nodes']],[n[1:] for n in gc.as_points([r])['nodes']],atol=1e-12)


def test_red_arrow_crop_retains_full_head_extent():
    r=row();r.update(start=[166.,789.],end=[166.,295.],via=[])
    view=dict(size=[324,1080],view_to_global=[[1/3.375,0,911.6481481481482],[0,1/3.375,158.64814814814815],[0,0,1]])
    f=gc.frame_for(gc.as_points([r]),view['size'],source_view=view)
    assert f['bbox'][2]-f['bbox'][0]==171
    assert f['crop_extent_original_px'][0]==pytest.approx(171/3.375)
    left=(f['bbox'][0]-.5)/3.375+911.6481481481482+.5
    right=(f['bbox'][2]-.5)/3.375+911.6481481481482+.5
    assert left<943 and right>976


def test_recover_forwards_view_to_every_review(tmp_path):
    r,size,view=scaled_geometry('vertical',(3.375,3.375));model=Model(['refine','accept'])
    result,_,_,calls=gc.recover(Image.new('RGB',size),[r],{r['id']:None},tmp_path,model,config(),source_view=view)
    assert calls==2
    frames=list(tmp_path.glob('*/revision_*/input/frame.json'))
    assert len(frames)==2
    for path in frames:
        f=json.loads(path.read_text())
        np.testing.assert_allclose(f['min_extent_local'],[84.375,84.375])


def test_saved_geometry_frame_is_used_for_offline_replay(tmp_path):
    r,size,view=scaled_geometry('horizontal',(2.,4.));r['geometry_frame']=view
    gc.refine_target(Image.new('RGB',size),[r],tmp_path,Model(['accept']),config())
    frame=json.loads((tmp_path/'revision_00/input/frame.json').read_text())
    np.testing.assert_allclose(frame['min_extent_local'],[50.,100.])


def test_invalid_source_view_fails_before_request(tmp_path):
    r,size,view=scaled_geometry('vertical',(2.,2.));view['size']=[10,10];model=Model(['accept'])
    with pytest.raises(ValueError,match='CLEAN size'):
        gc.refine_target(Image.new('RGB',size),[r],tmp_path,model,config(),source_view=view)
    assert model.calls==[]
