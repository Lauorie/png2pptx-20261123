import copy
import numpy as np
import pytest
import evidence as ev
import shared_repair as sr
import semantic_geometry as sem
import geometry_routing as routing


def row(rid='old'):
    return dict(id=rid,kind='polyline',start=[0.,0.],via=[[0.,100.],[100.,100.]],end=[100.,0.],heads=['none','none'])


def test_bbox_is_not_actual_coverage():
    r=row()
    assert not sr.visible(r,[40,40,60,60])
    assert sr.visible(r,[40,90,60,110])
    assert sr.visible(dict(r,kind='straight',start=[0,50],via=[],end=[100,50]),[40,40,60,60])


def test_patch_preserves_every_unmentioned_object_and_can_delete():
    old=[row('a'),row('b')];snapshot=copy.deepcopy(old)
    a=dict(status='repair',replace=['E1'],nodes=[],routes=[])
    checked=sr.validate(a,[100,100],[100,100],['E1'],[],True)
    after=sr.apply_patch(old,checked,[row('new')],{'E1':'a'})
    assert after==[old[1],row('new')] and old==snapshot
    assert sr.apply_patch(old,dict(replace=[]),[],{})==snapshot
    with pytest.raises(ValueError):sr.apply_patch(old,dict(replace=[]),[row('a')],{})


def test_unknown_edits_fail_and_expansion_does_not_apply_partial_edits():
    with pytest.raises(ValueError):sr.validate(dict(status='repair',replace=['E9'],routes=[],nodes=[]),[100,100],[100,100],['E1'],[],True)
    with pytest.raises(ValueError):sr.validate(dict(status='expand',replace=['E1'],context_bbox=[0,0,100,100]),[100,100],[100,100],['E1'],[],True)
    with pytest.raises(ValueError):sr.validate(dict(status='expand',replace=[],context_bbox=[0,0,100,100]),[100,100],[100,100],[],[],False)


def test_measurements_keep_mask_ids_and_local_coordinates():
    m=np.zeros((100,100),bool);m[10:12,20:50]=True
    got=routing.measurements(['M7'],{'M7':dict(mask=m)},dict(D9=dict(masks=['M7'],kind='straight',start=[20.123456,10.98],end=[49.,11.])))
    assert got==[dict(id='M7',bbox=[20,10,50,12],straight_estimates=[dict(id='D9',kind='straight',points=[[20.1,11.],[49.,11.]])])]


def test_sparse_followups_use_explicit_context_frame():
    raw=dict(nodes=[],routes=[],excluded=['G1'],uncertain=[],followups=[dict(context_bbox=[10,20,50,60],description='related return')])
    got=sem.validate(raw,[1080,500],['G1'],{},[],context_size=[768,300])
    assert got['followups']==raw['followups']
    with pytest.raises(ValueError):sem.validate(raw,[1080,500],['G1'],{},[],context_size=[30,30])


def test_current_point_table_rounds_and_roundtrips_coordinates():
    view=ev.identity_view([100,100]);view['global_to_view']=[[2,0,10],[0,2,20],[0,0,1]]
    r=row();r['short_id']='E1'
    table=sr.inventory([r],view)
    assert table[0]['points'][0]==[10.,20.] and table[0]['points'][-1]==[210.,20.]


def test_merge_results_does_not_drop_same_parent_siblings():
    groups=[dict(group='G1',rows=[dict(row('a'),parent_id='G1.T1'),dict(row('b'),parent_id='G1.T1')])]
    patch=dict(group='P1',replace=['a'],rows=[row('new')])
    result=sr.merge_results(groups,patch)
    assert [r['id'] for g in result for r in g['rows']]==['b','new']
    assert len(groups[0]['rows'])==2


def test_context_expansion_continues_same_geometry_task(tmp_path,monkeypatch):
    import threading
    from PIL import Image
    from transport.common import read_json
    calls=[];sam_views=[]
    def sample(crop,view,rows,hints,group_of,gid,folder,config,slot):
        sam_views.append(view)
        return [],[]
    monkeypatch.setattr(sr.local_sampling,'sample',sample)
    class Model:
        def ask(self,path,prompt,images,validator):
            if prompt.get('task')==sr.head_semantics.TASK:
                return validator(dict(heads={'R1':['none','none'],'R2':['none','triangle']}))
            calls.append(prompt)
            if len(calls)==1:
                return validator(dict(status='expand',replace=[],nodes=[],routes=[],context_bbox=[0,0,80,80],reason='need connection end'))
            return validator(dict(status='repair',replace=[],nodes=[['a',10,20],['j',40,20],['b',40,60]],
                routes=[dict(id='one',heads=['none','none'],line_style='solid',masks=[],segments=[dict(kind='line',nodes=['a','j'])]),
                        dict(id='two',heads=['none','triangle'],line_style='solid',masks=[],segments=[dict(kind='line',nodes=['j','b'])])]))
    # Style is independently tested: this test verifies geometry/ownership flow.
    monkeypatch.setattr(sr.style_windows,'recover',lambda *a:dict(width_px=1,color='#000000',dash=dict(kind='solid'),needs_review=False))
    monkeypatch.setattr(sr.style,'pending',lambda row:False)
    monkeypatch.setattr(sr.viz,'render',lambda *a:None)
    cfg=dict(crop_context_scale=2,context_padding=0,min_crop_side=20,context_side=100,display_side=100,repair_display_side=100,repair_context_scale=1)
    locks=dict(cpu=threading.Semaphore(1),sam=threading.Semaphore(1))
    existing=[dict(row(),id='old',start=[0,90],via=[],end=[100,90])]
    after,meta=sr.recover(Image.new('RGB',(100,100),'white'),dict(bbox=[10,10,50,50],description='related connection'),existing,'P1',tmp_path,cfg,Model(),locks)
    assert len(calls)==2 and calls[0]['task']==calls[1]['task']
    assert meta['head_calls']==1
    assert 'description' not in calls[0]['focus'] and 'description' not in calls[1]['focus']
    assert calls[0]['focus']['bbox']==[0.,0.,100.,100.]
    assert calls[1]['focus']['bbox']==[12.5,12.5,62.5,62.5]
    assert len(sam_views)==2 and sam_views[0]['bbox_global']!=sam_views[1]['bbox_global']
    assert after[0]==existing[0] and len(after)==3
    local=read_json(tmp_path/'rows_local.json')
    assert local[0]['end']==local[1]['start']==[40,20]
    assert local[0]['parent_id']==local[1]['parent_id']
    assert len(after[1]['model_nodes'])==3


def test_block_patch_keeps_contour_capability_and_line_patches_unchanged():
    a=dict(status='repair',replace=[],nodes=[],routes=[dict(id='wide',kind='block_arrow',masks=['M1'],heads=['none','triangle'],line_style='solid',segments=[])])
    checked=sr.validate(a,[100,100],[100,100],[],['M1'],True)
    assert checked==a
    invalid=copy.deepcopy(a);invalid['routes'][0]['masks']=[]
    with pytest.raises(ValueError):sr.validate(invalid,[100,100],[100,100],[],['M1'],True)


def test_offcrop_existing_points_are_reused_exactly_never_reestimated():
    answer=dict(status='repair',replace=['E1'],nodes=[['a',10,20],['b',140,20]],routes=[dict(id='r',heads=['none','triangle'],line_style='solid',masks=[],segments=[dict(kind='line',nodes=['a','b'])])])
    points={'E1':[[10,20],[140,20]],'E2':[[150,30],[180,30]]}
    assert sr.validate(answer,[100,100],[100,100],['E1','E2'],[],True,points)==answer
    # New off-crop estimates, or coordinates belonging only to an unrelated
    # unmodified route, must ask for context rather than being silently clamped.
    for xy in [[140.1,20],[150,30]]:
        changed=copy.deepcopy(answer);changed['nodes'][1][1:]=xy
        with pytest.raises(ValueError):sr.validate(changed,[100,100],[100,100],['E1','E2'],[],True,points)
    with pytest.raises(ValueError):sr.validate(answer,[100,100],[100,100],['E1'],[],True)


def test_empty_sam_reuses_attributed_mask_without_creating_or_editing_routes(tmp_path,monkeypatch):
    import threading
    from PIL import Image
    from transport.common import write_json,read_json
    old=tmp_path/'old';old.mkdir()
    write_json(old/'transform.json',ev.identity_view([100,100]))
    mask=np.zeros((100,100),dtype='uint8');mask[49:52,10:91]=255
    Image.fromarray(mask).save(old/'M1.png')
    existing=[dict(row('old'),kind='straight',start=[10,50],via=[],end=[90,50],
        source_mask_paths=[str(old/'M1.png')],local_input=str(old/'clean.png'))]
    monkeypatch.setattr(sr.local_sampling,'sample',lambda *a:([],[]))
    class Model:
        def ask(self,path,prompt,images,validator):
            assert prompt['mask_measurements'][0]['id']=='M1'
            assert prompt['mask_measurements'][0]['associated_existing']==['E1']
            assert prompt['mask_measurements'][0]['origin']=='earlier local SAM'
            return validator(dict(status='complete',replace=[],nodes=[],routes=[]))
    cfg=dict(crop_context_scale=2,context_padding=0,min_crop_side=20,context_side=100,
        display_side=100,repair_display_side=100,repair_context_scale=1,repair_reuse_cached_masks=True)
    locks=dict(cpu=threading.Semaphore(1),sam=threading.Semaphore(1))
    after,meta=sr.recover(Image.new('RGB',(100,100),'white'),
        dict(bbox=[0,0,100,100],description='check the existing line'),existing,'P1',tmp_path/'repair',cfg,Model(),locks)
    assert after==existing and meta['rows']==[] and meta['replace']==[]
    evidence=read_json(tmp_path/'repair/local/mask_evidence.json')
    assert evidence['fresh_selected']==0 and evidence['retained_selected']==1


def test_repair_geometry_defers_head_types_without_guessing_them():
    answer=dict(status='repair',replace=[],nodes=[['a',10,20],['b',50,20]],routes=[dict(
        id='r',kind='line',line_style='solid',masks=[],segments=[dict(kind='line',nodes=['a','b'])])])
    checked=sr.validate(answer,[100,100],[100,100],[],[],True)
    assert checked['routes'][0]['heads']==['unknown','unknown']
    assert 'heads' not in answer['routes'][0]


def test_multiple_paths_survive_repair_consumption(tmp_path,monkeypatch):
    import threading
    from PIL import Image
    monkeypatch.setattr(sr.local_sampling,'sample',lambda *a:([],[]))
    class Model:
        def ask(self,path,prompt,images,validator):
            return validator(dict(status='repair',replace=[],nodes=[['a',10,40],['j',50,40],['b',90,40],['c',50,10]],
                routes=[dict(id='one',kind='line',line_style='solid',masks=[],paths=[
                    dict(id='trunk',segments=[dict(kind='line',nodes=['a','j']),dict(kind='line',nodes=['j','b'])]),
                    dict(id='arm',segments=[dict(kind='line',nodes=['c','j'])])])]))
    def heads(crop,rows,supports,*args,**kwargs):
        assert len(rows)==2 and set(supports)=={r['id'] for r in rows}
        for r in rows:r['heads']=['none','none']
        return rows,[],0
    monkeypatch.setattr(sr.head_semantics,'recover',heads)
    monkeypatch.setattr(sr.style_windows,'recover',lambda *a:dict(width_px=1,color='#000000',dash=dict(kind='solid'),needs_review=False))
    monkeypatch.setattr(sr.style,'pending',lambda row:False)
    monkeypatch.setattr(sr.viz,'render',lambda *a:None)
    cfg=dict(crop_context_scale=1,context_padding=0,min_crop_side=20,context_side=100,display_side=100,repair_display_side=100,repair_context_scale=1)
    locks=dict(cpu=threading.Semaphore(1),sam=threading.Semaphore(1))
    after,meta=sr.recover(Image.new('RGB',(100,100),'white'),dict(bbox=[0,0,100,100],description='visible target'),[],
                          'P1',tmp_path,cfg,Model(),locks)
    assert len(after)==2 and not meta['issues']
    assert {r['id'] for r in after}=={'P1:one/trunk','P1:one/arm'}
    assert len({r['parent_id'] for r in after})==1
