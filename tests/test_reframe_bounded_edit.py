import copy
import numpy as np
from PIL import Image
from shape_integration.common import save
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.adapter import fit_object,transform_fit
from shape_integration.fitting.revision import plan_revision,reframe_context,merge_revision,preserve_fits
from shape_integration.fitting.scene import native_parameters
from shape_integration.fitting.local import prepare_candidate


def test_expanded_crop_preserves_exact_supported_contours_and_only_adds_missing_part(tmp_path):
    frame=FitFrame((200,100),(400,200),(400,200),(100,50,300,150))
    raw={'object_id':'owner','status':'ok','reason':'visible','uncertain':[],
         'points':[{'id':str(i),'xy':p} for i,p in enumerate([[20,20],[80,20],[80,70],[20,70]])],
         'parts':[{'id':'body','kind':'rectangle','point_ids':['0','1','2','3']}]}
    local=fit_object(raw,frame,object_id='owner',namespace='old');original=transform_fit(local)
    save(tmp_path/'fit_local.json',local)
    context=plan_revision(tmp_path,{'issues':[{'kind':'missing','parts':[],
        'evidence':'Another small circle is visible to the right.','action':'Add only the missing circle.'}]})
    enlarged=FitFrame((600,300),(400,200),(400,200),(80,30,380,180))
    before=copy.deepcopy(context);context=reframe_context(context,enlarged.metadata())
    assert before['base_observation']==raw
    for point,old in zip(context['base_observation']['points'],raw['points']):
        np.testing.assert_allclose(enlarged.to_original(point['xy']),frame.to_original(old['xy']))
    patch={'object_id':'owner','status':'ok','reason':'new','uncertain':[],
        'points':[{'id':'n'+str(i),'xy':p} for i,p in enumerate([[440,100],[460,100],[460,120],[440,120]])],
        'parts':[{'id':'new','kind':'rectangle','point_ids':['n0','n1','n2','n3']}]}
    combined=merge_revision(patch,context,'owner',enlarged.input_size,audit={})
    fitted=fit_object(combined,enlarged,object_id='owner',namespace='new');preserve_fits(fitted,context)
    transported=transform_fit(fitted);parts={p['id']:p for p in transported['parts']}
    assert parts['body']['commands']==original['parts'][0]['commands']
    assert native_parameters(parts['body'],transported['frame'])==native_parameters(original['parts'][0],original['frame'])
    assert set(parts)=={'body','new'}


def test_observer_receives_bounded_edit_after_actual_image_recrop(tmp_path):
    old=FitFrame((100,100),(300,200),(300,200),(50,50,150,150))
    raw={'object_id':'owner','status':'ok','reason':'','uncertain':[],
         'points':[{'id':str(i),'xy':p} for i,p in enumerate([[10,10],[60,10],[60,60],[10,60]])],
         'parts':[{'id':'body','kind':'rectangle','point_ids':['0','1','2','3']}]}
    local=fit_object(raw,old,object_id='owner',namespace='old');save(tmp_path/'fit_local.json',local)
    context=plan_revision(tmp_path,{'issues':[{'kind':'missing','parts':[],
        'evidence':'Missing detail.','action':'Add the missing detail.'}]})
    job=prepare_candidate(Image.new('RGB',(300,200),'white'),
        {'id':'owner','label':'same object','kind':'compound','bbox_2d':[60,60,110,110]},
        [300,200],tmp_path/'larger',crop_bounds_original=[30,30,250,170],revision_context=context)
    from shape_integration.fitting.observation import make_prompt
    prompt=make_prompt(job)
    assert prompt['revision_task']['preserved_parts']==[{'id':'body','kind':'rectangle'}]
    assert prompt['revision_task']['allow_new_parts_for_missing_geometry']


def test_controller_reuses_accepted_parts_when_scope_expands_instead_of_whole_redraw(tmp_path):
    from shape_integration.fitting.control import process_object
    from shape_integration.fitting.seed import revision_seed
    from shape_integration.fitting.local import observe_and_fit
    from shape_integration.common import read,sha
    image=tmp_path/'source.png';Image.new('RGB',(300,200),'white').save(image)
    candidate={'id':'owner','label':'same object','kind':'compound','bbox_2d':[60,60,110,110]}
    old=tmp_path/'old';old.mkdir()
    job=prepare_candidate(Image.open(image),candidate,[300,200],old/'revision_00',crop_bounds_original=[40,40,160,160])
    f=FitFrame.from_metadata(job['frame']);coords=[f.to_local(p) for p in [[65,65],[100,65],[100,100],[65,100]]]
    raw={'object_id':'owner','status':'ok','reason':'','uncertain':[],
         'points':[{'id':str(i),'xy':p} for i,p in enumerate(coords)],
         'parts':[{'id':'body','kind':'rectangle','point_ids':['0','1','2','3']}]}
    local=fit_object(raw,f,object_id='owner',namespace='old');save(old/'revision_00/fit_local.json',local)
    save(old/'revision_00/fit_original.json',transform_fit(local));save(old/'revision_00/observation.json',raw)
    save(old/'revision_00/status.json',{'status':'finished','part_counts':{'succeeded':1}})
    save(old/'record.json',{'state':'accepted_by_model','selected_revision':'revision_00'})
    seed=revision_seed(old);before=sha(old/'revision_00/fit_original.json')
    options={'crop_padding':.1,'crop_scale':3.,'crop_min_long_edge':1080,'crop_max_long_edge':2048,
        'max_revisions':1,'max_context_rounds':1,'check_enabled':False}
    item={'original':str(image),'input':str(image),'source_size':[300,200],'input_size':[300,200]}
    context={'coverage_source_extent_original':[60,60,230,110]}
    request={'decision':'revise','reason':'A missing part outside the old crop.',
             'issues':[{'kind':'missing','parts':[],'evidence':'A second rectangle is visible.',
                        'action':'Add only the missing rectangle.'}]}
    destination=tmp_path/'new'
    process_object(destination,candidate,item,tmp_path,None,options,namespace='control',
        prepare_only=True,seed=seed,review_seed=request,input_context=context)
    job=read(destination/'revision_00/job.json')
    assert job['revision_context']['preserved_ids']==['body']
    assert job['revision_context']['geometry_reframed_without_refitting']
    assert job['frame']['source_crop_half_open'][2]>=231
    assert sha(old/'revision_00/fit_original.json')==before


def test_failed_expanded_edit_preserves_original_fallback_without_claiming_acceptance(tmp_path):
    # A wider view can fail before returning a patch. Existing usable source
    # geometry still belongs in the unresolved result, rather than disappearing.
    from shape_integration.fitting.control import process_object
    from shape_integration.fitting.seed import revision_seed
    from shape_integration.common import read
    from shape_integration.common import Runtime
    source=tmp_path/'source.png';Image.new('RGB',(300,200),'white').save(source)
    candidate={'id':'owner','label':'same object','kind':'compound','bbox_2d':[60,60,110,110]}
    old=tmp_path/'old';old.mkdir();job=prepare_candidate(Image.open(source),candidate,[300,200],old/'revision_00',crop_bounds_original=[40,40,160,160])
    frame=FitFrame.from_metadata(job['frame']);raw={'object_id':'owner','status':'ok','reason':'','uncertain':[],
        'points':[{'id':str(i),'xy':frame.to_local(p)} for i,p in enumerate([[65,65],[100,65],[100,100],[65,100]])],
        'parts':[{'id':'body','kind':'rectangle','point_ids':['0','1','2','3']}]}
    fit=fit_object(raw,frame,object_id='owner',namespace='old')
    save(old/'revision_00/fit_local.json',fit);save(old/'revision_00/fit_original.json',transform_fit(fit));save(old/'revision_00/observation.json',raw)
    save(old/'revision_00/status.json',{'status':'finished','part_counts':{'succeeded':1}});save(old/'record.json',{'state':'accepted_by_model','selected_revision':'revision_00'})
    class Failing(Runtime):
        def call(self,*args,**kwargs):raise RuntimeError('invalid response')
    options={'crop_padding':.1,'crop_scale':3.,'crop_min_long_edge':1080,'crop_max_long_edge':2048,
        'max_revisions':1,'max_context_rounds':1,'check_enabled':False,'max_tokens':4096}
    result=process_object(tmp_path/'new',candidate,{'original':str(source),'input':str(source),'source_size':[300,200],'input_size':[300,200]},tmp_path,
        Failing({'api_concurrency':1,'fitting':{}},'offline',''),options,namespace='control',seed=revision_seed(old),
        review_seed={'decision':'revise','reason':'Missing detail.','issues':[{'kind':'missing','parts':[],'evidence':'Missing detail.','action':'Add detail.'}]},
        input_context={'coverage_source_extent_original':[60,60,230,110]})
    assert result['state']=='unresolved'
    assert result['selected_revision'] is not None


def test_prepared_expanded_part_edit_resumes_with_the_same_source_context(tmp_path):
    from shape_integration.fitting.control import prepared
    old=FitFrame((100,100),(300,200),(300,200),(50,50,150,150))
    raw={'object_id':'owner','status':'ok','reason':'','uncertain':[],
         'points':[{'id':str(i),'xy':p} for i,p in enumerate([[10,10],[60,10],[60,60],[10,60]])],
         'parts':[{'id':'body','kind':'rectangle','point_ids':['0','1','2','3']}]}
    local=fit_object(raw,old,object_id='owner',namespace='old');save(tmp_path/'fit_local.json',local)
    context=plan_revision(tmp_path,{'issues':[{'kind':'missing','parts':[],
        'evidence':'Missing detail.','action':'Add the missing detail.'}]})
    image=tmp_path/'source.png';Image.new('RGB',(300,200),'white').save(image)
    candidate={'id':'owner','label':'same object','kind':'compound','bbox_2d':[60,60,110,110]}
    item={'source_size':[300,200],'input_size':[300,200]}
    options={'crop_padding':.1,'crop_scale':3.,'crop_min_long_edge':1080,'crop_max_long_edge':2048}
    first=prepared(image,candidate,item,tmp_path/'new',options,[30,30,250,170],revision_context=context)
    resumed=prepared(image,candidate,item,tmp_path/'new',options,[30,30,250,170],revision_context=context)
    assert resumed==first
