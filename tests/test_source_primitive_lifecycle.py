from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image,ImageDraw

from shape_integration.common import save,sha
from shape_integration.fitting.adapter import fit_object,transform_fit
from shape_integration.fitting.contracts import normalize_observation,validate_observation
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.ownership import retained_receipt
from shape_integration.fitting.source_primitives import register
from shape_integration.fitting.scene import assemble,write_pptx
from shape_integration.fitting.critic import review_prompt,part_inventory
from shape_integration.fitting.revision import plan_revision,merge_revision,preserve_fits


def fixture(role='body'):
    source=Image.new('RGB',(240,250),'white');draw=ImageDraw.Draw(source)
    points=[];parts=[];roles=[]
    for index,y in enumerate((35,85,135,190)):
        draw.rectangle((170,y-9,188,y+9),fill='#FFCCED',outline='#FF78C7',width=1)
        if index==2:continue
        ids=[]
        for j,xy in enumerate(((170,y-9),(188,y-9),(188,y+9),(170,y+9))):
            ident=f'p{index}_{j}';ids.append(ident);points.append({'id':ident,'xy':list(xy)})
        ident=f'cell{index}';parts.append({'id':ident,'kind':'rectangle','point_ids':ids})
        roles.append({'id':ident,'role':role})
    raw={'status':'ok','points':points,'parts':parts,'ownership':{
        'unit':'other','unit_evidence':'Independent visible closed cells','parts':roles}}
    # An uncertain family remains visible for diagnostics, with its actual role.
    audit={};canonical=normalize_observation(raw,'owner',[240,250],audit=audit,ownership_required=True)
    frame=FitFrame((240,250),(240,250),(240,250),(0,0,240,250))
    local=fit_object(canonical,frame,object_id='owner',namespace='lifecycle')
    local['declared_ownership']=audit['declared_ownership']
    job={'object_id':'owner','label':'closed cell group','proposed_type':'compound_symbol',
         'frame':{**frame.metadata(),'source_crop_half_open':[0,0,240,250]},
         'identity_bbox_local':[80,10,155,225],'input_context':{}}
    return local,job,source


def added(local,job,source):
    existing={p['id']for p in local['parts']};audits=register(local,job,source)
    new=[p['id']for p in local['parts']if p['id']not in existing]
    return new,audits


def test_source_added_member_has_complete_geometry_roles_and_critic_identity():
    local,job,source=fixture();ids,audit=added(local,job,source)
    assert len(ids)==1
    ident=ids[0]
    assert local['declared_ownership']['retained_roles'][ident]=='body'
    validate_observation(local['observation'],'owner',[240,250])
    raw={**deepcopy(local['observation']),'ownership':{'unit':'other','unit_evidence':'Source closed members',
        'parts':[{'id':i,'role':role}for i,role in local['declared_ownership']['retained_roles'].items()]}}
    assert len(normalize_observation(raw,'owner',[240,250],ownership_required=True)['parts'])==4
    prompt=review_prompt(job,local)
    aliases={ident:alias for alias,ident in part_inventory(local)[0].items()}
    # Display labels point to the same newly added stable ID and body role.
    assert prompt['declared_branch_roles']['roles'][aliases[ident]]=='body'


def test_repeat_registration_and_pending_family_overlap_are_idempotent(monkeypatch):
    import shape_integration.fitting.source_primitives as module
    local,job,source=fixture();original=module.groups
    def duplicate_family(value,scale):
        groups=list(original(value,scale))
        yield from groups
        yield from groups
    monkeypatch.setattr(module,'groups',duplicate_family)
    ids,_=added(local,job,source)
    assert len(ids)==1 and len(local['parts'])==4
    for _ in range(3):register(local,job,source)
    assert len(local['parts'])==4
    assert len({p['id']for p in local['parts']})==len(local['parts'])
    assert len({p['id']for p in local['observation']['parts']})==4
    assert len({p['id']for p in local['observation']['points']})==16
    assert len({p['id']for p in local['points']})==16


def test_unresolved_family_role_cannot_silently_become_owned_added_geometry():
    local,job,source=fixture('uncertain');ids,audit=added(local,job,source)
    assert ids==[] and len(local['parts'])==3
    assert any(r['reason']=='source_member_family_role_unresolved'for r in audit)


def test_old_source_measured_member_restores_role_without_new_geometry():
    local,job,source=fixture();ids,_=added(local,job,source);ident=ids[0]
    local['declared_ownership']['retained_roles'].pop(ident)
    local['declared_ownership']['uncertain_parts']=[ident]
    before=deepcopy(local['parts'])
    audit=register(local,job,source)
    assert local['parts']==before
    assert local['declared_ownership']['retained_roles'][ident]=='body'
    assert local['declared_ownership']['uncertain_parts']==[]
    assert any(r.get('metadata_only')and r['part_id']==ident for r in audit)


def test_assembled_added_node_remains_shape_and_does_not_delete_attached_arrow(tmp_path):
    from full_recovery.quality_choice import candidates,nominate
    from full_recovery.handoff import reconcile
    from full_recovery.compose import merge,read_package,SLIDE,P
    from lxml import etree
    local,job,source=fixture();ids,_=added(local,job,source);ident=ids[0]
    save(tmp_path/'objects/owner/revision_00/fit_original.json',transform_fit(local))
    original=tmp_path/'source.png';source.save(original)
    item={'id':'sample','original':str(original),'source_size':list(source.size),'source_sha256':sha(original)}
    scene=assemble(tmp_path,item,[{'id':'owner','label':'closed cells','kind':'compound_symbol',
        'bbox_2d':[160,10,200,225]}],{'owner':{'state':'accepted_by_model','selected_revision':'revision_00','history':[]}}, {},[])
    assert len(scene['parts'])==4
    assert all(key in scene['points']for p in scene['parts'].values()for key in p['point_keys'])
    key=next(k for k,p in scene['parts'].items()if p['id']==ident)
    styles={k:{'fill':'#FFCCED','stroke':'#FF78C7','width_px':1,'dash':'solid'}for k in scene['parts']}
    arrow={'id':'directed','start':[125.,135.],'end':[170.,135.],
        'heads':['none','triangle'],'style':{'color':'#555555','width_px':1}}
    rows=candidates(scene,styles,[arrow]);pairs=nominate(rows,np.asarray(source))
    assert all('shape:'+key not in(p['a'],p['b'])for p in pairs)
    quality={'ordinary_line_owner':'local_competitor_groups','line_ownership_complete':True,
        'drop_shape_keys':[],'drop_shape_names':[],'drop_arrow_names':[],'reviewed_arrow_names':['directed']}
    plan=reconcile(scene,{'parts':{k:{'style':v}for k,v in styles.items()}},
        {'arrows':[arrow]},{'texts':[]},original,quality=quality)
    assert key not in plan['drop_shape_keys'] and plan['drop_arrow_names']==[]
    write_pptx(scene,tmp_path/'nodes.pptx')
    slide=etree.fromstring(read_package(tmp_path/'nodes.pptx')[SLIDE])
    names=[n.get('name')for n in slide.iter('{'+P+'}cNvPr')]
    assert sum(n.startswith('owner/'+ident+' ')for n in names if n)==1


def test_later_bounded_edit_and_role_inheritance_preserve_added_cell(tmp_path):
    local,job,source=fixture();ids,_=added(local,job,source);ident=ids[0]
    save(tmp_path/'fit_local.json',local)
    context=plan_revision(tmp_path,{'issues':[{'kind':'geometry','parts':[],
        'part_ids':['cell0'],'evidence':'Correct unrelated first cell','action':'Move that cell only'}]})
    assert ident in context['preserved_ids']
    part=deepcopy(local['observation']['parts'][0]);refs=set(part['point_ids'])
    patch={**deepcopy(local['observation']),'parts':[part],
        'points':[p for p in local['observation']['points']if p['id']in refs]}
    audit={};merged=merge_revision(patch,context,'owner',[240,250],audit=audit)
    fit=fit_object(merged,FitFrame.from_metadata(local['frame']),object_id='owner',namespace='edit')
    preserve_fits(fit,context)
    role=retained_receipt({'unit':'other','retained_roles':{'cell0':'body'}},
        local['declared_ownership'],[p['id']for p in fit['parts']])
    assert role['retained_roles'][ident]=='body'
    assert next(p for p in fit['parts']if p['id']==ident)['commands']==next(
        p for p in local['parts']if p['id']==ident)['commands']


def test_metadata_only_role_repair_is_persisted_without_more_model_calls(tmp_path,monkeypatch):
    from shape_integration.fitting.control import process_object
    from shape_integration.fitting.seed import revision_seed
    from test_pipeline import FakeRuntime,reviewed
    local,job,source=fixture();ids,_=added(local,job,source);ident=ids[0]
    local['declared_ownership']['retained_roles'].pop(ident)
    job['candidate']={'id':'owner','label':'closed cells','kind':'compound_symbol',
                      'bbox_2d':[160,10,200,225]}
    job['identity_descriptions']=[]
    job['input_sha256']={}
    original=tmp_path/'source.png';source.save(original)
    item={'id':'sample','original':str(original),'input':str(original),'source_size':list(source.size),
          'input_size':list(source.size),'source_sha256':sha(original),'input_sha256':sha(original)}
    old=tmp_path/'old';folder=old/'revision_00';folder.mkdir(parents=True)
    source.save(folder/'clean.png');source.save(folder/'overlay.png')
    save(folder/'prompt.json',{})
    save(folder/'frame.json',job['frame'])
    job['input_sha256']={name:sha(folder/name)for name in ('prompt.json','clean.png','overlay.png','frame.json')}
    save(folder/'job.json',job);save(folder/'fit_local.json',local)
    save(folder/'fit_original.json',transform_fit(local));save(folder/'observation.json',local['observation'])
    save(folder/'status.json',{'status':'finished','observation_status':'ok','fit_status':'ok',
                             'part_counts':local['part_counts']})
    save(old/'record.json',{'state':'accepted_by_model','selected_revision':'revision_00','history':[]})
    import shape_integration
    config=json.loads((Path(shape_integration.__file__).resolve().parents[2]/'config.json').read_text())
    def respond(tag,prompt,count):
        assert tag=='check'
        result=reviewed(prompt)
        result['source_observations']=[{'id':r['id'],'source_relation':'supported',
            'evidence':'Independent filled source members with complete contours'}
            for r in prompt.get('required_source_observations',{}).get('targets',[])]
        return result
    runtime=FakeRuntime(config,respond)
    # Isolate the metadata persistence boundary from unrelated remeasurement.
    import shape_integration.fitting.source_boxes as boxes
    def metadata_only(value,job,clean,**kwargs):
        value['declared_ownership']['retained_roles'][ident]='body'
        return [{'applied':False,'metadata_only':True,'part_id':ident,
                 'reason':'source_measured_family_role_restored','inherited_role':'body'}]
    monkeypatch.setattr(boxes,'register',metadata_only)
    import shape_integration.fitting.frame_edges as edges
    import shape_integration.fitting.rounded_corners as corners
    monkeypatch.setattr(edges,'refine',lambda *a,**k:[])
    monkeypatch.setattr(corners,'refine',lambda *a,**k:[])
    fresh=tmp_path/'new'
    record=process_object(fresh,job['candidate'],item,tmp_path,runtime,config['fitting'],
        namespace='metadata_test',seed=revision_seed(old))
    assert record['selected_revision']is not None,record
    saved=json.loads((fresh/record['selected_revision']/'fit_local.json').read_text())
    assert saved['declared_ownership']['retained_roles'][ident]=='body'
    assert saved['parts']==local['parts']
    assert [tag for tag,_ in runtime.events]==['check']
    assert record['source_registration']['metadata_only_parts']==[ident]


@pytest.mark.parametrize('other_role',['uncertain','contour'])
def test_mixed_family_roles_cannot_authorize_new_members(other_role):
    local,job,source=fixture()
    local['declared_ownership']['retained_roles']['cell0']=other_role
    ids,audit=added(local,job,source)
    assert ids==[]
    assert any(r['reason']=='source_member_family_role_unresolved'for r in audit)
