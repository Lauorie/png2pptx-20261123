from copy import deepcopy
import numpy as np
import pytest
from PIL import Image,ImageDraw
from shape_integration.fitting.source_ellipses import measure
from shape_integration.fitting.source_primitives import register,update


@pytest.mark.parametrize('axes,angle,outlined',[(20,0,True),(12,0,False),(20,35,True)])
def test_complete_source_circle_measurement_preserves_its_radius(axes,angle,outlined):
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.ellipse((50-axes,50-axes,50+axes,50+axes),fill='#AACC88',outline='#558833'if outlined else None,width=2)
    pixels=np.asarray(image);mask=(np.linalg.norm(pixels.astype(float)-[170,204,136],axis=2)<8).astype('uint8')
    result=measure(image,mask,[170,204,136])
    assert result is not None and result['source_circle_supported']
    assert np.linalg.norm(np.array(result['parameters']['center'])-[50,50])<.5
    radius=result['parameters']['width']/2
    assert abs(radius-(axes-1 if outlined else axes+.5))<1.2
    assert result['parameters']['width']==result['parameters']['height']


def test_real_ellipse_is_not_forced_to_circle_and_partial_disc_is_not_completed():
    image=Image.new('RGB',(140,100),'white');draw=ImageDraw.Draw(image)
    draw.ellipse((35,30,105,70),fill='#AACC88',outline='#558833',width=2)
    mask=(np.linalg.norm(np.asarray(image).astype(float)-[170,204,136],axis=2)<8).astype('uint8')
    result=measure(image,mask,[170,204,136]);assert result is not None
    assert max(result['parameters']['width'],result['parameters']['height'])/min(result['parameters']['width'],result['parameters']['height'])>1.5
    assert not result['source_circle_supported']
    ImageDraw.Draw(image).rectangle((0,0,140,49),fill='white')
    mask=(np.linalg.norm(np.asarray(image).astype(float)-[170,204,136],axis=2)<8).astype('uint8')
    assert measure(image,mask,[170,204,136])is None


def repeated(kind='ellipse',missing=False):
    image=Image.new('RGB',(240,280),'white');draw=ImageDraw.Draw(image);parts=[]
    for index,y in enumerate([35,90,155,235]):
        box=(140,y-12,164,y+12)
        if kind=='ellipse':draw.ellipse(box,fill='#AACC88',outline='#558833',width=2)
        else:draw.rectangle(box,fill='#AACC88',outline='#558833',width=2)
        if missing and index==2:continue
        p={'id':str(index),'object_id':'owner','key':['test','owner',str(index)],'status':'ok','kind':kind,
           'parameters':{'center':[145,y+5],'width':27.,'height':29.,'angle_radians':0.},'diagnostics':{},'observed_point_ids':[]}
        update(p);parts.append(p)
    job={'frame':{'source_crop_half_open':[0,0,240,280],'actual_crop_input_size_px':[240,280]},'identity_bbox_local':[100,5,180,270]}
    return {'parts':parts,'part_counts':{'observed':len(parts),'succeeded':len(parts)}},job,image


@pytest.mark.parametrize('kind',['ellipse','rectangle'])
def test_missing_complete_member_is_added_from_source_not_a_grid(kind):
    local,job,image=repeated(kind,True);audit=register(local,job,image)
    if kind=='rectangle':
        # The existing compact-rectangle scope deliberately ends at 24 px.
        # This fixture is a larger face and must retain its current behavior.
        assert len(local['parts'])==3 and not any(a.get('added')for a in audit)
        return
    assert len(local['parts'])==4 and sum(a.get('added',False)for a in audit)==1
    centers=sorted(p['parameters']['center'][1]for p in local['parts'])
    assert np.allclose(centers,[35,90,155,235],atol=1)
    assert all(abs(p['parameters']['center'][0]-152)<1 for p in local['parts'])


def test_blank_source_and_small_punctuation_do_not_add_or_move_nodes():
    local,job,image=repeated();before=deepcopy(local)
    assert register(local,job,Image.new('RGB',image.size,'white'))==[] and local==before
    draw=ImageDraw.Draw(image)
    for y in [190,200,210]:draw.ellipse((150,y,153,y+3),fill='black')
    audit=register(local,job,image)
    assert len(local['parts'])==4 and not any(a.get('added')for a in audit)


def test_another_existing_owner_part_is_not_added_again():
    local,job,image=repeated(missing=True)
    own=deepcopy(local['parts'][0]);own['id']='existing';own['parameters'].update(center=[152,155],width=24,height=24)
    update(own);local['parts'].append(own)
    register(local,job,image)
    assert len(local['parts'])==4


def test_registered_asset_and_source_owned_nodes_are_not_duplicated():
    local,job,image=repeated(missing=True)
    register(local,job,image,owned_regions=[[138,140,168,170]])
    assert len(local['parts'])==3


def test_source_added_member_is_idempotent_and_valid_for_later_bounded_edits():
    from shape_integration.fitting.adapter import fit_object
    from shape_integration.fitting.frames import FitFrame
    from shape_integration.fitting.contracts import validate_observation
    local,job,image=repeated(missing=True)
    points=[];parts=[]
    for p in local['parts']:
        params=p['parameters'];theta=np.linspace(0,2*np.pi,8,endpoint=False)
        xy=np.c_[np.cos(theta),np.sin(theta)]*[params['width']/2,params['height']/2]+params['center']
        ids=[p['id']+'_p'+str(i)for i in range(8)]
        points.extend({'id':pid,'xy':q.tolist()}for pid,q in zip(ids,xy));parts.append({'id':p['id'],'kind':'ellipse','point_ids':ids})
    frame=FitFrame((240,280),(240,280),(240,280),(0,0,240,280))
    local=fit_object({'object_id':'owner','status':'ok','points':points,'parts':parts,'reason':'','uncertain':[]},
                     frame,object_id='owner',namespace='test')
    register(local,job,image);validate_observation(local['observation'],'owner',(240,280))
    before=[p['id']for p in local['parts']];register(local,job,image)
    assert [p['id']for p in local['parts']]==before and len(before)==4


def test_complete_source_perimeter_keeps_width_when_sparse_profiles_disagree(tmp_path):
    from shape_integration.styling.native_stroke import NativeFitter
    from shape_integration.fitting.paths import ellipse_commands
    from unittest.mock import patch
    image=Image.new('RGB',(120,120),'white');ImageDraw.Draw(image).ellipse((40,40,80,80),fill='#AACC88',outline='#558833',width=2)
    source=tmp_path/'source.png';image.save(source)
    part={'id':'circle','object_id':'owner','kind':'ellipse','status':'ok','commands':ellipse_commands([60,60],[19,0],[0,19]),
          'bbox_2d':[41,41,79,79],'diagnostics_local':{'source_primitive_registration':{
          'perimeter_support':1.,'source_width_px':2.,'source_stroke':[85,136,51]}}}
    fitter=NativeFitter(source,{'circle':part},None,'case',tmp_path)
    with patch.object(fitter,'features',return_value=None):
        style,measurement=fitter.fit('circle',{'fill':'#AACC88','stroke':'#558833','width_px':9.,'dash':'solid'})
    assert style['width_px']==2 and measurement['source_perimeter_width_applied']


@pytest.mark.parametrize('kind',['ellipse','rectangle'])
def test_excess_predictions_do_not_hide_complete_source_family(kind):
    image=Image.new('RGB',(180,250),'white');draw=ImageDraw.Draw(image);parts=[]
    for y in (35,90,145):
        box=(100,y-9,118,y+9)
        if kind=='ellipse':draw.ellipse(box,fill='#AACC88',outline='#558833',width=2)
        else:draw.rectangle(box,fill='#AACC88',outline='#558833',width=2)
    for i,y in enumerate((38,94,147,195)):
        part={'id':'candidate'+str(i),'object_id':'owner','key':['test','owner',str(i)],
              'status':'ok','kind':kind,'parameters':{'center':[105,y],'width':20.,'height':21.,'angle_radians':0.},
              'diagnostics':{},'observed_point_ids':[]}
        update(part);parts.append(part)
    local={'parts':parts,'part_counts':{'observed':4,'succeeded':4}}
    job={'frame':{'source_crop_half_open':[0,0,180,250],'actual_crop_input_size_px':[180,250]},
         'identity_bbox_local':[80,10,135,225]}
    audit=register(local,job,image)
    assert len(local['parts'])==4
    assert any(row.get('unmatched_parts')==['candidate3'] for row in audit)
    assert all(abs(part['parameters']['center'][0]-109)<1 for part in local['parts'][:3])
    assert local['parts'][3]['parameters']['center']==[105,195]


def test_sparse_end_member_uses_its_own_source_identity_gap():
    image=Image.new('RGB',(180,250),'white');draw=ImageDraw.Draw(image);parts=[]
    for y in (35,65,95,190):
        draw.rectangle((100,y-9,118,y+9),fill='#AACC88',outline='#558833',width=2)
    for i,y in enumerate((38,68,98,225)):
        p={'id':str(i),'object_id':'owner','key':['test','owner',str(i)],
           'status':'ok','kind':'rectangle','parameters':{'center':[105,y],
           'width':20.,'height':21.,'angle_radians':0.},'diagnostics':{},'observed_point_ids':[]}
        update(p);parts.append(p)
    local={'parts':parts};job={'frame':{'source_crop_half_open':[0,0,180,250],
        'actual_crop_input_size_px':[180,250]},'identity_bbox_local':[80,10,135,245]}
    audits=register(local,job,image)
    assert len(parts)==4 and not any(a.get('added')for a in audits)
    assert np.allclose([p['parameters']['center'][1]for p in parts],[35,65,95,190],atol=1)
