"""Exact L presets and SAM context must preserve source geometry and identity."""
import copy,sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.fitting.native_presets import from_commands
from shape_integration.fitting.scene import write_pptx
from shape_integration.styling.export import render_native
from shape_integration.sam_guidance import GuidedSamEvidence


@pytest.mark.parametrize('flip_h,flip_v',[(False,False),(True,False),(False,True),(True,True)])
def test_native_corner_adjustments_and_flips_render_the_observed_polygon(tmp_path,flip_h,flip_v):
    q=np.array([[20,20],[380,20],[380,140],[460,140],[460,200],[20,200]],float)
    if flip_h:q[:,0]=480-q[:,0]
    if flip_v:q[:,1]=220-q[:,1]
    commands=[['M',*q[0]],*[['L',*p] for p in q[1:]],['Z']]
    preset=from_commands(commands)
    assert preset['preset']=='corner' and preset['flip_h']==flip_h and preset['flip_v']==flip_v
    assert max(preset['adjustments'])>1 # valid handle values use the short side, not width.
    part={'id':'frame','kind':'path','status':'ok','commands':commands,'native_parameters_original':preset}
    scene={'source_size':[500,240],'objects':{'T':{'id':'T','state':'accepted_by_model','part_keys':['p']}},'parts':{'p':part}}
    def paint(shape,part,scale):
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor(40,110,210);shape.line.fill.background()
    counts=write_pptx(scene,tmp_path/'corner.pptx',style_writer=paint)
    assert counts['native']==1 and counts['custom']==0
    actual=np.asarray(Image.open(render_native(tmp_path/'corner.pptx',tmp_path/'render',[500,240])).convert('RGB'))
    truth=Image.new('L',(500,240));ImageDraw.Draw(truth).polygon([tuple(p) for p in q],fill=255)
    target=np.asarray(truth)>0;painted=(actual[:,:,0]<100)&(actual[:,:,2]>150)
    assert np.sum(target&painted)/np.sum(target|painted)>.98


def test_open_rotated_or_self_crossing_paths_remain_custom():
    q=[[20,20],[380,20],[380,140],[460,140],[460,200],[20,200]]
    cs=[['M',*q[0]],*[['L',*p] for p in q[1:]],['Z']]
    assert from_commands(cs[:-1]) is None
    cs[2][1]+=.5
    assert from_commands(cs) is None
    q[1],q[4]=q[4],q[1]
    assert from_commands([['M',*q[0]],*[['L',*p] for p in q[1:]],['Z']]) is None


@pytest.mark.parametrize('usable',[True,False])
def test_mask_extent_grows_context_without_renaming_the_target(tmp_path,usable):
    mask=Image.new('L',(100,100));ImageDraw.Draw(mask).rectangle((10,10,90,90),outline=255,width=2);mask.save(tmp_path/'mask.png')
    prediction={'bbox':[40,10,90,90],'mask_bbox':[10,10,90,90],'mask_bbox_iou':.62,
        'score':.95,'category':'rectangle','mask_quality_ok':usable,'mask_path':str(tmp_path/'mask.png'),
        'frame':{'box':[0,0,100,100],'size':[100,100]}}
    evidence=GuidedSamEvidence([{'id':'S','representative':prediction}],min_mask_short_side=0)
    candidate={'id':'T','bbox_2d':[40,10,90,90]};crop=[35,5,95,95]
    before=copy.deepcopy(candidate);result,context=evidence.prepare(candidate,crop,(100,100))
    assert candidate==before and context['identity_bbox_original']==prediction['bbox']
    assert context['geometry_overwrite'] is False
    if usable:assert result[0]<=10
    else:assert result[0]>10


def test_sam_extent_is_hidden_from_observer_but_available_to_existing_critic(tmp_path):
    from shape_integration.sam_fixed import FixedSamEvidence
    from shape_integration.fitting.critic import source_observation_contract,validate_review
    mask=Image.new('L',(100,100));ImageDraw.Draw(mask).rectangle((10,10,90,90),outline=255,width=2)
    mask.save(tmp_path/'mask.png');Image.new('RGB',(100,100),'white').save(tmp_path/'clean.png')
    prediction={'bbox':[40,10,90,90],'mask_bbox':[10,10,90,90],'mask_bbox_iou':.62,
        'score':.95,'category':'rectangle','mask_quality_ok':True,'mask_path':str(tmp_path/'mask.png'),
        'frame':{'box':[0,0,100,100],'size':[100,100]}}
    evidence=FixedSamEvidence([{'id':'S','representative':prediction}],min_mask_short_side=0,
        observer_numeric_bbox=False,mode='critic')
    job={'object_id':'T','candidate':{'id':'T','bbox_2d':[40,10,90,90]},
         'frame':{'source_crop_half_open':[0,0,100,100],'actual_crop_input_size_px':[100,100]}}
    from shape_integration.common import save
    save(tmp_path/'fit_local.json',{'bbox_2d':[40,10,90,90]})
    observed,_=evidence.observe({},[],job,tmp_path)
    assert 'bbox_local' not in observed['sam_auxiliary_evidence']['measurement']
    prompt,images=evidence.critic({},[],job,tmp_path)
    assert prompt['extent_comparison']['fit_minus_sam_left_top_right_bottom']==[0.,0.,0.,0.]
    assert prompt['sam_auxiliary_evidence']['measurement']['bbox_local']==[40.,10.,90.,90.]
    assert prompt['sam_detector_mask_extent_conflict'] and len(images)==1
    ids=source_observation_contract(prompt,{'F1':'T'})
    assert ids==['extent_conflict']
    with pytest.raises(ValueError):validate_review({'decision':'accept','issues':[]},['F1'],ids)
