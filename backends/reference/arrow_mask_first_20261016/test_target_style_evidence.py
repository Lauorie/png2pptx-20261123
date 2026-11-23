import copy
import numpy as np
import pytest
from PIL import Image, ImageDraw
import target_locator,shaft_sampling,style_windows
from lib import fit,style


def test_global_extent_survives_local_head_only_evidence():
    view=dict(global_to_view=[[5,0,-50],[0,5,-100],[0,0,1]])
    target=target_locator.boxes([dict(id='G1',box=[12,25,40,30])],view)[0]
    assert target['bbox']==[10.,25.,150.,50.]
    with pytest.raises(ValueError):target_locator.boxes([dict(id='G1')],view)


def test_model_target_extent_and_style_sample_survive_full_fast_path(tmp_path):
    import threading,semantic_geometry,geometry_routing
    image=Image.new('RGB',(260,100),'white');d=ImageDraw.Draw(image)
    d.line([(40,50),(220,50)],fill='red',width=4);d.polygon([(20,50),(65,32),(65,68)],fill='red')
    mask=Image.new('L',image.size);ImageDraw.Draw(mask).polygon([(20,50),(65,32),(65,68)],fill=255)
    arr=np.array(mask)>0
    answer=dict(nodes=[['a',20,50],['b',220,50]],routes=[dict(id='r',sources=['G1'],masks=['M1'],kind='straight',
        proposal=None,nodes=['a','b'],geometry='mask',network=None,heads=['triangle','none'],line_style='solid',shaft_sample=[[100,50],[200,50]])],excluded=[],uncertain=[])
    answer=semantic_geometry.validate(answer,image.size,['G1'],{},['M1'])
    class NoExtraModel:
        def ask(self,*args):raise AssertionError('Simple selected mask/sample should not need another model request')
    rows,supports,issues,calls=geometry_routing.recover(answer,{},[dict(id='M1',path='unused',mask=arr)],'G1',image,tmp_path,NoExtraModel(),dict(cpu=threading.Semaphore(1)))
    assert not issues and calls==0 and len(rows)==1
    row=rows[0];assert abs(row['start'][0]-20)<1 and abs(row['end'][0]-220)<1
    measured=style_windows.recover(np.array(image),supports[row['id']],row)
    assert abs(measured['width_px']-4)<.6 and measured['start']['type']=='triangle'


def test_shaft_sample_overrides_head_mask_without_changing_geometry():
    im=Image.new('RGB',(240,100),'white');d=ImageDraw.Draw(im)
    for x in range(20,170,25):d.line([(x,50),(x+14,50)],fill='#d02030',width=4)
    d.polygon([(200,50),(170,30),(170,70)],fill='#d02030')
    mask=np.zeros((100,240),bool);mask[30:71,170:201]=True
    row=dict(id='a',kind='straight',start=[20,50],end=[200,50],via=[],heads=['none','triangle'],
             start_head=False,end_head=True,end_head_type='triangle',line_style='dashed',shaft_sample=[[25,50],[155,50]])
    original=copy.deepcopy(row);result=style_windows.recover(np.array(im),mask,row)
    assert row==original
    assert abs(result['width_px']-4)<.6
    assert abs(result['dash']['period_px']-25)<1
    assert result['end']['type']=='triangle'
    assert result['evidence']['source']=='glm_selected_shaft_clean_pixels'


def test_sample_absence_and_unresolved_are_distinct():
    row=dict(id='a',kind='straight',start=[10,20],end=[90,20],via=[],heads=['none','none'],
             start_head=False,end_head=False,line_style='solid')
    im=Image.new('RGB',(100,50),'white');ImageDraw.Draw(im).line([(10,20),(90,20)],fill='black',width=3)
    regular=style_windows.recover(np.array(im),None,row)
    assert regular==style_windows.recover(np.array(im),None,dict(row,shaft_sample='auto'))
    missing=style_windows.recover(np.array(im),None,dict(row,shaft_sample=None))
    assert missing['width_px'] is None and style.pending(dict(row,style=missing))


def test_sample_coordinate_conversion_and_invalid_values():
    row=dict(id='a',kind='straight',start=[0,0],end=[100,0],via=[],shaft_sample=[[20,0],[80,0]])
    view=dict(view_to_global=[[.2,0,10],[0,.2,20],[0,0,1]])
    result=fit.transform(row,view)
    assert result['shaft_sample']==[[14,20],[26,20]] and row['shaft_sample']==[[20,0],[80,0]]
    for bad in ([[5,5],[5,5]],[[0,0],[200,0]],[[float('nan'),0],[10,0]],'head'):
        with pytest.raises(ValueError):shaft_sampling.validate(bad,[100,100])


@pytest.mark.parametrize('reverse',[False,True])
def test_two_marks_clipped_by_sample_endpoints_can_measure_period(reverse):
    im=Image.new('RGB',(240,70),'white');d=ImageDraw.Draw(im)
    # A disjoint far-away mark must not influence this selected sample.
    for a,b in [(10,25),(70,89),(110,129),(170,225)]:d.line([(a,35),(b,35)],fill='#d02030',width=4)
    sample=[[80,35],[120,35]]
    if reverse:sample=sample[::-1]
    pattern=shaft_sampling.complete_sample_marks(np.array(im),sample,4,'#d02030')
    assert pattern['complete_marks']==2 and pattern['observed_gaps']==1
    assert abs(pattern['mark_px']-20)<1 and abs(pattern['gap_px']-20)<1
    assert abs(pattern['period_px']-40)<1
