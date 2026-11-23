import copy
import numpy as np
import pytest
from PIL import Image, ImageDraw
import mask_vectors
from lib import style


def test_head_only_support_preserves_complete_target_extent():
    image=Image.new('L',(260,100));ImageDraw.Draw(image).polygon([(20,50),(65,32),(65,68)],fill=255)
    row=mask_vectors.fit_route(np.asarray(image)>0,'straight',[[20,50],[220,50]],['triangle','none'])
    assert abs(row['start'][0]-20)<1 and abs(row['end'][0]-220)<1


def test_two_target_ranges_remain_distinct_on_one_mask():
    mask=np.zeros((100,280),bool);mask[48:53,15:260]=True
    a=mask_vectors.fit_route(mask,'straight',[[20,50],[100,50]],['none','none'])
    b=mask_vectors.fit_route(mask,'straight',[[170,50],[250,50]],['none','triangle'])
    assert a['end'][0]<110 and b['start'][0]>160


def test_complete_proposal_keeps_its_pixel_endpoints():
    mask=np.zeros((100,280),bool);mask[48:53,15:260]=True
    p=dict(kind='straight',start=[15,50],via=[],end=[259,50],width_estimate=5,masks=['M1'])
    answer=dict(nodes=[['a',15,50],['b',259,50]],routes=[dict(id='r',geometry='mask',kind='straight',
        proposal='D1',nodes=['a','b'],masks=['M1'],sources=['G1'],heads=['none','triangle'],line_style='solid')])
    rows,_,_=mask_vectors.physical_rows(answer,{'D1':p},[dict(id='M1',mask=mask,path='unused')],'G1')
    assert rows[0]['start']==p['start'] and rows[0]['end']==p['end']


@pytest.mark.parametrize('scale',[1,4])
def test_offset_shaft_measurement_is_scale_consistent(scale):
    im=Image.new('RGB',(280,130),'white');d=ImageDraw.Draw(im)
    d.line([(30,60),(230,60)],fill='black',width=4)
    d.polygon([(230,60),(207,45),(207,75)],fill='black')
    im=im.resize((280*scale,130*scale),Image.Resampling.LANCZOS)
    row=dict(id='r',kind='straight',start=[30*scale,63*scale],end=[230*scale,63*scale],via=[],
        heads=['none','triangle'],start_head=False,end_head=True,end_head_type='triangle',line_style='solid')
    before=copy.deepcopy(row);measured=style.recover(np.asarray(im),None,row,adaptive_heads=True)
    assert row==before
    assert abs(measured['width_px']/scale-4)<.8
