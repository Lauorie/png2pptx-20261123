"""A distant native face cannot be selected against an unrelated shaft."""
import numpy as np

from full_recovery.quality_choice import filled_routes
from full_recovery.handoff import resample


def propose(box,x,width=8.):
    x0,y0,x1,y1=box
    part={'status':'ok','kind':'rectangle','object_id':'owner','id':'face',
          'bbox_2d':box,'commands':[['M',x0,y0],['L',x1,y0],['L',x1,y1],['L',x0,y1],['Z']]}
    points=resample([np.array([[x,y0],[x,y1]],float)])
    arrow={'id':'arrow:route','layer':'arrows','points':points,'length':y1-y0,
           'style':{'color':'#AAAAAA','width_px':width},'heads':['none','none']}
    return filled_routes({'parts':{'face':part}},
                         {'face':{'fill':'#D9D9D9','stroke':'#000000','width_px':2}},[arrow],set())


def test_long_unrelated_face_does_not_compete_with_a_parallel_route():
    assert propose([80,0,140,400],25)==[]


def test_real_filled_shaft_and_small_localization_difference_remain_comparable():
    assert len(propose([20,0,28,100],24,2))==1
    assert len(propose([20,0,28,100],29,2))==1
