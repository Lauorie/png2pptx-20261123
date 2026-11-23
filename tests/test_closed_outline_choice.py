import numpy as np

from full_recovery.quality_choice import candidates, nominate, allowed_branches
from shape_integration.fitting.paths import ellipse_commands


def outline(fill):
    scene={'parts':{'loop':{'id':'loop','object_id':'o','status':'ok','kind':'ellipse',
        'bbox_2d':[20,20,80,80],
        'native_parameters_original':{'center':[50,50],'width':60,'height':60,'angle_radians':0},
        'commands':ellipse_commands([50,50],[30,0],[0,30])}}}
    styles={'loop':{'fill':fill,'stroke':'#000000','width_px':2,'dash':'solid'}}
    return scene,styles,candidates(scene,styles,[])


def arc(center=(50,50)):
    angles=np.linspace(.25*np.pi,1.75*np.pi,80)
    points=np.asarray(center)+30*np.c_[np.cos(angles),np.sin(angles)]
    return {'id':'arrow:arc','layer':'arrows','points':points,
        'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
        'style':{'width_px':2},'heads':['none','none'],'closed':False}


def test_native_loop_and_open_visible_arc_reach_one_source_comparison():
    for fill in (None,'#FFFFFF'):
        scene,styles,rows=outline(fill)
        assert len(rows)==1 and rows[0]['closed']
        rows.append(arc())
        pairs=nominate(rows)
        assert len(pairs)==1 and pairs[0]['nomination']=='overlapping_curved_routes'
        # Filled native objects keep their complete paint under the existing
        # immutable-border contract; ownership must not remove the node/face.
        assert allowed_branches(pairs,rows,scene,styles)==(['shape'] if fill else ['shape','arrows'])


def test_intersecting_loop_with_different_extent_is_not_an_alternative():
    _,_,rows=outline('#FFFFFF')
    rows.append(arc((90,50)))
    assert not nominate(rows)
