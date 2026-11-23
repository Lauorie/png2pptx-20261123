"""Stroked terminal wings stay with the exact native object they belong to."""
from copy import deepcopy
import numpy as np

from full_recovery.compound_routes import recognize
from full_recovery.quality_choice import candidates,apply_choices


def test_inline_wing_does_not_replace_the_shaft_endpoint_or_leave_an_orphan():
    commands=[['M',0,0],['L',100,0],['L',92,4],['M',92,-4],['L',100,0]]
    part={'object_id':'o','id':'route','status':'ok','kind':'path',
          'bbox_2d':[0,-4,100,4],'commands':commands}
    before=deepcopy(part)
    rows=candidates({'parts':{'route':part}},
        {'route':{'fill':None,'stroke':'#000000','width_px':1}},[])
    assert len(rows)==1 and rows[0]['id']=='shape:route'
    np.testing.assert_allclose(rows[0]['points'][[0,-1]],[[0,0],[100,0]])
    assert rows[0]['attached_stroked_head'] and part==before
    result=apply_choices(rows,[{'candidate_ids':['shape:route'],'keep_ids':[]}])
    assert result['drop_shape_keys']==['route']
    assert result['shape_subpath_suppression']=={}


def test_standalone_v_and_reversed_inline_wing_share_the_real_tip():
    for lines in [
        [np.array([[0,0],[100,0]]),np.array([[92,-4],[100,0],[92,4]])],
        [np.array([[92,4],[100,0],[0,0]]),np.array([[100,0],[92,-4]])]]:
        result=recognize(lines,1)
        assert result is not None and result['attached_stroked_head']
        np.testing.assert_allclose(result['points'][[0,-1]],[[0,0],[100,0]])


def test_distinct_parallel_strokes_and_detached_small_routes_stay_separate():
    assert recognize([np.array([[0,0],[100,0]]),np.array([[90,1],[100,1]])],1) is None
    assert recognize([np.array([[0,0],[100,0],[92,4]]),np.array([[92,-8],[100,-4]])],1) is None
    assert recognize([np.array([[0,0],[100,0]]),np.array([[0,10],[100,10]])],1) is None
