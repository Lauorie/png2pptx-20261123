from copy import deepcopy
import numpy as np
import pytest
from PIL import Image,ImageDraw
from full_recovery.source_route_nomination import shared_ridge
from full_recovery.quality_choice import nominate,apply_choices


def row(ident,layer,start,end,head='none'):
    points=np.linspace(start,end,80)
    return {'id':ident,'key':ident,'keys':[ident],'refs':[ident],'layer':layer,
            'points':points,'length':float(np.linalg.norm(points[-1]-points[0])),
            'heads':['none',head],'style':{'stroke':'#333333','color':'#333333','width_px':1}}


def fixture():
    image=Image.new('RGB',(180,110),'white');ImageDraw.Draw(image).line((35,45,125,45),fill='#333333',width=2)
    return image,[row('short','shape',[37,49],[89,49],'open'),row('full','arrows',[35,45],[125,45],'triangle')]


def test_shortened_headed_version_is_nominated_from_same_source_ridge():
    source,rows=fixture();before=deepcopy(rows)
    assert not nominate(rows)
    pairs=nominate(rows,np.asarray(source));assert len(pairs)==1
    assert pairs[0]['nomination']=='shared_original_straight_ridge'
    assert pairs[0]['source_relation']=='nomination_only_complete_native_review_required'
    assert not apply_choices(rows,[])['drop_shape_names']
    for a,b in zip(before,rows):np.testing.assert_array_equal(a['points'],b['points'])


def test_two_distinct_parallel_source_lines_remain_independent():
    source,rows=fixture();ImageDraw.Draw(source).line((35,54,125,54),fill='#333333',width=2)
    assert shared_ridge(*rows,np.asarray(source))is None
    rows[0]['points'][:,1]=54
    assert shared_ridge(*rows,np.asarray(source))is None


def test_blank_source_crossing_and_short_head_fragment_do_not_enter_review():
    source,rows=fixture()
    assert shared_ridge(*rows,np.full_like(np.asarray(source),255))is None
    crossing=row('vertical','arrows',[70,10],[70,95],'triangle')
    assert shared_ridge(rows[0],crossing,np.asarray(source))is None
    fragment=row('fragment','shape',[75,49],[88,49],'open')
    assert shared_ridge(fragment,rows[1],np.asarray(source))is None


def test_ambiguous_fit_between_two_source_ridges_abstains():
    source,rows=fixture();ImageDraw.Draw(source).line((35,53,125,53),fill='#333333',width=2)
    assert shared_ridge(*rows,np.asarray(source))is None


def test_existing_competition_is_not_nominated_again_or_changed_without_source():
    source,rows=fixture()
    from full_recovery.source_route_nomination import nominate as source_nominate
    existing=[{'id':'known','a':rows[0]['id'],'b':rows[1]['id']}]
    assert source_nominate(rows,np.asarray(source),existing)==[]
    assert nominate(rows)==[]


def test_complete_alternative_tail_keeps_its_whole_native_edit_scope():
    source,rows=fixture();rows.append(row('independent','arrows',[35,80],[125,80],'triangle'))
    pairs=nominate(rows,np.asarray(source))
    assert len(pairs)==1 and {pairs[0]['a'],pairs[0]['b']}=={'short','full'}
    selected=apply_choices(rows,[{'candidate_ids':['short','full'],'keep_ids':['full']}])
    assert selected['drop_shape_keys']==['short'] and selected['drop_arrow_names']==[]
    assert not selected['shape_segment_suppression'] and not selected['shape_subpath_suppression']
    np.testing.assert_array_equal(rows[1]['points'][-1],[125,45])


def test_headless_short_fragment_stays_in_existing_ownership_protocol():
    source,rows=fixture()
    for r in rows:r['heads']=['none','none']
    assert shared_ridge(*rows,np.asarray(source))is None


@pytest.mark.parametrize('scale,angle',[(1,0),(1,90),(2,28),(.7,160)])
def test_nomination_uses_source_scale_and_route_direction(scale,angle):
    source,rows=fixture();theta=np.radians(angle);m=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*scale
    image=Image.new('RGB',(500,500),'white');draw=ImageDraw.Draw(image)
    line=np.array([[35,45],[125,45]])@m.T+[170,170];draw.line([tuple(x)for x in line],fill='#333333',width=max(1,round(2*scale)))
    for row in rows:
        row['points']=row['points']@m.T+[170,170];row['length']*=scale;row['style']['width_px']*=scale
    assert shared_ridge(*rows,np.asarray(image))is not None
