from full_recovery.quality_choice import apply_choices


def row(ident,layer):
    return {'id':ident,'key':ident,'layer':layer,'refs':[ident],'keys':[ident]}


def test_legacy_arrow_only_vote_cannot_remove_delivered_branch_content():
    rows=[row('a','arrows'),row('b','arrows'),row('independent','arrows')]
    selected=apply_choices(rows,[{'candidate_ids':['a','b'],'keep_ids':['a']}])
    assert selected['drop_arrow_names']==[] and selected['removed']==[]


def test_cross_branch_choice_preserves_its_existing_authority():
    rows=[row('a','arrows'),row('s','shape'),row('independent','arrows')]
    selected=apply_choices(rows,[{'candidate_ids':['a','s'],'keep_ids':['s']}])
    assert selected['drop_arrow_names']==['a'] and selected['drop_shape_names']==[]


def test_diverging_wings_stay_with_their_whole_native_return_path():
    import numpy as np
    from full_recovery.compound_routes import recognize
    body=np.array([[100,100],[0,100],[0,20],[80,20],[90,16]],float)
    wing=np.array([[80,20],[90,24]],float)
    found=recognize([body,wing],1)
    assert found is not None and found['attached_stroked_head']
    assert found['heads']==['none','open']
    assert recognize([body,wing+np.array([0,10])],1) is None

