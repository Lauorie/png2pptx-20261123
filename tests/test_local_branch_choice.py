import numpy as np
import pytest

from full_recovery.quality_choice import comparison_groups,validate_selection


def test_local_selection_cannot_keep_both_implementations():
    branches={'C1':'shape','C2':'arrows','C3':'arrows'}
    assert validate_selection({'keep_ids':['C2','C3'],'reason':'Complementary native fragments'},
        branches,branch_by_id=branches)['keep_ids']==['C2','C3']
    with pytest.raises(ValueError,match='one branch'):
        validate_selection({'keep_ids':['C1','C2'],'reason':'Both versions'},
            branches,branch_by_id=branches)
    with pytest.raises(ValueError):
        validate_selection({'keep_ids':[],'reason':'Remove both'},branches,branch_by_id=branches)


def test_shared_owner_and_node_do_not_join_independent_subpaths():
    rows=[{'id':n,'owner':'same_network','parent_key':'same_compound','endpoint_nodes':['shared','other'],
           'points':np.array([[0.,y],[40.,y]])}
          for n,y in [('shape:a',10),('arrow:a',10),('shape:b',30),('arrow:b',30)]]
    pairs=[{'id':'p1','a':'shape:a','b':'arrow:a'},
           {'id':'p2','a':'shape:b','b':'arrow:b'}]
    assert len(comparison_groups(pairs,rows))==2
