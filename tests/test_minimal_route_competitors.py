"""Source routes remain independent of branch provenance and node containment."""
import numpy as np
from full_recovery.quality_choice import candidates, nominate, apply_choices
from full_recovery.line_group_choice import eligible


def test_a_filled_graph_face_inside_a_node_is_not_a_connection_shaft():
    face={'id':'face','object_id':'graph','status':'ok','kind':'path',
          'bbox_2d':[30,20,70,80],
          'commands':[['M',50,20],['L',70,80],['L',30,80],['Z']]}
    node={'id':'node','object_id':'graph','status':'ok','kind':'ellipse',
          'bbox_2d':[0,0,100,100]}
    arrow={'id':'edge','start':[50,20],'end':[50,80],
           'style':{'color':'#AAAAAA','width_px':2},'heads':['none','none']}
    rows=candidates({'parts':{'face':face,'node':node}},
                    {'face':{'fill':'#BBBBBB','stroke':'#000000','width_px':1}},[arrow])
    assert not any(row.get('filled_route') for row in rows)


def test_a_transitive_group_with_same_branch_duplicates_needs_subset_choice():
    rows=[{'id':n,'layer':layer}for n,layer in [('a','arrows'),('s1','shape'),('s2','shape')]]
    assert eligible(rows,[{'a':'a','b':'s1'},{'a':'a','b':'s2'}])
    assert not eligible(rows,[{'a':'a','b':'s1'},{'a':'s1','b':'s2'}])
    assert not eligible(rows+[{'id':'a2','layer':'arrows'}])


def test_delivered_arrow_branch_has_no_internal_review_or_deletion():
    def edge(ident,start,end):
        p=np.linspace(start,end,80)
        return {'id':ident,'key':ident,'keys':[],'refs':[ident],
                'layer':'arrows','points':p,'length':float(np.linalg.norm(p[-1]-p[0])),
                'endpoint_nodes':('u','v'),'node_radius_px':15,
                'style':{'color':'#888888','width_px':1},'heads':['none','none']}
    rows=[edge('a',[20,20],[50,140]),edge('b',[29,17],[54,139])]
    pairs=nominate(rows)
    assert pairs==[]
    rows.append(edge('unmatched',[100,20],[130,140]))
    choice={'candidate_ids':['a','b'],'keep_ids':['a']}
    result=apply_choices(rows,[choice])
    assert result['drop_arrow_names']==[]
