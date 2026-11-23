"""Connected branch choices and source-based topology diagnostics."""
import unittest
from full_recovery.quality_choice import candidate_components,comparison_groups,apply_choices
from shape_integration.fitting.connectivity import fitted_connections
from shape_integration.fitting.critic import source_observation_contract,validate_review

class NetworkTests(unittest.TestCase):
    def test_enclosing_panel_does_not_steal_rectangular_node_endpoints(self):
        parts={n:{'status':'ok','kind':'rectangle','bbox_2d':b} for n,b in
               [('panel',[0,0,200,150]),('a',[20,20,40,40]),('b',[140,100,160,120])]}
        parts['edge']={'status':'ok','kind':'path','commands':[['M',40,30],['L',140,110]]}
        result=fitted_connections(parts)
        self.assertEqual(result['routes'][0]['nodes'],['a','b'])

    def test_connected_network_keeps_unpaired_routes_outside_local_choices(self):
        rows=[{'id':n,'layer':branch,'owner':'same','endpoint_nodes':pair,'key':n,
               'keys':[n] if branch=='shape' else [],'refs':[n]}
              for n,branch,pair in [('s1','shape',('a','b')),('s2','shape',('b','c')),
                                   ('a1','arrows',('a','b')),('a2','arrows',('b','c')),
                                   ('unpaired','arrows',('a','c')),('separate','shape',('x','y'))]]
        pairs=[{'id':'p1','a':'s1','b':'a1'},{'id':'p2','a':'s2','b':'a2'}]
        groups=comparison_groups(pairs,rows);self.assertEqual(len(groups),2)
        component=candidate_components(rows)['s1'];self.assertEqual(component,{'s1'})
        plan=apply_choices(rows,[{'keep':'shape','candidate_ids':sorted({p[k] for p in group for k in ('a','b')})}
                                for group in groups])
        self.assertEqual(plan['drop_arrow_names'],['a1','a2'])
        self.assertNotIn('separate',plan['drop_shape_keys'])

    def test_crossing_lines_without_shared_nodes_are_separate(self):
        rows=[{'id':n,'endpoint_nodes':p} for n,p in [('s1',('a','b')),('a1',('a','b')),('s2',('c','d')),('a2',('c','d'))]]
        pairs=[{'id':'p1','a':'s1','b':'a1'},{'id':'p2','a':'s2','b':'a2'}]
        self.assertEqual(len(comparison_groups(pairs,rows)),2)

    def test_wrong_endpoint_detected_despite_distinct_semantic_names(self):
        parts={n:{'status':'ok','kind':'ellipse','bbox_2d':b} for n,b in
               [('top',[40,0,60,20]),('left',[0,80,20,100]),('right',[80,80,100,100])]}
        parts.update({n:{'status':'ok','kind':'path','commands':[['M',50,20],['L',x,80]]}
                      for n,x in [('edge_to_left',10),('edge_to_right',12)]})
        result=fitted_connections(parts)
        self.assertEqual(len(result['routes']),2)
        self.assertEqual(result['same_endpoint_routes'][0]['nodes'],['left','top'])
        self.assertEqual(len(parts),5)  # diagnostics never mutate or delete geometry

    def test_parallel_connections_require_source_evidence_not_automatic_rejection(self):
        prompt={'network_connections':{'same_endpoint_routes':[]}}
        ids=source_observation_contract(prompt,{'F1':'edge'})
        self.assertEqual(ids,['connectivity'])
        with self.assertRaises(ValueError):validate_review({'decision':'accept','issues':[]},['F1'],ids)
        value={'decision':'accept','issues':[],'source_observations':[{
            'id':'connectivity','source_relation':'supported','evidence':'CLEAN has two distinct parallel routes connecting the nodes.'}]}
        self.assertEqual(validate_review(value,['F1'],ids),value)
