import unittest
from residual_review import preserve_chain_scope,nominees
from test_light import row

class ResidualTests(unittest.TestCase):
    def test_chain_retains_complete_source_but_not_neighbor(self):
        source=row('G/a',[10,10,300,50]);neighbor=row('G/b',[50,80,250,110])
        o=row('o',[12,12,298,35],'serial_chain');o['source_ids']=['G/a','G/b']
        out,a=preserve_chain_scope([o],[source,neighbor])
        self.assertEqual(out[0]['bbox_native'],[10,10,300,50]);self.assertEqual(a[0]['source'],'G/a')
    def test_no_inferred_chain_expansion_for_displaced_or_independent_line(self):
        source=row('G/a',[10,10,300,50])
        objects=[row('o',[20,90,300,110],'serial_chain'),row('s',[12,12,298,35])]
        for o in objects:o['source_ids']=['G/a']
        out,a=preserve_chain_scope(objects,[source]);self.assertEqual(a,[])
        self.assertEqual([o['bbox_native'] for o in out],[o['bbox_native'] for o in objects])
    def test_two_box_nomination_is_not_a_merge(self):
        rs=[row('a',[0,0,100,20]),row('b',[0,3,100,23]),row('c',[0,40,100,60])]
        pairs,deferred=nominees(rs);self.assertEqual(len(pairs),1)
        self.assertEqual({pairs[0]['a'],pairs[0]['b']},{'a','b'});self.assertEqual(len(rs),3)
    def test_same_request_splits_do_not_become_duplicate_candidates(self):
        rs=[row('a',[0,0,100,20]),row('b',[0,0,100,20])]
        for r in rs:r['split_families']=['same_request/1']
        self.assertEqual(nominees(rs)[0],[])
    def test_single_head_fragment_is_compared_with_serial_chain_owner(self):
        from repair_policy import link_fragments
        chain=row('G/chain',[10,10,300,60],'serial_chain')
        head=row('T/head',[280,20,300,50],'single','partial',view='T')
        outside=row('T/other',[320,20,350,50],'single','partial',view='T')
        linked,links=link_fragments([chain,head,outside])
        self.assertEqual(links,[{'child':'T/head','possible_parent':'G/chain','confirmed_duplicate':False}])
        pairs,_=nominees([chain,head,outside])
        self.assertEqual(len(pairs),1)
        self.assertEqual({pairs[0]['a'],pairs[0]['b']},{'G/chain','T/head'})

if __name__=='__main__':unittest.main()
