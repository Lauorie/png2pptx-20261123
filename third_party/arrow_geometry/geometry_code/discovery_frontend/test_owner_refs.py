import copy
import unittest
from owner_refs import absorbed_candidates, risky_targets, accept_identity, reconcile, relation_pairs, attach_recovery_links

def obj(i,b,kind='single',verified=False):
    return {'id':i,'kind':kind,'bbox_native':b,'extent':'complete','locator':i,
            'source_ids':[i],'uncertain':False,'verified_independent':verified}

class OwnerPolicyTests(unittest.TestCase):
    def test_contained_arrow_is_risky_not_removed(self):
        a=obj('a',[20,20,40,30]);g=obj('g',[0,0,100,100],'dense_network')
        self.assertEqual([x['id'] for x in risky_targets([a,g],[],set())],['a'])
        self.assertEqual(len([a,g]),2)

    def test_recover_absorbed_candidates_without_suppressing_nested_lines(self):
        a=obj('a',[20,20,40,30]);b=obj('b',[20,21,40,31]);g=obj('g',[0,0,100,100],'dense_network')
        g['interior_member_observations']=[a,b]
        self.assertEqual(len(absorbed_candidates([g])),2)

    def test_failed_check_retains_original_and_recovered_candidate(self):
        a=obj('a',[20,20,40,30]);r=obj('r',[40,40,60,50]);proposal=obj('new',[21,20,40,30]);proposal['input_ids']=['a']
        active,_=accept_identity([a],[r],[{'objects':[proposal],'decisions':{'new':{'accepted':False}}}])
        self.assertEqual({o['id'] for o in active},{'a','r'})
        self.assertTrue(next(o for o in active if o['id']=='r')['uncertain'])

    def test_accepted_same_identity_preserves_all_source_responsibilities(self):
        a=obj('a',[20,20,40,30]);b=obj('b',[20,21,40,31]);proposal=obj('new',[20,20,40,31]);proposal['input_ids']=['a','b']
        active,audit=accept_identity([a,b],[],[{'objects':[proposal],'decisions':{'new':{'accepted':True}}}])
        self.assertEqual(len(active),1);self.assertEqual(set(active[0]['input_ids']),{'a','b'})

    def test_multiple_outputs_keep_original_residual(self):
        a=obj('a',[0,0,100,100]);p=obj('p',[0,0,40,20]);q=obj('q',[0,40,40,60])
        p['input_ids']=q['input_ids']=['a']
        active,_=accept_identity([a],[],[{'objects':[p,q],'decisions':{'p':{'accepted':True},'q':{'accepted':True}}}])
        self.assertEqual({o['id'] for o in active},{'a','p','q'})
        self.assertEqual(next(o for o in active if o['id']=='a')['certainty'],'source_split_pending')

    def test_high_overlap_does_not_merge_without_same(self):
        a=obj('a',[0,0,100,100],verified=True);b=obj('b',[0,1,100,101],verified=True)
        self.assertEqual(len(relation_pairs([a,b])),1)
        active,merged,_=reconcile([a,b],[{'a':'a','b':'b','relation':'different'}])
        self.assertEqual(len(active),2);self.assertFalse(merged)

    def test_transitive_conflict_blocks_merge(self):
        rows=[obj(i,[0,0,100,100],verified=True) for i in ['a','b','c']]
        ds=[{'a':'a','b':'b','relation':'same'},{'a':'b','b':'c','relation':'same'},{'a':'a','b':'c','relation':'different'}]
        active,_,conflicts=reconcile(rows,ds)
        self.assertEqual(len(active),2);self.assertEqual(len(conflicts),1)

    def test_alias_merge_keeps_original_input_ids(self):
        a=obj('a',[0,0,100,100],verified=True);b=obj('b',[0,1,100,101],verified=True)
        a['input_ids']=['x','y'];b['input_ids']=['z']
        active,_,_=reconcile([a,b],[{'a':'a','b':'b','relation':'same'}])
        self.assertEqual(set(active[0]['input_ids']),{'x','y','z'})

    def test_committed_spatial_children_remain_stable(self):
        a=obj('a',[0,0,100,100],verified=True);g=obj('g',[0,0,120,120],'dense_region')
        self.assertEqual(risky_targets([a,g],[],{'a'}),[])

    def test_new_identity_is_compared_to_unverified_existing_owner(self):
        old=obj('old',[0,0,100,20]);new=obj('new',[1,0,101,20],verified=True)
        self.assertEqual(len(relation_pairs([old,new])),1)

    def test_historical_owner_is_checked_even_after_bbox_moves(self):
        old=obj('old',[0,0,20,20]);rec=obj('REC1',[100,100,120,120]);rec['recovered_from_parent']='old'
        new=obj('new',[100,100,120,120],verified=True);new['input_ids']=['REC1']
        rows=attach_recovery_links([old,new],[rec])
        self.assertEqual(len(relation_pairs(rows)),1)
        self.assertNotIn('identity_parent_candidates',new)

    def test_relation_work_is_not_truncated_at_32(self):
        rows=[obj(str(i),[0,i,100,100+i],verified=True) for i in range(12)]
        self.assertEqual(len(relation_pairs(rows)),66)

    def test_stable_owner_receives_checked_identity_evidence(self):
        old=obj('old',[0,0,100,20]);old['display_id']='I0001';old['source_ids']=['old_obs']
        new=obj('new',[1,0,101,20],verified=True);new.update(certainty='independently_checked',locator='complete physical route',identity_check={'accepted':True},source_ids=['new_obs'])
        rows,_,_=reconcile([old,new],[{'a':'old','b':'new','relation':'same'}])
        self.assertEqual(rows[0]['id'],'old');self.assertEqual(rows[0]['display_id'],'I0001')
        self.assertTrue(rows[0]['verified_independent']);self.assertEqual(rows[0]['locator'],'complete physical route')
        self.assertEqual(set(rows[0]['source_ids']),{'old_obs','new_obs'})

    def test_committed_partition_id_wins_over_duplicate_legacy_id(self):
        a=obj('legacy',[0,0,100,100],verified=True);a['display_id']='I0001'
        b=obj('committed',[1,0,100,100],verified=True);b.update(display_id='I0080',certainty='prior_partition_checked')
        rows,_,_=reconcile([a,b],[{'a':'legacy','b':'committed','relation':'same'}])
        self.assertEqual(rows[0]['id'],'committed');self.assertEqual(rows[0]['display_id'],'I0080')

if __name__=='__main__':unittest.main()
