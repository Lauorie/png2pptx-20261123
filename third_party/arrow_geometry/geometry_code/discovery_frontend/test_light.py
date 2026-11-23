import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from light_core import compress,classify,repair_batches,select_clean_sam
from light_run import run_case
from pipeline_frames import Crop
from transport.common import write_json as write


def row(oid,b,kind='single',extent='complete',view='G'):
    return {'id':oid,'kind':kind,'extent':extent,'bbox_native':b,'locator':'visible arrow',
            'source_view':view,'sam_only':False}


class LightTests(unittest.TestCase):
    def test_one_pixel_line_keeps_pixel_coverage_and_recall(self):
        import numpy as np
        from evidence import sam_only_observations,select_masks
        from transport.common import sha256
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'line.png';pixels=np.zeros((30,30),dtype='uint8');pixels[8:15,10]=255
            Image.fromarray(pixels).save(p);view=Crop((30,30),(0,0,30,30),(30,30))
            mask={'id':'s','mask_path':str(p),'mask_sha256':sha256(p),'frame':view.metadata(view.global_size),
                  'bbox_mask':[10,8,10,14],'bbox_native':[10,8,10,14],'category':'line','score':.9}
            covered={'id':'g','bbox_native':[5,5,20,20]}
            proposed,audit=sam_only_observations([mask],[covered])
            self.assertEqual(proposed,[]);self.assertEqual(audit[0]['fraction_inside_glm_boxes'],1)
            proposed,audit=sam_only_observations([mask],[{'id':'g','bbox_native':[18,18,25,25]}])
            self.assertEqual(len(proposed),1);self.assertEqual(audit[0]['fraction_inside_glm_boxes'],0)
            self.assertEqual(len(select_masks([mask],[covered],view)[0]),1)

    def test_identical_multiview_candidates_become_one_union(self):
        rs=[row('G/A1',[10,10,100,20]),row('T/A1',[9,11,101,21],view='T')]
        out,audit=compress(rs)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['bbox_native'],[9,10,101,21])
        self.assertEqual(set(out[0]['source_ids']),{'G/A1','T/A1'})

    def test_contained_fragment_absorbed_by_full_arrow(self):
        rs=[row('g',[10,10,100,20]),row('t',[10,11,35,19],extent='partial',view='T')]
        self.assertEqual(len(compress(rs)[0]),1)

    def test_neighboring_parallel_and_serial_arrows_survive(self):
        rs=[row('a',[10,10,100,20]),row('b',[10,30,100,40],view='T'),row('c',[105,10,190,20],view='U')]
        self.assertEqual(len(compress(rs)[0]),3)

    def test_growing_union_does_not_chain_across_array(self):
        rs=[row('a',[0,0,100,10]),row('b',[20,0,120,10],view='T'),row('c',[40,0,140,10],view='U')]
        self.assertEqual(len(compress(rs)[0]),2)

    def test_same_view_distinct_objects_are_protected(self):
        self.assertEqual(len(compress([row('a',[0,0,100,10]),row('b',[5,0,105,10])])[0]),2)

    def test_dense_group_interior_and_external(self):
        rs=[row('n',[0,0,100,100],'dense_network'),row('i',[20,40,70,45],view='T'),row('o',[98,40,130,45],view='T')]
        out,a=compress(rs)
        self.assertEqual(len(out),2);self.assertEqual(len(a['network_members']),1)
        self.assertTrue(any('o' in r['source_ids'] and r['kind']=='single' for r in out))

    def test_explicit_model_split_not_remerged(self):
        a=row('a',[0,0,100,20]);b=row('b',[0,0,100,20],view='T')
        a['source_ids']=b['source_ids']=['same_original']
        self.assertEqual(len(compress([a,b],protect_splits=True)[0]),2)

    def test_mask_cannot_expand_far_outside_glm_scope(self):
        r=row('a',[10,10,100,20])
        bad={'id':'m','bbox_native':[0,0,100,100]}
        self.assertIsNone(select_clean_sam(r,[bad])[0])

    def test_unique_close_mask_and_competing_masks(self):
        r=row('a',[10,10,100,20]);a={'id':'a','bbox_native':[12,11,98,19]}
        self.assertEqual(select_clean_sam(r,[a])[0]['id'],'a')
        b={'id':'b','bbox_native':[20,10,90,20]}
        self.assertEqual(select_clean_sam(r,[a,b])[1],'competing_masks')

    def test_only_ambiguous_candidates_enter_repair(self):
        rs=classify(compress([row('a',[10,10,100,20]),row('b',[300,30,390,40])])[0])
        self.assertEqual(repair_batches(rs,(500,100),{'repair_batch_max':10,'repair_neighbor_gap':160,'repair_span_px':720}),[])

    def test_offline_end_to_end_sources_and_pending(self):
        for partial in [False,True]:
            with self.subTest(partial=partial),tempfile.TemporaryDirectory() as t:
                root=Path(t);cid='fixture';inp=root/'inputs'/cid;inp.mkdir(parents=True)
                im=Image.new('RGB',(100,100),'white');im.save(inp/'source.png');views=[]
                for vid,scale in [('G',1),('T',2)]:
                    v=Crop((100,100),(0,0,100,100),(100*scale,100*scale));f=inp/vid;f.mkdir();v.image(im).save(f/'target.png')
                    views.append({'view_id':vid,'path':str((f/'target.png').relative_to(root)),'transform':v.metadata((100,100))})
                write(inp/'views.json',{'views':views})
                cfg={'view_workers':16,'refine_workers':16,'sam_evidence_max':8,'repair_batch_max':10,'repair_neighbor_gap':160,'repair_span_px':720,
                     'sam_extra_min_score':.65,'sam_extra_uncovered_fraction':.8,'sam_extra_limit':8}
                class Fake:
                    config=cfg
                    def ask(self,path,q,images,validator):
                        path.mkdir(parents=True,exist_ok=True)
                        if 'focus' not in q:
                            s=q['target_size_px'][0]/100
                            a={'objects':[{'id':'A1','kind':'single','bbox':[(v+.5)*s-.5 for v in [20,28,60,34]],'locator':'arrow','extent':'partial' if partial else 'complete'}]}
                        else:
                            ids=[r['id'] for r in q['focus']]
                            a={'objects':[{'id':'O1','source_ids':ids,'kind':'single','locator':'arrow','extent':'complete',
                                'bbox_source':'model','bbox':q['focus'][0]['bbox'],'original_id':None,'sam_ids':[],'complete_support':False,'uncertain':False,'reason':'visible'}],
                                'removed':[],'unresolved':[]}
                        return validator(a)
                with patch('light_run.publish',lambda r:None),patch('light_run.global_sam',lambda *a:[]),patch('light_run.local_sam',lambda root,cid,r,cfg:(r,[])):
                    result=run_case(root,cid,Fake(),'scope')
                self.assertEqual(result['output_identity_count'],1);self.assertEqual(len(result['source_ledger']),2)
                self.assertEqual(result['repair_batches'],int(partial));self.assertFalse(result['objects'][0]['geometry_ready'])


if __name__=='__main__':unittest.main()
