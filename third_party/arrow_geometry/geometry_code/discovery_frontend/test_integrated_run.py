import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from integrated_run import prepare,run_case,read,write,STAGES

class IntegratedRunTests(unittest.TestCase):
    def test_fresh_png_runs_stages_in_order_and_keeps_sam_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);image=tmp/'example.png';Image.new('RGB',(80,60),'white').save(image)
            root=tmp/'run';prepare(root,[image]);cfg=read(root/'config.json');seen=[]
            base={'id':'old','display_id':'I0001','kind':'single','bbox_native':[5,20,45,28],
                  'locator':'arrow','source_ids':['observation'],'extent':'complete','uncertain':False,'verified_independent':True}
            addition={**base,'id':'post_sam/N1','display_id':'I0002','bbox_native':[50,30,70,38],'source_ids':['sam/M1']}
            def stage(name):
                def call(folder,cid,*args):
                    seen.append(name);c=folder/'cases'/cid
                    if name=='discovery':result={'objects':[copy.deepcopy(base)]}
                    else:result=read(c/'baseline_result.json')
                    if name=='partition':result['region_ledger']=[{'parent_id':'retired','commit':True,'active_owner_ids':['old'],'coverage_checks':[]}]
                    if name=='post_sam':result={**result,'objects':result['objects']+[addition],'added_count':1,'pending_count':0,'source_objects_unchanged':True}
                    write(c/'result.json',result);write(c/'finished.json',{})
                    return result
                return call
            with patch('integrated_run.light_run.run_case',stage('discovery')),patch('integrated_run.region_partition.execute_case',stage('partition')),patch('integrated_run.owner_refs.execute_case',stage('owners')),patch('integrated_run.post_tile_sam.execute_case',stage('post_sam')):
                result=run_case(root,'example',{s:None for s in STAGES},cfg)
                self.assertEqual(seen,list(STAGES));self.assertEqual(result['objects'][0],base)
                self.assertEqual(result['sam_added_count'],1);self.assertTrue(result['sam_existing_unchanged'])
                again=run_case(root,'example',{s:None for s in STAGES},cfg)
                self.assertEqual(again,result);self.assertEqual(seen,list(STAGES))
            self.assertTrue((root/'reports/完整结果对照.md').exists())
            self.assertTrue((root/'cases/example/reconstruction_handoff.json').exists())
            handoff=read(root/'cases/example/reconstruction_handoff.json')
            self.assertEqual(handoff['context_regions'][0]['owner_ids'],['old'])

if __name__=='__main__':unittest.main()
