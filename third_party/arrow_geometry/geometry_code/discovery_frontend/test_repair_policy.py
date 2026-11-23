import threading
import time
import unittest
from repair_policy import local_view,network_units,normalize_answer,link_fragments
from fair_pool import FairPool
from visual_labels import restore_answer_ids
from test_light import row

class RepairTests(unittest.TestCase):
    def test_all_local_frames_bounded_and_invertible(self):
        for size in [(40,20),(600,300),(1800,120),(4000,3000)]:
            v=local_view(size,(0,0,*size),{})
            self.assertTrue(1080<=max(v.input_size)<=2048)
            for point in [(0,0),(size[0]-1,size[1]-1),(15.2,5.1)]:
                out=v.to_global(v.to_local(point))
                for a,b in zip(point,out):self.assertAlmostEqual(a,b)

    def test_network_unit_keeps_fragments_for_semantic_review(self):
        rs=[row('G/n',[100,100,200,200],'dense_network'),
            row('T/edge',[120,99,170,150],view='T'),
            row('T/part',[110,120,180,190],'shared_network','partial',view='T'),
            row('T/output',[195,150,245,160],view='T'),
            row('G/out',[115,105,150,110])]
        for r in rs:r['source_ids']=[r['id']]
        out,audit=network_units(rs)
        self.assertEqual(len(out),3)
        unit=next(r for r in out if r.get('network_region'))
        self.assertEqual(len(unit['region_members']),3)
        self.assertFalse(audit[0]['identity_confirmed'])
        self.assertEqual({s for r in rs for s in r['source_ids']},{s for r in out for s in r['source_ids']})

    def test_extra_arrow_does_not_invalidate_focus(self):
        obj={'id':'a','source_ids':['1'],'kind':'single','bbox':[10,10,20,20],
             'extent':'complete','uncertain':False,'locator':'line','reason':'visible'}
        extra={**obj,'id':'b','source_ids':[],'bbox':[50,50,60,60]}
        out=normalize_answer({'objects':[obj,extra],'removed':[],'unresolved':[]},(100,100),{'1':'C1'},restore_answer_ids)
        self.assertEqual(len(out['objects']),1);self.assertEqual(len(out['extras']),1)
        with self.assertRaises(ValueError):
            normalize_answer({'objects':[extra],'removed':[],'unresolved':[]},(100,100),{'1':'C1'},restore_answer_ids)

    def test_shared_pool_uses_spare_capacity_and_keeps_cap(self):
        pool=FairPool(24);gate=threading.Event();lock=threading.Lock();stats=[0,0]
        def task():
            with lock:stats[0]+=1;stats[1]=max(stats)
            gate.wait(2)
            with lock:stats[0]-=1
        fs=[pool.submit('image_a',task) for _ in range(30)]
        deadline=time.monotonic()+2
        while stats[1]<24 and time.monotonic()<deadline:time.sleep(.001)
        gate.set()
        for f in fs:f.result()
        pool.shutdown();self.assertEqual(stats[1],24)

    def test_empty_dispositions_and_literal_boolean_keep_strict_accounting(self):
        obj={'id':'a','source_ids':['1'],'kind':'serial_chain','bbox':[10,10,20,20],
             'extent':'complete','uncertain':'false','locator':'continuous connection','reason':'all heads visible'}
        out=normalize_answer({'objects':[obj]},(100,100),{'1':'C1'},restore_answer_ids)
        self.assertIs(out['objects'][0]['uncertain'],False)
        self.assertEqual(out['removed'],[])
        with self.assertRaises(ValueError):
            normalize_answer({'objects':[obj]},(100,100),{'1':'C1','2':'C2'},restore_answer_ids)

if __name__=='__main__':unittest.main()
