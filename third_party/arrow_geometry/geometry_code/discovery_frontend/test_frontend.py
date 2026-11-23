import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from core import batches,resolve,ledger,make_pairs,pending_object
from evidence import project,frame
from frontend import materialize,run_case
from pipeline_frames import Crop
from protocol import validate_refine,validate_discovery,validate_relations
from transport.client import build_payload
from transport.common import sha256,write_json as write

def obj(oid,box,kind='single',sources=None):
 return {'id':oid,'kind':kind,'bbox_native':box,'source_ids':sources or [oid], 'extent':'complete','locator':oid,
         'bbox_source':'sam','uncertain':False,'reason':'visible object','sam_ids':[]}

def answer(oid='O1',source='a',masks=None):
 return {'objects':[{'id':oid,'source_ids':[source],'kind':'single','locator':'arrow','extent':'complete','bbox_source':'sam' if masks else 'model',
                     'sam_ids':masks or [],'complete_support':bool(masks),'bbox':None if masks else [10,10,20,20],'uncertain':False,'reason':'visible'}], 'removed':[],'unresolved':[]}

class FrontendTests(unittest.TestCase):
 def test_ownership_batches_partition(self):
  rows=[obj(str(i),[i*3,10,i*3+8,18]) for i in range(19)]+[obj('net',[0,0,100,100],'dense_network')]
  cfg={'simple_batch_max':8,'complex_batch_max':2,'neighbor_gap_native_px':64,'simple_batch_span_native_px':480,'context_padding_native_px':48,'context_candidate_max':8}
  plan=batches(rows,(800,600),cfg);owners=[s for b in plan for s in b['focus_ids']]
  self.assertEqual(len(owners),len(set(owners)));self.assertEqual(set(owners),{r['id'] for r in rows})
  self.assertTrue(all(not set(b['focus_ids'])&set(b['context_ids']) for b in plan))

 def test_missing_source_rejected(self):
  with self.assertRaises(ValueError):validate_refine(answer(),(100,100),['a','b'],[])

 def test_context_cannot_be_mutated(self):
  with self.assertRaises(ValueError):validate_refine(answer(source='neighbor'),(100,100),['a'],[])

 def test_shared_complete_mask_cannot_be_two_single_objects(self):
  a=answer(masks=['M1']);a['objects']+=answer('O2','b',['M1','M2'])['objects']
  with self.assertRaises(ValueError):validate_refine(a,(100,100),['a','b'],['M1','M2'])

 def test_same_merges_with_explicit_evidence(self):
  rows=[obj('a',[0,0,10,10]),obj('b',[1,0,11,10])];pairs=[{'id':'p','a':'a','b':'b'}]
  result=resolve(rows,pairs,[{'pair_id':'p','relation':'same','reason':'same arrow'}])
  self.assertEqual(len(result['objects']),1);self.assertEqual(set(result['objects'][0]['source_ids']),{'a','b'})

 def test_overlapping_independent_arrows_survive(self):
  rows=[obj('a',[0,0,10,10]),obj('b',[0,0,10,10])];pairs=[{'id':'p','a':'a','b':'b'}]
  self.assertEqual(len(resolve(rows,pairs,[{'pair_id':'p','relation':'distinct','reason':'different crossings'}])['objects']),2)

 def test_transitive_distinct_guard(self):
  rows=[obj(k,[0,0,10,10]) for k in ['a','b','c']];pairs=[{'id':'1','a':'a','b':'b'},{'id':'2','a':'b','b':'c'},{'id':'3','a':'a','b':'c'}]
  ds=[{'pair_id':str(i+1),'relation':r,'reason':'visible'} for i,r in enumerate(['same','same','distinct'])]
  result=resolve(rows,pairs,ds);self.assertEqual(len(result['objects']),2);self.assertTrue(result['conflicts'])

 def test_explicit_split_not_undone(self):
  rows=[obj('a',[0,0,10,10],sources=['s']),obj('b',[0,0,10,10],sources=['s'])]
  result=resolve(rows,[{'id':'p','a':'a','b':'b'}],[{'pair_id':'p','relation':'same','reason':'same'}])
  self.assertEqual(len(result['objects']),2)

 def test_network_member_moves_source_not_pixels(self):
  rows=[obj('n',[0,0,100,100],'dense_network'),obj('e',[20,30,60,40]),obj('out',[99,30,150,40])]
  pairs=[{'id':'p','a':'e','b':'n'},{'id':'q','a':'out','b':'n'}]
  ds=[{'pair_id':'p','relation':'a_in_b','reason':'internal edge'},{'pair_id':'q','relation':'distinct','reason':'external output'}]
  result=resolve(rows,pairs,ds);self.assertEqual(len(result['objects']),2)
  owner=next(o for o in result['objects'] if o['id']=='n');self.assertEqual(owner['bbox_native'],[0,0,100,100]);self.assertEqual(set(owner['source_ids']),{'n','e'})
  self.assertEqual(result['absorbed_members'][0]['member']['bbox_native'],[20,30,60,40])

 def test_bad_network_owner_retains_member(self):
  for kind,extent in [('dense_region','complete'),('dense_network','partial')]:
   rows=[obj('n',[0,0,100,100],kind),obj('e',[20,30,60,40])];rows[0]['extent']=extent
   result=resolve(rows,[{'id':'p','a':'e','b':'n'}],[{'pair_id':'p','relation':'a_in_b','reason':'member'}])
   self.assertEqual(len(result['objects']),2);self.assertTrue(result['conflicts'])

 def test_failed_relation_keeps_all(self):
  rows=[obj('a',[0,0,10,10]),obj('b',[1,0,11,10])]
  r=resolve(rows,[{'id':'p','a':'a','b':'b'}],[]);self.assertEqual(len(r['objects']),2);self.assertTrue(all(o['uncertain'] for o in r['objects']))

 def test_ledger_requires_every_observation(self):
  with self.assertRaises(AssertionError):ledger([obj('a',[0,0,10,10])],{'objects':[]},[])

 def test_mask_projection_identity_and_global_to_crop(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'m.png';a=np.zeros((40,50),dtype='uint8');a[10:17,20:24]=255;Image.fromarray(a).save(p)
   v=Crop((50,40),(0,0,50,40),(50,40));m={'mask_path':str(p),'mask_sha256':sha256(p),'frame':v.metadata(v.global_size)}
   np.testing.assert_array_equal(project(m,v),a>0)
   target=Crop((50,40),(10,5,40,25),(60,40));expected=np.asarray(Image.fromarray(a).crop((10,5,40,25)).resize((60,40),Image.Resampling.NEAREST))>0
   np.testing.assert_array_equal(project(m,target),expected)

 def test_sam_bbox_uses_pixels_and_coordinate_mapping(self):
  v=Crop((100,100),(10,20,50,60),(80,80));m=np.zeros((80,80),bool);m[20:30,30:40]=True
  out=materialize(answer(masks=['M1']),[obj('a',[10,20,30,40])],[{'alias':'M1','id':'S1','projected':m}],v,'b')[0]
  self.assertEqual(out['bbox_target'],[28,18,41,31]);self.assertEqual(out['bbox_native'],v.to_global([28,18])+v.to_global([41,31]))

 def test_default_thinking_113k_payload(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'a.png';Image.new('RGB',(8,8),'white').save(p)
   cfg={'model':'glm-5.3-flash','temperature':.1,'max_tokens':113000,'thinking':{'type':'enabled'}}
   payload=build_payload(cfg,'system',{},p);self.assertEqual(payload['max_tokens'],113000);self.assertNotIn('reasoning_effort',payload)

 def test_end_to_end_offline_discovery_refine_ownership_and_expansion(self):
  for partial in [False,True]:
   with self.subTest(partial=partial),tempfile.TemporaryDirectory() as temp:
    root=Path(temp);cid='fixture';inp=root/'inputs'/cid;inp.mkdir(parents=True)
    image=Image.new('RGB',(100,100),'white');image.save(inp/'source.png');views=[]
    for vid,scale in [('G',1),('T',2)]:
     view=Crop((100,100),(0,0,100,100),(100*scale,100*scale));folder=inp/vid;folder.mkdir();view.image(image).save(folder/'target.png')
     views.append({'view_id':vid,'path':str((folder/'target.png').relative_to(root)),'transform':view.metadata((100,100))})
    write(inp/'views.json',{'views':views});write(inp/'baseline.json',{'sam_cache':str(root/'unused')})
    cfg={'view_workers':2,'refine_workers':2,'simple_batch_max':8,'complex_batch_max':2,'context_candidate_max':8,'sam_evidence_max':32,
         'neighbor_gap_native_px':64,'simple_batch_span_native_px':480,'context_padding_native_px':48,'max_expansion_rounds':2,'pair_batch_max':6,'pair_object_max':10}
    class FakeModel:
     config=cfg
     def ask(self,path,q,images,validator):
      path.mkdir(parents=True,exist_ok=True)
      if 'focus' not in q:
       size=q['target_size_px'];s=size[0]/100
       a={'objects':[{'id':'A1','kind':'single','bbox':[(v+.5)*s-.5 for v in [20,28,60,33]],'locator':'visible arrow','extent':'partial' if partial else 'complete'}]}
      else:
       ids=[r['id'] for r in q['focus']];o=answer(source=ids[0])['objects'][0];o['source_ids']=ids;o['bbox']=q['focus'][0]['bbox'];o['extent']='partial' if partial and not q['task'].startswith('扩大') else 'complete';a={'objects':[o],'removed':[],'unresolved':[]}
      return validator(a)
    with patch('frontend.publish',lambda r:None),patch('frontend.get_sam',lambda *a,**kw:[]),patch('frontend.sam_only_observations',lambda m,o:([],[])):
     result=run_case(root,cid,FakeModel(),'scope')
    self.assertEqual(result['state'],'completed');self.assertEqual(result['output_identity_count'],1)
    self.assertEqual(len(result['source_ledger']),2);self.assertTrue(result['objects'][0]['geometry_ready'])

if __name__=='__main__':unittest.main()
