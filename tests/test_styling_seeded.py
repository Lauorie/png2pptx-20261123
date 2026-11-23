"""Semantic seed repair must preserve geometry/colours and issue no model calls."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import cairosvg
from PIL import Image

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;SRC=PROJECT/'src'
if '--src' in sys.argv:
    index=sys.argv.index('--src');SRC=Path(sys.argv[index+1]);del sys.argv[index:index+2]
sys.path[:0]=[str(SRC),str(PROJECT/'vendor')]
from shape_integration.common import read,save,sha,digest
from shape_integration.styling import options,occlusion
from shape_integration.styling.repair import run_case


class SemanticRepair(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'source.png'
        self.source.write_bytes(cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><rect width="40" height="40" fill="#DAE3F3"/><rect x="10" y="10" width="20" height="20" fill="#4472C4"/></svg>'))
        self.dot=json.dumps(['test','A','dot']);self.bg=json.dumps(['test','A','bg'])
        def part(name,box):
            x0,y0,x1,y1=box
            return {'id':name,'object_id':'A','kind':'rectangle','status':'ok','key':['test','A',name],
                'native_parameters_original':None,'commands':[['M',x0,y0],['L',x1,y0],['L',x1,y1],['L',x0,y1],['Z']]}
        self.scene={'case_id':'test','source_sha256':sha(self.source),'source_size':[40,40],
            'parts':{self.dot:part('dot',[9.5,9.5,29.5,29.5]),self.bg:part('bg',[0,0,40,40])},
            'objects':{'A':{'id':'A','state':'accepted_by_model','part_keys':[self.dot,self.bg]}}}
        self.seed=self.root/'seed/test/styling';self.seed.mkdir(parents=True)
        save(self.seed/'scene_source.json',self.scene)
        save(self.seed/'styles.json',{'state':'finished','parts':{self.dot:{'style':{'fill':'#4472C4','stroke':'#4472C4','width_px':6.,'dash':'solid'}},
            self.bg:{'style':{'fill':'#DAE3F3','stroke':None,'width_px':0.,'dash':'solid'}}}})
        _,relations=occlusion.discover(self.scene['parts'],options({}));relations=occlusion.order(self.scene,self.scene['parts'],relations)
        save(self.seed/'occlusion.json',relations);self.opt=options({'styling':{'semantic_seed_root':str(self.root/'seed')}})

    def test_keeps_semantics_restores_dot_and_never_calls_model(self):
        class ForbiddenModel:
            def __getattr__(self,name):raise AssertionError('Model touched: '+name)
        before=digest(self.scene);seed_hash=sha(self.seed/'styles.json')
        result=run_case(self.scene,self.source,self.root/'out',ForbiddenModel(),self.opt,None,threading.BoundedSemaphore(1))
        saved=read(self.root/'out/styles.json')['parts'];self.assertEqual(saved[self.dot]['style']['fill'],'#4472C4')
        self.assertIsNone(saved[self.dot]['style']['stroke']);self.assertEqual(saved[self.bg]['style']['fill'],'#DAE3F3')
        self.assertEqual(result['renders'],1);self.assertEqual(result['model_calls'],{})
        self.assertTrue(read(self.root/'out/export.json')['geometry_unchanged']);self.assertEqual(digest(self.scene),before)
        self.assertEqual(sha(self.seed/'styles.json'),seed_hash)
        self.assertLess(sum(abs(a-b) for a,b in zip(Image.open(self.root/'out/native.png').convert('RGB').getpixel((20,20)),(68,114,196))),15)

    def test_rejects_seed_from_different_geometry(self):
        wrong={**self.scene,'source_size':[41,40]};save(self.seed/'scene_source.json',wrong)
        with self.assertRaisesRegex(ValueError,'geometry differs'):
            run_case(self.scene,self.source,self.root/'bad',None,self.opt,None,threading.BoundedSemaphore(1))


if __name__=='__main__':unittest.main()
