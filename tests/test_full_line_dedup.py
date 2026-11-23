"""Source-backed duplicate identity preserves parallel lines, heads and styles."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_recovery.handoff import duplicate_arrows
from full_recovery.handoff import duplicate_arrow_routes


class LineDedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'source.png'
        self.scene={'parts':{'edge':{'id':'edge','object_id':'network','status':'ok',
            'commands':[['M',20,50],['L',220,50]]}}}
        self.styles={'edge':{'stroke':'#666666','width_px':2,'dash':'solid'}}
        self.arrow={'id':'a','start':[20,55],'end':[220,55],'heads':['none','none'],
            'style':{'color':'#666666','width_px':2,'dash':{'preset':'solid'}}}
        im=Image.new('RGB',(250,100),'white');ImageDraw.Draw(im).line([(20,52),(220,52)],fill='#666666',width=2);im.save(self.path)

    def test_single_source_stroke_resolves_independent_fit_offsets(self):
        drop,audit=duplicate_arrows(self.scene,self.styles,[self.arrow],self.path)
        self.assertEqual(drop,['a']);self.assertGreaterEqual(audit[0]['single_source_stroke_fraction'],.8)

    def test_close_parallel_original_strokes_survive(self):
        im=Image.new('RGB',(250,100),'white');d=ImageDraw.Draw(im)
        for y in (50,55):d.line([(20,y),(220,y)],fill='#666666',width=2)
        im.save(self.path)
        self.assertEqual(duplicate_arrows(self.scene,self.styles,[self.arrow],self.path)[0],[])
        im=Image.new('RGB',(250,100),'white');d=ImageDraw.Draw(im)
        for y in (50,53):d.line([(20,y),(220,y)],fill='#666666',width=1)
        im.save(self.path);self.arrow.update(start=[20,53],end=[220,53])
        self.assertEqual(duplicate_arrows(self.scene,self.styles,[self.arrow],self.path)[0],[])

    def test_pattern_width_and_missing_endpoint_heads_block_dedup(self):
        for edit in ('dash','width','head','dot'):
            arrow=copy.deepcopy(self.arrow)
            if edit=='dash':arrow['style']['dash']={'preset':'dot'}
            if edit=='width':arrow['style']['width_px']=8
            if edit=='head':arrow['heads']=['none','triangle']
            if edit=='dot':arrow['style']['end']={'type':'oval'}
            self.assertEqual(duplicate_arrows(self.scene,self.styles,[arrow],self.path)[0],[],edit)

    def test_partial_or_crossing_route_is_not_a_duplicate(self):
        for start,end in (([20,55],[120,55]),([110,20],[110,80])):
            arrow={**self.arrow,'start':start,'end':end}
            self.assertEqual(duplicate_arrows(self.scene,self.styles,[arrow],self.path)[0],[])

    def test_same_branch_exact_duplicate_needs_one_source_stroke(self):
        arrows=[copy.deepcopy(self.arrow),copy.deepcopy(self.arrow)]
        arrows[0]['id']='a';arrows[1]['id']='b'
        drop,audit=duplicate_arrow_routes(arrows,self.path)
        self.assertEqual(drop,['b'])
        self.assertEqual(audit[0]['duplicate_of'],'a')
        im=Image.new('RGB',(250,100),'white');draw=ImageDraw.Draw(im)
        draw.line([(20,50),(220,50)],fill='#666666',width=2)
        draw.line([(20,55),(220,55)],fill='#666666',width=2);im.save(self.path)
        self.assertEqual(duplicate_arrow_routes(arrows,self.path)[0],[])

    def test_same_branch_route_with_different_heads_is_preserved(self):
        a=copy.deepcopy(self.arrow);a['id']='a'
        b=copy.deepcopy(self.arrow);b['id']='b';b['heads']=['none','triangle']
        self.assertEqual(duplicate_arrow_routes([a,b],self.path)[0],[])

    def test_dense_network_uses_unique_node_pair_and_preserves_parallel_multiedges(self):
        for ident,x in (('n1',20),('n2',220)):
            self.scene['parts'][ident]={'kind':'ellipse','status':'ok','object_id':'network',
                'bbox_2d':[x-10,40,x+10,60]}
        self.arrow.update(start=[20,58],end=[220,58])
        im=Image.open(self.path);draw=ImageDraw.Draw(im)
        for x in range(25,220,2):draw.line([(x,35),(x,75)],fill='#666666',width=1)
        im.save(self.path)
        drop,audit=duplicate_arrows(self.scene,self.styles,[self.arrow],self.path)
        self.assertEqual(drop,['a']);self.assertIn('unique_node_pair',audit[0]['reason'])
        self.scene['parts']['parallel']={**self.scene['parts']['edge'],'commands':[['M',20,51],['L',220,51]]}
        self.styles['parallel']=self.styles['edge']
        self.assertEqual(duplicate_arrows(self.scene,self.styles,[self.arrow],self.path)[0],[])


if __name__=='__main__':unittest.main()
