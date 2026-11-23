import tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev,mask_vectors,semantic_geometry,group_recovery,mask_grouping,style_windows


class NoSkeleton(unittest.TestCase):
    def test_complex_polyline_preserves_model_bends_despite_mask_pull(self):
        mask=np.zeros((100,120),bool);mask[65:70,15:110]=True
        a=dict(nodes=[['a',10,20],['b',90,20],['c',90,80]],excluded=[],uncertain=[],routes=[
            dict(id='r',sources=['G1'],masks=['M1'],kind='polyline',proposal=None,nodes=['a','b','c'],geometry='mask',heads=['none','triangle'])])
        c=dict(id='M1',mask=mask,path='mask')
        rows,_,_=mask_vectors.physical_rows(a,{},[c],'G1')
        self.assertEqual(rows[0]['start'],[10,20]);self.assertEqual(rows[0]['via'],[[90,20]])
        self.assertEqual(rows[0]['end'],[90,80]);self.assertEqual(rows[0]['coordinate_source'],'glm_keypoints')

    def test_model_curve_interpolation_does_not_bow_a_constant_x_span(self):
        from lib import fit
        p,curves=fit.interpolate_points([[80,857],[100,820],[102,560],[102,310]])
        np.testing.assert_allclose(np.asarray(curves[-1])[:,0],102,atol=1e-9)
        self.assertLessEqual(p[:,0].max(),102+1e-9)

    def test_identity_does_not_receive_invented_curve_across_disconnected_masks(self):
        from PIL import ImageDraw
        im=Image.new('L',(180,120));d=ImageDraw.Draw(im)
        d.line([(10,25),(80,25)],fill=255,width=5);d.line([(130,50),(130,90),(165,90)],fill=255,width=5)
        candidate=dict(id='M1',mask=np.asarray(im)>0,scores={'G1':.4,'G2':.3})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'input').mkdir()
            proposals,info=semantic_geometry.prepare(Image.new('RGB',im.size,'white'),[],[candidate],out)
        self.assertFalse(proposals);self.assertEqual(info[0]['components'],2)

    def test_model_selected_block_mask_needs_no_handwritten_vertices(self):
        im=Image.new('L',(100,80));ImageDraw.Draw(im).polygon([(10,30),(60,30),(60,20),(90,40),(60,60),(60,50),(10,50)],fill=255)
        candidate=dict(id='M1',mask=np.asarray(im)>0,path='mask')
        raw=dict(nodes=[],routes=[dict(id='r',sources=['G1'],masks=['M1'],kind='block_arrow',proposal=None,nodes=[],geometry='mask',heads=['none','triangle'])],excluded=[],uncertain=[])
        a=semantic_geometry.validate(raw,[100,80],['G1'],{},['M1'],[candidate])
        self.assertEqual(a['routes'][0]['semantic_heads'],['none','triangle'])
        self.assertEqual(a['routes'][0]['heads'],['none','none'])
        self.assertEqual(a,semantic_geometry.validate(a,[100,80],['G1'],{},['M1'],[candidate]))
        rows,_,_=mask_vectors.physical_rows(a,{},[candidate],'G1')
        self.assertEqual(rows[0]['geometry_method'],'model_selected_mask_contour')

    def test_cached_proposal_expansion_is_idempotent(self):
        p=dict(kind='straight',start=[10,30],via=[],end=[90,30],masks=['M1'])
        raw=dict(nodes=[],routes=[dict(id='r',sources=['G1'],masks=['M1'],kind='straight',proposal='D1',nodes=[],geometry='mask',heads=['none','triangle'],line_style='solid')],excluded=[],uncertain=[])
        a=semantic_geometry.validate(raw,[100,80],['G1'],{'D1':p},['M1'])
        b=semantic_geometry.validate(a,[100,80],['G1'],{'D1':p},['M1'])
        self.assertEqual(a,b)

    def test_model_can_change_a_curve_proposal_to_straight_without_retry(self):
        p=dict(kind='curve',start=[10,30],via=[[50,31]],end=[90,30],masks=['M1'])
        a=dict(nodes=[],routes=[dict(id='r',sources=['G1'],masks=['M1'],kind='straight',proposal='D1',nodes=[],geometry='mask',heads=['none','triangle'],line_style='solid')],excluded=[],uncertain=[])
        a=semantic_geometry.validate(a,[100,80],['G1'],{'D1':p},['M1'])
        self.assertEqual(a['routes'][0]['kind'],'straight');self.assertIsNone(a['routes'][0]['proposal'])
        self.assertEqual(len(a['routes'][0]['nodes']),2)

    def test_shared_nodes_survive_with_two_different_mask_measurements(self):
        a=dict(nodes=[['a',10,30],['j',60,30],['b',100,30],['c',60,85]],excluded=[],uncertain=[],routes=[
            dict(id='r1',sources=['G1'],masks=[],kind='straight',proposal=None,nodes=['a','j','b'],geometry='points',heads=['none','triangle'],line_style='solid'),
            dict(id='r2',sources=['G1'],masks=[],kind='straight',proposal=None,nodes=['j','c'],geometry='points',heads=['none','triangle'],line_style='solid')])
        a=semantic_geometry.validate(a,[120,100],['G1'],{},[])
        rows,_,_=mask_vectors.physical_rows(a,{},[],'G1')
        self.assertEqual(rows[0]['via'][0],rows[1]['start'])
        self.assertEqual(rows[0]['parent_id'],rows[1]['parent_id'])

    def test_curve_proposal_never_calls_skeletonize(self):
        image=Image.new('L',(180,250));ImageDraw.Draw(image).arc((10,10,160,240),-80,80,fill=255,width=6)
        with patch('skimage.morphology.skeletonize',side_effect=AssertionError('Forbidden skeletonization')):
            r=mask_vectors.propose(np.array(image)>0)
        self.assertEqual(r['kind'],'curve')
        for a,b in zip(r['beziers'],r['beziers'][1:]):
            u=np.subtract(a[3],a[2]);v=np.subtract(b[1],b[0])
            self.assertGreater(u@v/np.linalg.norm(u)/np.linalg.norm(v),.9999)

    def test_mask_refinement_recovers_an_underestimated_free_endpoint(self):
        image=Image.new('L',(160,140));ImageDraw.Draw(image).line([(30,110),(30,30),(120,30),(120,100)],fill=255,width=5)
        r=mask_vectors.fit_route(np.asarray(image)>0,'polyline',[[30,80],[30,30],[120,30],[120,100]],['none','triangle'],'mask')
        self.assertLess(abs(r['start'][0]-30),1)
        self.assertGreater(r['start'][1],108)

    def test_observation_groups_need_no_skeleton(self):
        m=np.zeros((40,100),bool);m[19:22,10:90]=True;s,b=ev.support(m)
        rows=[dict(id='G1',box=b,area=int(m.sum()),_support=s)]
        with patch('skimage.morphology.skeletonize',side_effect=AssertionError('Forbidden skeletonization')):
            groups,_=mask_grouping.group(rows,[100,40],dict(containment=.8,endpoint_gap_at_2048=12))
        self.assertEqual(len(groups),1)

    def test_empty_mask_does_not_turn_orange_contour_black(self):
        image=np.full((80,100,3),255,np.uint8);image[20:60,20:80]=[237,127,49]
        row=dict(id='r',kind='block_arrow',start=[25,25],via=[[75,25],[75,55]],end=[25,55])
        s=style_windows.recover(image,np.zeros((80,100),bool),row)
        self.assertEqual(s['color'],'#ed7f31');self.assertGreater(s['evidence']['fill_samples'],0)

    def test_local_dashed_leg_preserves_whole_polyline_geometry(self):
        import copy
        im=Image.new('RGB',(300,150),'white');d=ImageDraw.Draw(im)
        for x in range(30,270,24):d.line([(x,50),(x+11,50)],fill='black',width=4)
        # The lower leg is deliberately missing: style can use the upper leg.
        row=dict(id='r',kind='polyline',start=[30,50],via=[[270,50]],end=[270,120],beziers=[],
                 heads=['none','none'],start_head=False,end_head=False,line_style='dashed',width_estimate=4)
        before=copy.deepcopy(row);image=np.asarray(im);s=style_windows.recover(image,np.any(image<200,axis=2),row)
        self.assertEqual(row,before);self.assertEqual(s['dash']['type'],'dashed')
        self.assertIn('mark_px',s['dash']);self.assertIn('style_window',s['evidence'])

    def test_missing_group_uses_same_identity_geometry_and_style_entry(self):
        im=Image.new('RGB',(160,80),'white');d=ImageDraw.Draw(im)
        for x in range(20,140,20):d.line([(x,40),(x+10,40)],fill='black',width=4)
        mask=np.any(np.asarray(im)<200,axis=2)
        def prepare(image,group,out,cfg,box):
            (out/'input').mkdir(parents=True,exist_ok=True)
            for n in ['clean','context','global_hints']:image.save(out/'input'/f'{n}.png')
            return image,ev.identity_view(image.size)
        c=dict(id='M1',mask=mask,path='test-mask',category='line',scores={})
        class Model:
            def ask(self,folder,prompt,images,validator):
                return validator(dict(nodes=[],routes=[dict(id='r1',sources=['O1'],masks=['M1'],kind='straight',proposal='D1',nodes=[],geometry='mask',heads=['none','none'],line_style='dashed')],excluded=[],uncertain=[]))
        with tempfile.TemporaryDirectory() as tmp,patch.object(group_recovery.local_sampling,'prepare',prepare),patch.object(group_recovery.local_sampling,'sample',return_value=([c],[c])):
            c['path']=str(Path(tmp)/'native_mask.png')
            Image.fromarray(mask.astype('uint8')*255).save(c['path'])
            result=group_recovery.group_task(im,[],[],{},'O1',Path(tmp),{},Model(),dict(cpu=threading.Semaphore(1),sam=threading.Semaphore(1)),synthetic_source='O1')
        self.assertEqual(result['rows'][0]['style']['dash']['type'],'dashed')


if __name__=='__main__':unittest.main()
