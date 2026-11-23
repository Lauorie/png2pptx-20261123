import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pipeline,straight_mask,pixels
from lib import style


def make_line(dashed=False,angle=0):
    im=Image.new('RGB',(360,160),'white');d=ImageDraw.Draw(im)
    for x in (range(35,315,40) if dashed else [35]):
        d.line([(x,80),(min(x+(24 if dashed else 275),310),80)],fill='red',width=6)
    d.polygon([(322,80),(302,69),(302,91)],fill='red')
    im=im.rotate(angle,resample=Image.Resampling.NEAREST,fillcolor='white')
    return im,np.any(np.asarray(im)<200,axis=2)


class StraightDash(unittest.TestCase):
    def test_whole_dashed_mask_is_one_line_with_phase(self):
        for angle in [0,13,-19]:
            image,mask=make_line(True,angle)
            with tempfile.TemporaryDirectory() as tmp:
                data,support=pixels.draft([dict(id='M1',mask=mask)],image.size,Path(tmp))
            self.assertEqual(len(data['paths']),1)
            self.assertEqual(data['diagnostics']['method'],'direct_straight_mask')
            self.assertTrue(np.array_equal(support,mask))
            row=data['paths'][0];row.update(end_head=True,end_head_type='triangle')
            s=style.recover(np.asarray(image),mask,row)
            self.assertEqual(s['dash']['type'],'dashed')
            self.assertAlmostEqual(s['width_px'],6,delta=1)
            self.assertAlmostEqual(s['dash']['mark_px'],25,delta=2)
            self.assertAlmostEqual(s['dash']['gap_px'],15,delta=2)
            self.assertIn('phase_px',s['dash'])

    def test_solid_and_single_occlusion_are_not_periodic(self):
        image,mask=make_line()
        r=straight_mask.estimate(mask);r.update(end_head=True,end_head_type='triangle')
        self.assertEqual(style.recover(np.asarray(image),mask,r)['dash']['type'],'solid')
        s=np.arange(200,dtype=float);signal=np.ones(200,bool);signal[70:95]=False
        self.assertNotIn(style.dash_fit(s,signal,np.ones(200,bool),5)['type'],('dashed','dotted'))

    def test_complete_edge_runs_are_retained(self):
        s=np.arange(160,dtype=float);signal=np.zeros(160,bool)
        for x in [10,60,110]:signal[x:x+25]=True
        result=style.dash_fit(s,signal,np.ones(160,bool),5)
        self.assertEqual(result['complete_marks'],3)
        self.assertEqual(result['type'],'dashed')
        self.assertEqual(result['mark_px'],25)

    def test_segmentation_gaps_do_not_make_a_solid_arrow_dashed(self):
        image,mask=make_line();row=straight_mask.estimate(mask)
        row.update(end_head=True,end_head_type='triangle')
        for x in range(50,290,15):mask[:,x:x+4]=False
        result=style.recover(np.asarray(image),mask,row)
        self.assertEqual(result['dash']['type'],'solid')

    def test_few_periodic_defects_cannot_describe_an_entire_solid_shaft(self):
        s=np.arange(400,dtype=float);signal=np.ones(400,bool)
        for x in [10,22,34,46]:signal[x:x+3]=False
        self.assertEqual(style.dash_fit(s,signal,np.ones(400,bool),35)['type'],'solid')

    def test_filled_mask_does_not_make_a_dashed_arrow_solid(self):
        image,_=make_line(True);_,mask=make_line(False)
        row=straight_mask.estimate(mask);row.update(end_head=True,end_head_type='triangle')
        self.assertEqual(style.recover(np.asarray(image),mask,row)['dash']['type'],'dashed')

    def test_branches_parallel_lines_and_curve_are_not_flattened(self):
        for kind in ('branch','parallel','curve'):
            im=Image.new('L',(350,220));d=ImageDraw.Draw(im)
            if kind=='curve':d.arc((20,20,310,210),0,180,fill=255,width=6)
            else:
                d.line([(30,50),(310,50)],fill=255,width=6)
                d.line([(170,50),(170,185)] if kind=='branch' else [(30,150),(310,150)],fill=255,width=6)
            self.assertIsNone(straight_mask.estimate(np.asarray(im)>0),kind)

    def test_one_speck_does_not_extend_line(self):
        _,mask=make_line(True);mask[0,0]=True
        row=straight_mask.estimate(mask)
        self.assertIsNotNone(row)
        self.assertLess(abs(row['start'][1]-80),1)
        self.assertGreater(row['start'][0],30)

    def test_reversed_fragments_keep_spatial_head_and_shared_network(self):
        _,mask=make_line(True)
        base=dict(kind='straight',via=[],coordinate_source='local_sam_mask',identity_index=1,
            shared_nodes=[],globals=['G1'])
        rows=[dict(base,id='E1',start=[100,80],end=[35,80],heads=['none','none'],nodes=['N1','N2'],node_ids=['N1','N2']),
              dict(base,id='E2',start=[120,80],end=[322,80],heads=['none','triangle'],nodes=['N3','N4'],node_ids=['N3','N4'])]
        result=straight_mask.refit_confirmed(rows,[dict(id='M1',mask=mask)])
        self.assertEqual(len(result),1);self.assertEqual(result[0]['heads'],['none','triangle'])
        self.assertEqual(result[0]['identity_index'],1)
        shared=copy.deepcopy(rows);shared[0]['shared_nodes']=['N1']
        self.assertEqual(straight_mask.refit_confirmed(shared,[dict(id='M1',mask=mask)]),shared)
        proposal={'paths':[{'id':x} for x in ['E1','E2','E3']]}
        self.assertEqual(straight_mask.refit_confirmed(rows,[dict(id='M1',mask=mask)],proposal),rows)


if __name__=='__main__':unittest.main()
