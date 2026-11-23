import copy
import unittest
import numpy as np
from PIL import Image,ImageDraw
import curve_sampling


class CurveSampling(unittest.TestCase):
    def make_arrow(self,angle=0,scale=1):
        theta=np.linspace(-1.4,-.1,200)
        p=np.c_[130*np.cos(theta),130*np.sin(theta)]
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        p=(p@rot.T+180)*scale
        d=p[-1]-p[-2];d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
        im=Image.new('L',(round(360*scale),round(360*scale)))
        draw=ImageDraw.Draw(im);draw.line([tuple(v) for v in p],fill=255,width=round(6*scale))
        tip=p[-1];base=tip-d*23*scale
        draw.polygon([tuple(tip),tuple(base+n*11*scale),tuple(base-n*11*scale)],fill=255)
        seed=p.copy();seed[-25:]+=np.sin(np.linspace(0,np.pi,25))[:,None]*n*4*scale
        return np.asarray(im)>0,dict(kind='curve',start=seed[0].tolist(),via=seed[1:-1].tolist(),end=seed[-1].tolist(),
            heads=['none','triangle'],width_estimate=6*scale),d

    def test_fixed_endpoints_and_tip_direction_across_rotation_and_scale(self):
        for angle,scale in [(0,1),(.7,1),(1.7,2)]:
            with self.subTest(angle=angle,scale=scale):
                mask,row,true_direction=self.make_arrow(angle,scale);original=copy.deepcopy(row)
                result=curve_sampling.refine(mask,row)
                self.assertEqual(result['start'],row['start']);self.assertEqual(result['end'],row['end'])
                self.assertEqual(row,original)
                c=np.asarray(result['beziers'][-1]);d=c[3]-c[2];d/=np.linalg.norm(d)
                self.assertGreater(d@true_direction,np.cos(np.deg2rad(15)))
                for a,b in zip(result['beziers'],result['beziers'][1:]):
                    np.testing.assert_allclose(a[3],b[0],atol=1e-6)
                    u=np.subtract(a[3],a[2]);v=np.subtract(b[1],b[0])
                    self.assertGreater(u@v/np.linalg.norm(u)/np.linalg.norm(v),.999)

    def test_remote_specks_do_not_drag_endpoints(self):
        mask,row,_=self.make_arrow();base=curve_sampling.refine(mask,row)
        noisy=mask.copy();noisy[2,2]=True;noisy[355,355]=True
        result=curve_sampling.refine(noisy,row)
        self.assertEqual(result['start'],row['start']);self.assertEqual(result['end'],row['end'])
        np.testing.assert_allclose(result['beziers'],base['beziers'])

    def test_biased_interior_guides_do_not_make_a_hook_before_the_tip(self):
        # Endpoints are correct but interior model guides deviate by several
        # shaft widths. Updating the measuring axis must recover from that bias.
        from scipy.spatial import cKDTree
        for angle,scale in [(0,1),(.7,1),(1.7,2)]:
            with self.subTest(angle=angle,scale=scale):
                mask,row,direction=self.make_arrow(angle,scale)
                theta=np.linspace(-1.4,-.1,200)
                rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                true=(np.c_[130*np.cos(theta),130*np.sin(theta)]@rot.T+180)*scale
                guide=true.copy();normal=np.c_[-np.gradient(true[:,1]),np.gradient(true[:,0])]
                normal/=np.linalg.norm(normal,axis=1,keepdims=True)
                guide+=30*scale*np.sin(np.linspace(0,np.pi,len(guide)))[:,None]*normal
                guide[[0,-1]]=true[[0,-1]]
                row.update(start=guide[0].tolist(),via=guide[1:-1].tolist(),end=guide[-1].tolist())
                result=curve_sampling.refine(mask,row)
                p=np.asarray([result['start'],*result['via'],result['end']])
                distances=cKDTree(true).query(p)[0]
                self.assertLess(np.percentile(distances,95),3*scale)
                tail=np.diff(p[-max(8,len(p)//5):],axis=0)
                self.assertTrue((tail@direction>0).all())
                np.testing.assert_array_equal(p[[0,-1]],true[[0,-1]])

    def test_no_heads_preserves_both_endpoints(self):
        mask,row,_=self.make_arrow();row['heads']=['none','none']
        result=curve_sampling.refine(mask,row)
        self.assertEqual(result['start'],row['start']);self.assertEqual(result['end'],row['end'])

    def test_short_wide_curve_keeps_its_bend_instead_of_becoming_a_diagonal(self):
        from scipy.spatial import cKDTree
        theta=np.linspace(np.pi/2,0,80)
        arc=np.c_[160+40*np.cos(theta),170+40*np.sin(theta)]
        p=np.vstack([np.c_[np.linspace(70,160,70),np.full(70,210)],arc,
                     np.c_[np.full(60,200),np.linspace(170,90,60)]])
        for scale in [1,2]:
            with self.subTest(scale=scale):
                q=p*scale;mask=Image.new('L',(300*scale,300*scale));draw=ImageDraw.Draw(mask)
                draw.line([tuple(a) for a in q],fill=255,width=12*scale)
                draw.polygon([(200*scale,90*scale),(178*scale,135*scale),(222*scale,135*scale)],fill=255)
                row=dict(kind='curve',start=q[0].tolist(),via=q[1:-1].tolist(),end=q[-1].tolist(),
                         heads=['none','triangle'],width_estimate=12*scale)
                result=curve_sampling.refine(np.asarray(mask)>0,row)
                measured=np.asarray([result['start'],*result['via'],result['end']])
                distance=cKDTree(q).query(measured)[0]
                self.assertLess(np.percentile(distance,95),5*scale)
                c=np.asarray(result['beziers'][-1]);direction=c[-1]-c[-2];direction/=np.linalg.norm(direction)
                self.assertGreater(direction@np.array([0,-1]),np.cos(np.deg2rad(15)))

    def test_no_evidence_reports_failure(self):
        mask,row,_=self.make_arrow()
        with self.assertRaises(ValueError):curve_sampling.refine(np.zeros_like(mask),row)

    def test_dashed_mark_parameters_do_not_make_large_spline_excursions(self):
        mask,row,direction=self.make_arrow()
        p=np.array([row['start'],*row['via'],row['end']]);im=Image.new('L',(360,360));draw=ImageDraw.Draw(im)
        for i in range(0,len(p)-25,22):draw.line([tuple(v) for v in p[i:i+10]],fill=255,width=6)
        tip=p[-1];n=np.array([-direction[1],direction[0]]);base=tip-direction*23
        draw.polygon([tuple(tip),tuple(base+n*11),tuple(base-n*11)],fill=255)
        row['line_style']='dashed';result=curve_sampling.refine(np.asarray(im)>0,row)
        q=np.asarray([result['start'],*result['via'],result['end']])
        self.assertTrue((q>=p.min(axis=0)-15).all());self.assertTrue((q<=p.max(axis=0)+15).all())
        self.assertEqual(result['start'],row['start']);self.assertEqual(result['end'],row['end'])


if __name__=='__main__':unittest.main()
