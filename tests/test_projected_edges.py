"""Independent source geometry tests for shared extrusion direction fitting."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.fitting.adapter import fit_object
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.projected_edges import refine,face_conflicts


def solid(depth):
    root=np.array([120.,95.]);a=np.array([-50.,0.]);b=np.array([0.,100.]);c=np.array(depth,float)
    points={'r':root,'a':root+a,'b':root+b,'c':root+c,'ab':root+a+b,'ac':root+a+c,'bc':root+b+c}
    def path(ident,ids):return {'id':ident,'kind':'path','closed':True,'segments':[
        {'kind':'line','point_ids':[x,y]} for x,y in zip(ids,ids[1:]+ids[:1])]}
    return {'status':'ok','object_id':'object','reason':'','uncertain':[],
        'points':[{'id':k,'xy':v.tolist()} for k,v in points.items()],
        'parts':[{'id':'front','kind':'rectangle','point_ids':['a','r','b','ab']},
                 path('roof',['a','r','c','ac']),path('side',['r','c','bc','b'])]}


class ProjectedEdges(unittest.TestCase):
    def test_same_solid_face_fold_is_reported_to_existing_critic(self):
        from shape_integration.fitting.critic import review_prompt
        local,job,source=self.fixture((35,35));job.update(proposed_type='projected_3d',label='solid')
        self.assertTrue(face_conflicts(local,job))
        prompt=review_prompt(job,local)
        self.assertIn('projected_face_conflicts',prompt)
        for row in prompt['projected_face_conflicts']['overlapping_interiors']:
            self.assertTrue(all(k in {'F1','F2','F3'} for k in row['parts']))
        normal,normal_job,_=self.fixture((55,-55))
        self.assertEqual(face_conflicts(normal,normal_job),[])

    def fixture(self,depth=(30,-71.8)):
        truth=solid((55,-55));table={p['id']:p['xy'] for p in truth['points']}
        source=Image.new('RGB',(300,240),'white');draw=ImageDraw.Draw(source)
        for ids,colour in [(['a','r','c','ac'],'#5050C9'),(['r','c','bc','b'],'#A6A6CF'),(['a','r','b','ab'],'#4040EE')]:
            q=[tuple(table[k]) for k in ids];draw.polygon(q,fill=colour);draw.line(q+[q[0]],fill='black',width=1)
        raw=solid(depth);frame=FitFrame((300,240),(300,240),(300,240),(0,0,300,240))
        local=fit_object(raw,frame,object_id='object',namespace='test')
        job={'candidate':{'kind':'projected_3d'},'frame':{**frame.metadata(),'source_crop_half_open':[0,0,300,240]}}
        return local,job,source

    def test_source_direction_corrects_shared_depth_and_preserves_anchor_and_observation(self):
        local,job,source=self.fixture();raw=copy.deepcopy(local['observation']);front=copy.deepcopy(local['parts'][0])
        audit=refine(local,job,source)
        self.assertTrue(audit['applied'],audit)
        self.assertLess(audit['loss_after_source_px'],.85*audit['loss_before_source_px'])
        self.assertEqual(local['observation'],raw)
        self.assertEqual(local['parts'][0]['commands'],front['commands'])
        self.assertEqual(set(p['id'] for p in local['parts']),{'front','roof','side'})
        self.assertLess(local['shared_point_diagnostics']['shared_points'][0]['max_separation_px'],1e-6)

    def test_matching_source_and_empty_source_do_not_change_geometry(self):
        for depth,blank in [((55,-55),False),((30,-71.8),True)]:
            local,job,source=self.fixture(depth);before=copy.deepcopy(local)
            if blank:source=Image.new('RGB',source.size,'white')
            audit=refine(local,job,source)
            self.assertFalse(audit['applied'],audit);self.assertEqual(local,before)

    def test_partial_edit_cannot_move_preserved_shared_parts(self):
        local,job,source=self.fixture();before=copy.deepcopy(local)
        job['revision_context']={'preserved_ids':['roof']}
        self.assertFalse(refine(local,job,source)['applied']);self.assertEqual(local,before)


if __name__=='__main__':unittest.main()
