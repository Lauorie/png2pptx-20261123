"""Behavior checks for identity-scoped line fitting, not skeleton fragments."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import instance_line_trial as trial


class WholeInstanceLine(unittest.TestCase):
    def test_separate_dotted_instances_remain_three_lines(self):
        candidates=[]
        for i,y in enumerate([30,75,120],1):
            m=Image.new('L',(240,150));d=ImageDraw.Draw(m)
            for x in range(20,220,15):d.ellipse((x-2,y-2,x+2,y+2),fill=255)
            candidates.append(dict(id=f'M{i}',selected=True,best_global=f'G{i}',mask=np.array(m)>0))
        dec=dict(targets=[dict(sources=[f'G{i}'],paths=[],filled_masks=[f'M{i}'],repair=None) for i in range(1,4)])
        rows,supports,skipped=trial.propose(dec,candidates,'G001')
        self.assertFalse(skipped);self.assertEqual(len(rows),3)
        for r,c in zip(rows,candidates):
            self.assertEqual(r['geometry_method'],'direct_straight_mask')
            self.assertTrue(np.array_equal(supports[r['id']],c['mask']))
            self.assertEqual(r['via'],[])

    def test_connections_only_follow_model_junctions(self):
        rows=[dict(id='T1',start=[0.,50.],end=[100.,50.],node_ids=['T1.start','T1.end']),
              dict(id='T2',start=[98.,53.],end=[98.,100.],node_ids=['T2.start','T2.end'])]
        decision=dict(targets=[dict(id=r['id'],status='keep',heads=['none','none'],line_style='unknown') for r in rows],junctions=[])
        separate,_=trial.materialize(rows,decision)
        self.assertNotEqual(separate[0]['end'],separate[1]['start'])
        decision['junctions']=[['T1.end','T2.start']]
        connected,j=trial.materialize(rows,trial.validate(decision,rows))
        self.assertEqual(connected[0]['end'],connected[1]['start'])
        self.assertEqual(connected[0]['end'],[98.,50.])

    def test_conflicting_fragment_heads_are_not_silently_reinterpreted(self):
        m=np.zeros((60,160),bool);m[28:33,10:150]=True
        candidates=[dict(id='M1',mask=m)]
        row=dict(id='T1',parent_id='G001.T1',source_masks=['M1'])
        base=dict(parent_id='G001.T1',identity_index=1,kind='straight',coordinate_source='local_sam_mask')
        old=[dict(base,id='E1',start=[10,30],end=[20,30],heads=['triangle','triangle']),
             dict(base,id='E2',start=[30,30],end=[150,30],heads=['none','none'])]
        answer=trial.offline_decision([row],old,candidates)
        self.assertEqual(answer['targets'][0]['heads'],['unknown','unknown'])


if __name__=='__main__':unittest.main()
