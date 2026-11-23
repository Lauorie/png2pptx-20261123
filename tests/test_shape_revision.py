"""Regression cases for observed geometry, paint and stage-ownership failures."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from shapely.geometry import GeometryCollection

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'vendor'),str(ROOT/'tests')]
from test_fitting import rr_points
from shape_integration.fitting.primitives import fit_boundary,FitError
from shape_integration.styling import occlusion,options
from native_qwen.element_protocol import validator
from png2pptx.images.coverage import check_regions
from full_recovery.handoff import duplicate_arrows,duplicate_operator_text,duplicate_bracket_text
from shape_integration.sam_guidance import GuidedSamEvidence
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.observation import source_clipping


def path(commands,kind='path',owner='symbol'):
    points=np.array([c[1:3] for c in commands if c[0] in ('M','L')])
    return {'status':'ok','kind':kind,'object_id':owner,'commands':commands,
            'bbox_2d':[*points.min(0),*points.max(0)]}


class ShapeRevisionTests(unittest.TestCase):
    def test_delimiter_reconciliation_preserves_nested_and_embedded_brackets(self):
        from PIL import ImageDraw
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.png';image=Image.new('RGB',(160,100),'white');draw=ImageDraw.Draw(image)
            commands=[['M',43,20],['L',34,20],['L',34,80],['L',43,80]]
            draw.line([tuple(c[1:]) for c in commands],fill='black',width=3);image.save(source)
            part=path(commands);scene={'parts':{'bracket':part}};styles={'bracket':{'fill':None,'stroke':'#000000','width_px':3}}
            rows=[{'id':'delimiter','text':'[','bbox_2d':[12,15,32,85],'color':'#000000'},
                  {'id':'formula','text':'[x]','latex':'[x]','bbox_2d':[30,20,70,80],'color':'#000000'}]
            drop,_=duplicate_bracket_text(scene,styles,rows,source)
            self.assertEqual(drop,['delimiter'])
            draw.line([(25,20),(16,20),(16,80),(25,80)],fill='black',width=3);image.save(source)
            drop,_=duplicate_bracket_text(scene,styles,rows,source)
            self.assertEqual(drop,[])

    def test_context_exception_only_at_original_source_edges(self):
        frame={'original_image_size_px':[962,749], 'source_crop_half_open':[208,558,962,749]}
        self.assertEqual(source_clipping({'frame':frame})['image_edges'],['right','bottom'])
        frame['source_crop_half_open']=[208,558,960,748]
        self.assertIsNone(source_clipping({'frame':frame}))
        self.assertIsNone(source_clipping({'frame':{}}))

    def test_loose_sam_container_match_keeps_nested_target_identity(self):
        target={'id':'inner','label':'nested frame','kind':'rectangle','bbox_2d':[70,40,110,80]}
        group={'id':'support','representative':{'bbox':[20,30,120,90],'score':.9,'category':'rectangle'}}
        evidence=GuidedSamEvidence([group])
        with TemporaryDirectory() as root:
            job=prepare_candidate(Image.new('RGB',(200,120),'white'),target,(200,120),
                                  Path(root)/'candidate',spatial_evidence=evidence)
            frame=FitFrame.from_metadata(job['frame'])
            self.assertEqual(job['identity_bbox_local'],frame.to_local([70,40])+frame.to_local([110,80]))
            self.assertEqual(job['sam_support']['identity_source'],'candidate_sam_extent_disagrees')
            self.assertEqual(job['sam_support']['expanded_crop'],job['sam_support']['original_crop'])
            self.assertTrue(job['sam_support']['context_expansion_withheld'])
            _,images=evidence.observe({'task':'observe'},[],job,Path(root))
            self.assertEqual(images,[])
            self.assertLess(job['frame']['source_crop_half_open'][0],70)
        group['representative']['bbox']=[69,39,111,81]
        _,support=evidence.prepare(target,[65,35,116,86],(200,120))
        self.assertEqual(support['identity_bbox_original'],target['bbox_2d'])

    def test_matching_sam_content_region_does_not_remove_frame_header_from_target(self):
        target={'id':'frame','label':'container','kind':'rectangle','bbox_2d':[20,20,100,140]}
        group={'id':'body','representative':{'bbox':[20,45,100,140],'score':.95,'category':'rectangle'}}
        evidence=GuidedSamEvidence([group])
        with TemporaryDirectory() as root:
            job=prepare_candidate(Image.new('RGB',(160,180),'white'),target,(160,180),
                                  Path(root)/'candidate',spatial_evidence=evidence)
            frame=FitFrame.from_metadata(job['frame'])
            self.assertEqual(job['identity_bbox_local'],frame.to_local([20,20])+frame.to_local([100,140]))
            self.assertEqual(job['sam_support']['bbox_original'],[20,45,100,140])

    def test_capsule_coincident_roles_are_valid_but_other_duplicates_fail(self):
        for width,height in [(120,30),(30,120)]:
            p=np.round(rr_points(width,height,min(width,height)/2,angle=0),6)
            fit=fit_boundary('rounded_rectangle',p)
            self.assertAlmostEqual(fit['radius'],min(width,height)/2,delta=.1)
        p=rr_points();p[7]=p[0]
        with self.assertRaises(FitError):fit_boundary('rounded_rectangle',p)
        with self.assertRaises(FitError):fit_boundary('ellipse',np.zeros((8,2)))

    def test_precise_lower_score_sam_extent_outweighs_a_high_score_parent(self):
        target={'id':'inner','label':'nested frame','kind':'rectangle','bbox_2d':[70,40,110,80]}
        groups=[{'id':'parent','representative':{'bbox':[20,30,120,90],'score':.95,'category':'rectangle'}},
                {'id':'local','representative':{'bbox':[69,39,111,81],'score':.54,'category':'rectangle'}}]
        evidence=GuidedSamEvidence(groups);evidence.set_inventory([target])
        self.assertEqual(evidence.support(target)['id'],'local')
        groups[1]['representative']['bbox']=[90,50,120,80]
        self.assertEqual(evidence.support(target)['id'],'parent')

    def test_half_open_box_edge_is_bounded_normalization(self):
        raw={'objects':[{'id':'g','kind':'rectangle','label':'box','bbox_2d':[10,10,100,80]}],
             'uncertain':[],'more_pending':False}
        result=validator([100,80],'shape_bbox','absolute')(deepcopy(raw))
        self.assertEqual(result['objects'][0]['bbox_2d'],[10,10,99,79])
        raw['objects'][0]['bbox_2d'][2]=101
        with self.assertRaises(ValueError):validator([100,80],'shape_bbox','absolute')(raw)

    def test_missing_owner_list_only_normalized_for_empty_inventory(self):
        rows=[{'id':'r','bbox':[0,0,80,60],'reason':'inspect'}]
        check_regions(rows,[100,80],set(),set());self.assertEqual(rows[0]['target_ids'],[])
        with self.assertRaises(ValueError):check_regions([{'id':'r','bbox':[0,0,80,60]}],[100,80],{'I1'},{'I1'})

    def test_invalid_interior_path_does_not_hide_whole_component(self):
        panel=path([['M',0,0],['L',100,0],['L',100,100],['L',0,100],['Z']],owner='panel')
        body=path([['M',20,20],['L',40,20],['L',40,40],['L',20,40],['Z']])
        bad=path([['M',23,23],['L',37,37],['L',23,37],['L',37,23],['Z']])
        _,rel=occlusion.discover({'panel':panel,'body':body,'bad':bad},options({'styling':{}}))
        self.assertIn({'back':'panel','front':'symbol','origin':'contains'},rel['edges'])
        self.assertIn('bad',rel['geometry_issues'])

    def test_attached_line_behind_node_but_operator_stroke_stays_in_front(self):
        node=path([['M',20,20],['L',40,20],['L',40,40],['L',20,40],['Z']])
        edge=path([['M',30,30],['L',80,30]])
        plus=path([['M',25,30],['L',35,30]])
        table={'node':node,'edge':edge,'plus':plus};scene={'objects':{'symbol':{'part_keys':list(table)}}}
        geoms,_=occlusion.discover(table,options({'styling':{}}))
        styles={'node':{'fill':'#FFFFFF','stroke':'#000000'},'edge':{'fill':None,'stroke':'#000000'},'plus':{'fill':None,'stroke':'#000000'}}
        result=occlusion.order_internal_by_paint(scene,table,{'object_order':['symbol']},styles,geoms)
        order=result['part_orders']['symbol']
        self.assertLess(order.index('edge'),order.index('node'));self.assertLess(order.index('node'),order.index('plus'))

    def test_only_coincident_intrinsic_lines_are_deduplicated(self):
        p=path([['M',10,10],['L',90,10]])
        scene={'parts':{'edge':p}};styles={'edge':{'stroke':'#000000','fill':None,'width_px':1}}
        row={'id':'a','kind':'straight','start':[10,10],'end':[90,10],'heads':['none','none'],
             'style':{'width_px':1,'color':'#000000'}}
        other={**row,'id':'b','start':[10,16],'end':[90,16]}
        drop,_=duplicate_arrows(scene,styles,[row,other]);self.assertEqual(drop,['a'])
        # A real arrowhead cannot be deleted just because its shaft matches.
        drop,_=duplicate_arrows(scene,styles,[{**row,'heads':['none','triangle']}]);self.assertEqual(drop,[])

    def test_plain_label_inside_circle_is_not_duplicate_operator(self):
        h=path([['M',20,30],['L',40,30]]);v=path([['M',30,20],['L',30,40]])
        scene={'parts':{'h':h,'v':v}};styles={k:{'stroke':'#000000','fill':None} for k in scene['parts']}
        rows=[{'id':'plus','text':'+','bbox_2d':[20,20,40,40]},
              {'id':'word','text':'Add','bbox_2d':[20,20,40,40]}]
        drop,_=duplicate_operator_text(scene,styles,rows);self.assertEqual(drop,['plus'])


if __name__=='__main__':unittest.main()
