"""Retained geometry/export regressions and stable-context controller behavior."""
from pathlib import Path
import sys
import unittest
import math
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
SRC=PROJECT/'src'
if '--src' in sys.argv:
    index=sys.argv.index('--src');SRC=Path(sys.argv[index+1]);del sys.argv[index:index+2]
sys.path[:0]=[str(SRC),str(PROJECT/'vendor'),str(PROJECT/'tests')]
from shape_integration.styling.controller import run_case
import importlib.util
_spec=importlib.util.spec_from_file_location('styling_protocol10_fixture',PROJECT/'tests/fixtures/styling_protocol10.py')
old=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=old;_spec.loader.exec_module(old)
rectangle=old.rectangle;scene_of=old.scene_of;FakeRuntime=old.FakeRuntime;sha=old.sha;options=old.options

if not hasattr(old,'StyleTests'):old.StyleTests=old.ControllerContext
from shape_integration.common import read,digest
from shape_integration.styling.controller import run_case


def retained_base():
    class Retained(unittest.TestCase):pass
    for name in ['test_curve_bounds_use_extrema_not_control_points', 'test_containment_uses_geometry_not_only_bbox', 'test_shared_edge_is_not_partial_overlap', 'test_relation_crops_separate_disconnected_components', 'test_inner_object_above_outer_and_hollow_mask', 'test_cross_group_cycle_is_exposed', 'test_internal_overlap_and_containment_never_become_relations_or_masks', 'test_compound_candidates_are_unique_whole_component_pairs', 'test_whole_component_containment_and_union_holes', 'test_unknown_order_preserves_pixels_but_confirmed_foreground_masks_all_its_parts', 'test_gradient_contract_and_no_implicit_open_fill', 'test_gradient_estimator_and_solid_negative_control', 'test_masked_foreground_does_not_contaminate_fill', 'test_crop_follows_geometry_and_half_pixel_transform', 'test_native_gradient_export_and_geometry_invariance', 'test_uniform_crop_resampling_preserves_measured_width_and_position', 'test_pdf_crop_registration_at_both_zoom_directions_and_page_edges', 'test_pdf_crop_keeps_thin_vector_stroke_instead_of_enlarging_raster', 'test_pdf_crop_handles_blank_page_and_rejects_non_pdf_or_multiple_slides', 'test_grouping_protocol_rejects_duplicate_or_unknown_members', 'setUp', 'fake_export', 'test_controller_one_refine_and_preserves_unresolved_geometry', 'test_off_image_target_does_not_abort_other_styles']:
        setattr(Retained,name,getattr(old.StyleTests,name))
    return Retained

class ControllerContext(retained_base()):
    def test_disabled_critic_exports_selected_style_without_claiming_acceptance(self):
        def respond(tag, prompt):
            if tag.startswith(('critic', 'refine')):
                raise AssertionError('Disabled critic/refinement issued a request')
            return {'styles': [{'id': t['id'], 'fill': '#CC5533', 'stroke': None}
                               for t in prompt['targets']]}
        output = self.folder/'critic-disabled'
        opt = {**self.opt, 'critic_enabled': False}
        before = digest(self.scene)
        with ThreadPoolExecutor(max_workers=2) as pool, patch(
                'shape_integration.styling.controller.export', self.fake_export):
            result = run_case(self.scene, self.source, output, FakeRuntime(respond), opt,
                              pool, threading.BoundedSemaphore(1))
        part = read(output/'styles.json')['parts'][self.k]
        self.assertEqual(part['style']['fill'], '#CC5533')
        self.assertEqual(part['state'], 'styled_unreviewed')
        self.assertEqual(result['critic_accepted'], 0)
        self.assertEqual(result['model_calls']['critic'], 0)
        self.assertEqual(result['model_calls']['refine'], 0)
        self.assertEqual(digest(self.scene), before)

    def test_unpainted_occluding_face_enters_existing_semantic_refinement(self):
        entries=[rectangle('A','rear',80,70,80,60),rectangle('A','front',110,90,80,60)]
        scene=scene_of(entries,source_hash=sha(self.source));refined=[]
        def respond(tag,p):
            if tag.startswith('critic'):
                return {'checks':[({'id':t['id'],'verdict':'revise','affected_fields':['occlusion'],
                    'source_evidence':'CLEAN shows the front white face hiding the rear edges.',
                    'rendered_difference':'Rear edges show through the unfilled foreground.',
                    'suggestion':'Restore visible source occlusion.','in_front_of':['P0']}
                    if t['id']=='P1' and t['current_style']['fill'] is None
                    else {'id':t['id'],'verdict':'accept'}) for t in p['targets']]}
            if tag.startswith('refine'):
                refined.extend(t['id'] for t in p['targets'])
                return {'styles':[{'id':t['id'],'fill':'#FFFFFF','stroke':'#000000'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#000000'} for t in p['targets']]}
        output=self.folder/'paint-order-dependency'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,output,FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(refined,['P1'])
        self.assertEqual(read(output/'styles.json')['parts'][entries[1][0]]['style']['fill'],'#FFFFFF')
        self.assertEqual(result['model_calls']['refine'],1)
        self.assertEqual(len(read(output/'paint_dependencies.json')),1)

    def test_second_order_finding_preserves_the_first_corrected_pair(self):
        entries=[rectangle('A',f'face{i}',100,85,140,100) for i in range(3)]
        scene=scene_of(entries,source_hash=sha(self.source));seen=0
        def respond(tag,p):
            nonlocal seen
            if tag.startswith('critic'):
                seen+=1
                return {'checks':[({'id':t['id'],'verdict':'revise','affected_fields':['occlusion'],
                    'source_evidence':'This face is visible in front of the named neighbour.',
                    'rendered_difference':'The named neighbour covers this face.',
                    'suggestion':'Keep the previously resolved order and fix this pair.',
                    'in_front_of':['P1' if seen==1 else 'P2']}
                    if t['id']=='P0' else {'id':t['id'],'verdict':'accept'}) for t in p['targets']]}
            if tag.startswith('refine'):raise AssertionError('Order-only feedback needs no style request')
            return {'styles':[{'id':t['id'],'fill':'#CC5533','stroke':None,
                               'behind':['P1','P2'] if t['id']=='P0' else []} for t in p['targets']]}
        output=self.folder/'incremental-order'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,output,FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        rel=read(output/'occlusion.json');front=entries[0][0]
        pairs={(r['back'],r['front']) for r in rel['internal_model_edges']}
        self.assertTrue({(entries[1][0],front),(entries[2][0],front)}<=pairs)
        self.assertEqual(result['model_calls']['refine'],0)
        self.assertEqual(seen,2)

    def test_projected_face_may_hide_its_inferred_rear_geometry(self):
        from shapely.geometry import box
        from shape_integration.styling.controller import observed_order
        table={'rear':{'object_id':'solid','kind':'path'},'front':{'object_id':'solid','kind':'path'}}
        geoms={'rear':box(2,2,4,4),'front':box(0,0,10,10)};relations={}
        self.assertTrue(observed_order(table,geoms,relations,'rear',behind=['front']))
        self.assertEqual([(r['back'],r['front']) for r in relations['internal_model_edges']],[('rear','front')])
        self.assertFalse(relations.get('rejected_internal_orders'))

    def test_semantic_refine_keeps_accepted_neighbour_as_context_only(self):
        entries=[rectangle('A','left',60,85,50,80),rectangle('A','right',160,85,50,80)]
        scene=scene_of(entries,source_hash=sha(self.source));seen=[]
        def respond(tag,p):
            if tag.startswith('critic'):
                return {'checks':[({'id':t['id'],'verdict':'revise','affected_fields':['fill'],
                    'source_evidence':'Left face is red.','rendered_difference':'Left face is blue.',
                    'suggestion':'Use the visible red fill.'} if t['id']=='P0' and t['current_style']['fill']=='#0000FF'
                    else {'id':t['id'],'verdict':'accept'}) for t in p['targets']]}
            if tag.startswith('refine'):
                seen.append(([t['id'] for t in p['targets']],p['context_ids']))
                return {'styles':[{'id':t['id'],'fill':'#CC5533' if t['id']=='P0' else '#00FF00','stroke':None} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':'#0000FF' if t['id']=='P0' else '#CC5533','stroke':None} for t in p['targets']]}
        output=self.folder/'targeted-refine'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            run_case(scene,self.source,output,FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(seen,[(['P0'],['P0','P1'])])
        self.assertEqual(read(output/'styles.json')['parts'][entries[1][0]]['style']['fill'],'#CC5533')

    def test_visible_node_survives_contradictory_behind_own_container_reply(self):
        from PIL import Image,ImageDraw
        self.image=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(self.image)
        draw.rectangle((10,10,110,70),fill='#CC5533')
        draw.rectangle((40,30,60,50),fill='white')
        self.image.save(self.source)
        entries=[rectangle('A','panel',60,40,100,60),rectangle('A','node',50,40,20,20)]
        scene=scene_of(entries,size=(120,80),source_hash=sha(self.source))
        def respond(tag,p):
            if tag.startswith('critic'):
                return {'checks':[{'id':t['id'],'verdict':'accept'} for t in p['targets']]}
            return {'styles':[{'id':'P0','fill':'#CC5533','stroke':None},
                              {'id':'P1','fill':'#FFFFFF','stroke':None,'behind':['P0']}]}
        output=self.folder/'visible-internal-node'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,output,FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        rel=read(output/'occlusion.json');panel,node=[x[0] for x in entries]
        self.assertGreater(rel['part_orders']['A'].index(node),rel['part_orders']['A'].index(panel))
        self.assertEqual(read(output/'styles.json')['parts'][node]['style']['fill'],'#FFFFFF')
        self.assertTrue(rel['rejected_internal_orders'])
        self.assertEqual(result['model_calls']['refine'],0)

    def test_critic_orders_across_batches_without_semantic_refine(self):
        entries=[old.rectangle('A',f'face{i}',100,85,140,100) for i in range(3)]
        scene=old.scene_of(entries,source_hash=old.sha(self.source));self.opt['max_target_parts']=1
        seen={}
        def respond(tag,p):
            target=p['targets'][0]['id']
            if tag.startswith('critic'):
                seen[target]=seen.get(target,0)+1
                if target=='P0' and seen[target]==1:
                    return {'checks':[{'id':target,'verdict':'revise','affected_fields':['occlusion'],
                        'source_evidence':'The red face is in front.','rendered_difference':'It is hidden.',
                        'suggestion':'Put it in front of the two other faces.','in_front_of':['P1','P2']}]}
                return {'checks':[{'id':target,'verdict':'accept'}]}
            if tag.startswith('refine'):raise AssertionError('Pure order correction needs no semantic model call')
            return {'styles':[{'id':target,'fill':'#CC5533','stroke':None,
                               'behind':{'P0':['P1'],'P1':['P2'],'P2':[]}[target]}]}
        output=self.folder/'direct-order'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,output,old.FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        rel=read(output/'occlusion.json')
        self.assertEqual(rel['part_orders']['A'],[entries[i][0] for i in (1,2,0)])
        self.assertEqual(result['model_calls']['refine'],0)
        self.assertEqual(result['renders'],2)
        self.assertEqual(seen,{'P0':2,'P1':2,'P2':2})
        self.assertTrue(read(output/'context_recheck.json')['reordered_parts'])

    def test_batched_compound_can_order_a_face_in_another_batch(self):
        entries=[old.rectangle('A',f'face{i}',100,85,140,100) for i in range(25)]
        scene=old.scene_of(entries,source_hash=old.sha(self.source));self.opt['max_target_parts']=24
        def respond(tag,p):
            self.assertEqual(len(p['context_ids']),25)
            if tag.startswith('critic'):
                return {'checks':[{'id':t['id'],'verdict':'accept'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':'#CC5533','stroke':None,
                               'behind':['P24'] if t['id']=='P0' else []} for t in p['targets']]}
        output=self.folder/'cross-batch'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            run_case(scene,self.source,output,old.FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        relations=read(output/'occlusion.json')
        self.assertIn({'back':entries[0][0],'front':entries[24][0],'origin':'model_selection'},relations['internal_model_edges'])
        self.assertEqual(set(read(output/'styles.json')['parts']),set(k for k,_ in entries))

    def test_auxiliary_observation_cannot_clear_explicit_primary_width_finding(self):
        from PIL import Image,ImageDraw
        self.image=Image.new('RGB',(120,50),'white');ImageDraw.Draw(self.image).line([(10,25),(110,25)],fill='#444444',width=2)
        self.image.save(self.source);key,part=old.rectangle('A','edge',60,25,100,2)
        part.update(kind='path',commands=[['M',10,25.5],['L',110,25.5]],native_parameters_original=None)
        scene=old.scene_of([(key,part)],size=(120,50),source_hash=old.sha(self.source));self.opt['conditional_pattern_critic']=True
        self.opt['maximum_stroke_width_pt']=20.
        def respond(tag,p):
            if tag=='pattern_observation':return {'source_pattern':'solid','stroke_weight':'match','source_evidence':'Both corresponding strokes have the same visible weight.'}
            if tag.startswith('critic'):return {'checks':[{'id':t['id'],'verdict':'revise','affected_fields':['stroke_width'],
                'expected_stroke_width':'decrease','source_evidence':'The source looks lighter.','rendered_difference':'The result looks heavy.','suggestion':'Use a thinner stroke.'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#444444'} for t in p['targets']]}
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,self.folder/'width-confirmation',old.FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        record=read(self.folder/'width-confirmation/styles.json')['parts'][key]
        self.assertEqual(result['renders'],1);self.assertEqual(result['model_calls']['refine'],0)
        self.assertNotEqual(record['state'],'critic_accept')
        self.assertNotIn('width_critic_resolution',record)
        self.assertEqual(record['line']['source_observed_width_direction'],'decrease')
        self.assertAlmostEqual(record['style']['width_px'],2.,delta=.4)

    def test_short_pattern_observation_reuses_render_and_skips_semantic_refine(self):
        from PIL import Image,ImageDraw
        self.image=Image.new('RGB',(120,50),'white');draw=ImageDraw.Draw(self.image)
        for x in (20,50,80):draw.line([(x,25),(x+15,25)],fill='#444444',width=2)
        self.image.save(self.source)
        key,part=old.rectangle('A','edge',57,25,100,2)
        part.update(kind='path',commands=[['M',15,25.5],['L',100,25.5]],native_parameters_original=None)
        scene=old.scene_of([(key,part)],size=(120,50),source_hash=old.sha(self.source))
        self.opt['conditional_pattern_critic']=True
        critic_calls=0
        def respond(tag,p):
            nonlocal critic_calls
            if tag=='pattern_observation':return {'source_pattern':'dash','source_evidence':'Three separated gray marks of equal length are visible.'}
            if tag.startswith('critic'):
                critic_calls+=1
                return {'checks':[{'id':t['id'],'verdict':'uncertain' if critic_calls==1 else 'accept',
                    'source_evidence':'Only a few repeats are visible.', 'rendered_difference':'The source pattern is ambiguous.',
                    'suggestion':'Inspect the corresponding source spans.'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#444444'} for t in p['targets']]}
        runtime=old.FakeRuntime(respond)
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,self.folder/'short-pattern',runtime,self.opt,pool,threading.BoundedSemaphore(1))
        record=read(self.folder/'short-pattern/styles.json')['parts'][key]
        self.assertEqual(record['style']['dash'],'dash')
        self.assertEqual(result['model_calls']['pattern_observation'],1)
        self.assertEqual(result['model_calls']['refine'],0)
        self.assertEqual(result['renders'],2)
        self.assertEqual([tag for tag,_ in runtime.events].count('pattern_observation'),1)

    def test_accepted_style_is_not_overwritten_by_an_unknown_pattern_fallback(self):
        from PIL import Image,ImageDraw
        image=Image.new('RGB',(300,80),'white');draw=ImageDraw.Draw(image)
        draw.line([(10,40),(110,40)],fill='black',width=2)
        draw.line([(190,40),(290,40)],fill='black',width=2);image.save(self.source)
        key,part=old.rectangle('A','edge',150,40,280,2)
        part.update(kind='path',commands=[['M',10,40.5],['L',290,40.5]],native_parameters_original=None)
        scene=old.scene_of([(key,part)],size=(300,80),source_hash=old.sha(self.source))
        self.opt['conditional_pattern_critic']=True
        self.opt['sam_visual_pattern']=True
        def respond(tag,p):
            if tag=='pattern_observation':raise AssertionError('Accepted style must not be reclassified.')
            if tag.startswith('critic'):return {'checks':[{'id':t['id'],'verdict':'accept'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#000000'} for t in p['targets']]}
        runtime=old.FakeRuntime(respond)
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,self.folder/'accepted-pattern',runtime,self.opt,pool,threading.BoundedSemaphore(1))
        record=read(self.folder/'accepted-pattern/styles.json')['parts'][key]
        self.assertFalse(record['line']['dash_reliable'])
        self.assertEqual(record['style']['dash'],'solid')
        self.assertEqual(result['model_calls']['pattern_observation'],0)
        self.assertEqual(result['renders'],1)

    def test_sam_visual_review_resolves_reliable_pattern_conflict_without_semantic_retry(self):
        from PIL import Image,ImageDraw
        from shape_integration.common import save
        self.image=Image.new('RGB',(120,120),'white');draw=ImageDraw.Draw(self.image)
        mask=Image.new('L',(120,120),0);md=ImageDraw.Draw(mask)
        for x in (20,50,80):
            draw.line([(x,60),(x+15,60)],fill='#444444',width=2)
            md.line([(x,60),(x+15,60)],fill=255,width=2)
        self.image.save(self.source)
        key,part=old.rectangle('A','edge',57,60,100,2)
        part.update(kind='path',commands=[['M',15,60.5],['L',100,60.5]],native_parameters_original=None)
        scene=old.scene_of([(key,part)],size=(120,120),source_hash=old.sha(self.source))
        scene['case_id']='sam-conflict';evidence=self.folder/'sam';tile=evidence/'sam-conflict/tile'
        tile.mkdir(parents=True);self.image.save(tile/'tile.png');mask.save(tile/'target.png')
        save(evidence/'sam-conflict/line_evidence.json',{'case':'sam-conflict','size':120,'config':{'side':120},
            'parts':{key:{'tiles':[{'target':True,'dir':str(tile),'box':[0,0,120],
                'target_mask':str(tile/'target.png')}]}}})
        self.opt.update(conditional_pattern_critic=True,sam_visual_pattern=True)
        critics=0
        def respond(tag,p):
            nonlocal critics
            if tag=='pattern_observation':return {'source_pattern':'dash','stroke_weight':'match',
                'source_evidence':'Three separated marks are visible in CLEAN and the aligned target mask.'}
            if tag.startswith('critic'):
                critics+=1
                return {'checks':[{'id':t['id'],'verdict':'revise' if critics==1 else 'accept',
                    'source_evidence':'Repeated dash gaps are visible.', 'rendered_difference':'The pattern is continuous.',
                    'suggestion':'Use dash.','affected_fields':['line_pattern'],'expected_line_pattern':'dash'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#444444'} for t in p['targets']]}
        test=self
        class EvidenceRuntime(old.FakeRuntime):
            def call(self,root,tag,system,prompt,images,validator,budget):
                if tag=='pattern_observation':
                    test.assertTrue(any(label.startswith('SAM input RGB') for label,_ in images))
                    test.assertTrue(any(label.startswith('Aligned target foreground mask') for label,_ in images))
                return super().call(root,tag,system,prompt,images,validator,budget)
        output=self.folder/'sam-conflict-result'
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.controller.export',self.fake_export),patch(
                'shape_integration.styling.native_stroke.classify_pattern',return_value={'dash':'solid','dash_reliable':True}):
            result=run_case(scene,self.source,output,EvidenceRuntime(respond),self.opt,pool,
                threading.BoundedSemaphore(1),line_evidence=evidence)
        record=read(output/'styles.json')['parts'][key]
        self.assertEqual(record['style']['dash'],'dash')
        self.assertEqual(record['line']['pattern_origin'],'glm_sam_rgb_visual_observation')
        self.assertEqual(result['model_calls']['pattern_observation'],1)
        self.assertEqual(result['model_calls']['refine'],0)
        snapshots=sorted((output/'snapshots').glob('*/styles.json'))
        self.assertEqual(len(snapshots),2)
        self.assertEqual(read(snapshots[0])[key]['width_px'],read(snapshots[1])[key]['width_px'])

    def test_unconfirmed_rule_pattern_does_not_survive_an_unknown_native_fit(self):
        from PIL import Image,ImageDraw
        from shape_integration.styling import rules
        self.image=Image.new('RGB',(300,80),'white');draw=ImageDraw.Draw(self.image)
        draw.line([(10,40),(110,40)],fill='black',width=2)
        draw.line([(190,40),(290,40)],fill='black',width=2)
        self.image.save(self.source)
        key,part=old.rectangle('A','edge',150,40,280,2)
        part.update(kind='path',commands=[['M',10,40.5],['L',290,40.5]],native_parameters_original=None)
        scene=old.scene_of([(key,part)],size=(300,80),source_hash=old.sha(self.source))
        estimate=rules.estimate
        def coarse(*args,**kwargs):
            result=estimate(*args,**kwargs);result['style']['dash']='dash';return result
        with ThreadPoolExecutor(max_workers=2) as pool,patch('shape_integration.styling.rules.estimate',coarse),patch('shape_integration.styling.controller.export',self.fake_export):
            run_case(scene,self.source,self.folder/'unconfirmed',None,self.opt,pool,threading.BoundedSemaphore(1),rules_only=True)
        result=read(self.folder/'unconfirmed/styles.json')['parts'][key]
        self.assertEqual(result['history'][0]['style']['dash'],'dash')
        self.assertFalse(result['line']['dash_reliable'])
        self.assertEqual(result['style']['dash'],'solid')

    def test_neighbour_refine_rechecks_both_visible_crops(self):
        entries=[old.rectangle('A','face',70,85,60,60),old.rectangle('B','face',132,85,60,60)]
        scene=old.scene_of(entries,source_hash=old.sha(self.source));before=digest(scene);counts={'critic':0}
        def respond(tag,p):
            mode=tag.split('_')[0]
            if mode=='group':return {'groups':[]}
            if mode=='critic':
                counts['critic']+=1
                return {'checks':[{'id':t['id'],'verdict':'revise' if counts['critic']==1 else 'accept',
                    'source_evidence':'source fill differs','rendered_difference':'wrong fill','suggestion':'correct fill',
                    'affected_fields':['fill']} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':'#3388CC' if mode=='refine' else '#CC5533','stroke':None} for t in p['targets']]}
        with ThreadPoolExecutor(max_workers=4) as pool,patch('shape_integration.styling.controller.export',self.fake_export):
            result=run_case(scene,self.source,self.folder/'context',old.FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(result['renders'],2);self.assertEqual(counts['critic'],4)
        self.assertEqual(set(read(self.folder/'context/context_recheck.json')['reviewed_units']),{'A','B'})
        self.assertEqual(digest(scene),before)
        self.assert_style_context_matches_snapshot(self.folder/'context')

    def test_minimum_stroke_config_rejects_invalid_lengths(self):
        self.assertEqual(old.options({})['minimum_stroke_width_px'],0.)
        for value in (-1, float('nan'),float('inf'),True,'0.8',None):
            with self.assertRaises(ValueError):old.options({'styling':{'minimum_stroke_width_px':value}})

    # Run the inherited integration behavior with the new semantic-only context.
    def assert_style_context_matches_snapshot(self,folder):
        for path in folder.rglob('job.json'):
            if path.parent.name not in ('critic_000','refine_000'):continue
            job=read(path);saved=read(folder/job['render_snapshot']/'styles.json')
            field='current_style' if path.parent.name=='critic_000' else 'previous_style'
            for target in job['prompt']['targets']:
                key=job['output_mapping'][target['id']][0]
                fields=('fill','stroke','dash') if field=='current_style' else ('fill','stroke')
                self.assertEqual(target[field],{k:saved[key][k] for k in fields})
                self.assertNotIn('width_px',target[field])


def suite():
    names=['curve_bounds_use_extrema_not_control_points','containment_uses_geometry_not_only_bbox',
        'shared_edge_is_not_partial_overlap','relation_crops_separate_disconnected_components',
        'inner_object_above_outer_and_hollow_mask','cross_group_cycle_is_exposed',
        'internal_overlap_and_containment_never_become_relations_or_masks',
        'compound_candidates_are_unique_whole_component_pairs','whole_component_containment_and_union_holes',
        'unknown_order_preserves_pixels_but_confirmed_foreground_masks_all_its_parts',
        'gradient_contract_and_no_implicit_open_fill','gradient_estimator_and_solid_negative_control',
        'masked_foreground_does_not_contaminate_fill','crop_follows_geometry_and_half_pixel_transform',
        'native_gradient_export_and_geometry_invariance',
        'uniform_crop_resampling_preserves_measured_width_and_position',
        'pdf_crop_registration_at_both_zoom_directions_and_page_edges',
        'pdf_crop_keeps_thin_vector_stroke_instead_of_enlarging_raster',
        'pdf_crop_handles_blank_page_and_rejects_non_pdf_or_multiple_slides',
        'grouping_protocol_rejects_duplicate_or_unknown_members']
    tests=[old.StyleTests('test_'+n) for n in names]
    tests += [ControllerContext('test_controller_one_refine_and_preserves_unresolved_geometry'),
              ControllerContext('test_minimum_stroke_config_rejects_invalid_lengths'),
              ControllerContext('test_neighbour_refine_rechecks_both_visible_crops'),
              ControllerContext('test_off_image_target_does_not_abort_other_styles')]
    return unittest.TestSuite(tests)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(suite());raise SystemExit(not result.wasSuccessful())
