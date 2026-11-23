"""Behavioral integration checks: no external model calls."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor')]
from shape_integration.common import Runtime, read, save, sha, usage
from shape_integration.fitting.control import process_object
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.scene import native_parameters, assemble, export
from shape_integration.fitting.sizing import crop_size
from shape_integration.pipeline import source_manifest, run_case


def observed(oid, status='ok'):
    raw = {'object_id': oid, 'status': status, 'reason': 'visible cells', 'uncertain': [],
           'points': [{'id': str(i), 'xy': p} for i, p in enumerate([[100, 100], [300, 100], [300, 300], [100, 300], [500, 100], [500, 300]])],
           'parts': [{'id': 'a', 'kind': 'rectangle', 'point_ids': ['0', '1', '2', '3']},
                     {'id': 'b', 'kind': 'rectangle', 'point_ids': ['1', '4', '5', '2']}]}
    if status in ('needs_context', 'unsupported'):
        raw.update(points=[], parts=[])
    return raw


def reviewed(prompt, decision='accept'):
    refs = [p['id'] for p in prompt['fitted_parts']]
    return {'decision': decision,
            'issues': [] if decision == 'accept' else [
                {'kind': 'geometry', 'parts': refs, 'evidence': 'Left fitted edge lies inside the source cell fill, away from the source outline', 'action': 'move the left edge to the outline centerline'}]}


class FakeRuntime(Runtime):
    def __init__(self, config, responder=None):
        super().__init__(config, 'offline', 'offline')
        self.events = []; self.responder = responder

    def call(self, root, tag, system, prompt, images, validator, budget, **kwargs):
        self.events.append((tag, prompt))
        folder = Path(root)/(tag+'_00'); folder.mkdir(parents=True, exist_ok=True)
        save(folder/'request.json', {'prompt': prompt, 'images': [str(p) for _, p in images]})
        if self.responder:
            raw = self.responder(tag, prompt, len(self.events))
        elif tag == 'observe':
            raw = observed(prompt['target']['id'])
        elif tag == 'check':
            raw = reviewed(prompt)
        else:
            raise AssertionError('Unexpected offline call '+tag)
        if tag == 'observe' and isinstance(raw,dict) and isinstance(raw.get('parts'),list) and 'ownership' not in raw:
            raw={**raw,'ownership':{'unit':'other','unit_evidence':'Visible geometric cells','parts':[{'id':part['id'],'role':'body'}for part in raw['parts']]}}
        if tag == 'check' and 'source_unit' in prompt.get('output_schema',{}).get('properties',{}):
            raw={**raw,'source_unit':'other','source_unit_evidence':'Visible geometric cells'}
        if tag in ('check', 'sam_check') and 'revision_effect' in prompt.get('output_schema',{}).get('properties',{}) and 'revision_effect' not in raw:
            raw={**raw,'revision_effect':{'outcome':'uncertain','evidence':'Offline fixture does not assess revision progress.'}}
        save(folder/'raw.json', raw)
        return validator(raw)


class PipelineTests(unittest.TestCase):
    def test_case_identifiers_are_safe_path_components_without_losing_valid_punctuation(self):
        from shape_integration.fitting.control import safe_id
        self.assertEqual(safe_id('paper.1005.v2'), 'paper.1005.v2')
        self.assertEqual(safe_id('图表_一-2'), '%E5%9B%BE%E8%A1%A8_%E4%B8%80-2')
        self.assertEqual(safe_id('case name'), 'case%20name')
        self.assertEqual(safe_id('case%2Fname'), 'case%252Fname')
        for value in ('', '.', '..', 'case\nname'):
            with self.assertRaises(ValueError):safe_id(value)
        for value in ('case/name', 'case\\name'):
            self.assertNotIn('/', safe_id(value))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests'); self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work/'original.png'; self.input = self.work/'input.png'
        im = Image.new('RGB', (400, 200), 'white'); im.save(self.source); im.resize((2048, 1024)).save(self.input)
        self.item = dict(id='case', original='original.png', input='input.png', source_size=[400, 200],
                         input_size=[2048, 1024], source_sha256=sha(self.source), input_sha256=sha(self.input))
        self.candidate = dict(id='G1', label='compound', kind='compound_symbol', bbox_2d=[70, 40, 180, 140])
        self.config = read(ROOT/'config.json'); self.config['fitting']['coverage_rounds'] = 0
        self.options = self.config['fitting']; self.folder = self.work/'fitting'

    def test_sizing_floor_interior_cap_and_rejected_policy(self):
        for extent, expected in [([200, 100], [1080, 540]), ([500, 250], [1500, 750]),
                                 ([900, 450], [2048, 1024]), ([4000, 2000], [2048, 1024])]:
            self.assertEqual(list(crop_size(extent)[0]), expected)
        for extent in ([.1, 900], [900, .1], [359.99, 33], [360, 180], [682.666, 40]):
            size, _ = crop_size(extent); self.assertTrue(1080 <= max(size) <= 2048)
        with self.assertRaises(ValueError):
            crop_size([300, 150], min_long_edge=3000)

    def test_native_source_size_is_not_the_scale_reference(self):
        with Image.open(self.source) as image:
            job = prepare_candidate(image, self.candidate, self.item['input_size'], self.work/'prepared', padding=.2)
        m = job['frame']; f = FitFrame.from_metadata(m)
        self.assertNotEqual(m['sizing']['crop_size_global_px'], m['source_crop_size'])
        self.assertTrue(1080 <= max(f.input_size) <= 2048)
        for q in [[10.123, 20.456], [1000, 900]]:
            np.testing.assert_allclose(f.to_local(f.to_original(q)), q, atol=1e-9)
        with Image.open(self.work/'prepared/clean.png') as clean, Image.open(self.work/'prepared/overlay.png') as hint:
            self.assertEqual(clean.size, hint.size)
        p = read(self.work/'prepared/prompt.json')['prompt']
        self.assertEqual(p['local_coordinates']['width_px'], f.input_size[0])

    def test_stage_runtime_shares_limiter_but_not_retry_config(self):
        runtime = FakeRuntime(self.config); other = runtime.for_stage(shape_schema_retries=0)
        self.assertIs(runtime.slots, other.slots)
        self.assertEqual(runtime.config['shape_schema_retries'], 1)
        self.assertEqual(other.config['shape_schema_retries'], 0)

    def test_accounting_ignores_unpublished_preparation_trees(self):
        save(self.work/'object/revision_00.preparing/response_0.json', {'usage': {'total_tokens': 999}})
        save(self.work/'object/revision_01/calls/observe_00/response_0.json', {'usage': {'total_tokens': 42}})
        self.assertEqual(usage(self.work)['totals']['total_tokens'], 42)

    def run_object(self, runtime, prepare=False):
        return process_object(self.folder/'objects/G1', self.candidate, self.item, self.work,
                              runtime.for_stage(shape_schema_retries=0), self.options, namespace='test/case', prepare_only=prepare)

    def test_one_compound_observation_check_resume_and_tamper(self):
        runtime = FakeRuntime(self.config)
        result = self.run_object(runtime)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check'])
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(self.run_object(runtime), result)
        self.assertEqual(len(runtime.events), 2)
        target = self.folder/'objects/G1/revision_00/fit_original.json'
        target.write_text(target.read_text()+' ')
        with self.assertRaises(ValueError):
            self.run_object(runtime)

    def test_partial_without_critic_repair_target_stops_without_redrawing(self):
        def responder(tag, prompt, count):
            if tag == 'check':
                return reviewed(prompt)
            return observed('G1', 'partial') if count == 1 else {'invalid': True}
        runtime = FakeRuntime(self.config, responder)
        result = self.run_object(runtime)
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check'])
        self.assertFalse((self.folder/'objects/G1/revision_01').exists())
        self.assertEqual(result['stop_reason'], 'partial_without_actionable_revision')
        self.assertNotEqual(result['visual_status'], 'accepted_by_model')
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': result}, {}, [])
        counts = export(scene, self.folder, self.source)
        self.assertEqual(counts['geometry_final']['native'], 2)
        self.assertEqual(counts['geometry_accepted']['native'], 0)
        self.assertEqual(scene['objects']['G1']['selected_revision'], 'revision_00')

    def test_failed_actionable_revision_still_retains_partial_geometry(self):
        def responder(tag, prompt, count):
            if tag == 'check':return reviewed(prompt, 'revise')
            return observed('G1', 'partial') if count == 1 else {'invalid': True}
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertTrue((self.folder/'objects/G1/revision_01/calls/observe_00/raw.json').exists())

    def test_partial_with_missing_member_outside_crop_recrops_before_critic(self):
        self.options['recover_clipped_partial']=True
        seen=[]
        def responder(tag,prompt,count):
            if tag=='observe':
                seen.append(tag)
                return observed('G1','partial' if len(seen)==1 else 'ok')
            return reviewed(prompt)
        runtime=FakeRuntime(self.config,responder)
        with patch('shape_integration.fitting.control.withheld_source_context',return_value=True), \
             patch('shape_integration.fitting.control.context_crop',return_value=[0,0,400,200]) as recrop:
            result=self.run_object(runtime)
        self.assertEqual([tag for tag,_ in runtime.events],['observe','observe','check'])
        self.assertEqual(recrop.call_count,1)
        self.assertEqual(result['context_rounds'],1)
        self.assertEqual(result['state'],'accepted_by_model')

    def test_unaccepted_refinements_cannot_replace_baseline_when_budget_exhausts(self):
        self.options['max_revisions'] = 3
        attempts = []
        def responder(tag, prompt, count):
            if tag == 'check':
                return reviewed(prompt, 'revise')
            raw = observed('G1')
            for point in raw['points']:
                point['xy'][0] += 20 * len(attempts)
            attempts.append(raw)
            return raw
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['corrections'], 3)
        self.assertEqual(result['stop_reason'], 'revision_budget_exhausted')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['unaccepted_revision'], 'revision_03')
        before = copy.deepcopy(result)
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': result}, {}, [])
        selected = read(self.folder/'objects/G1/revision_00/fit_original.json')
        self.assertEqual(list(scene['points'].values()), selected['points'])
        counts = export(scene, self.folder, self.source)
        self.assertEqual(counts['geometry_final']['native'], 2)
        self.assertEqual(counts['geometry_accepted']['native'], 0)
        self.assertEqual(scene['counts']['accepted_by_model'], 0)
        self.assertEqual(scene['counts']['unresolved_objects'], 1)
        self.assertEqual(result, before)
        self.assertEqual(scene['objects']['G1']['stop_reason'], 'revision_budget_exhausted')
        self.assertIn('data-state="unresolved"', (self.folder/'geometry_final.svg').read_text())
        self.assertNotIn('G1/a', (self.folder/'geometry_accepted.svg').read_text())
        with ZipFile(self.folder/'geometry_final.pptx') as archive:
            xml = archive.read('ppt/slides/slide1.xml').decode()
        self.assertIn('G1/a [unresolved]', xml)
        self.assertIn('G1/b [unresolved]', xml)
        for suffix in ['svg', 'png', 'pptx']:
            self.assertEqual(sha(self.folder/('geometry_final.'+suffix)),
                             sha(self.folder/('geometry_diagnostic.'+suffix)))
        self.assertEqual(read(self.folder/'export.json')['primary_output'], 'geometry_final')

    def test_completed_object_rejects_a_different_target_before_reusing_fit(self):
        runtime = FakeRuntime(self.config)
        self.run_object(runtime)
        changed = {**self.candidate, 'label': 'another source instance'}
        with self.assertRaisesRegex(ValueError, 'target identity changed'):
            process_object(self.folder/'objects/G1', changed, self.item, self.work,
                           runtime, self.options, namespace='test/case')
        self.assertEqual(len(runtime.events), 2)

    def test_final_export_retains_geometry_after_critic_rejection(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                return observed('G1')
            raw = reviewed(prompt, 'reject')
            raw['issues'][0]['kind'] = 'wrong_target'
            return raw
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'rejected_by_model')
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': result}, {}, [])
        counts = export(scene, self.folder, self.source)
        self.assertEqual(counts['geometry_final']['native'], 2)
        self.assertEqual(counts['geometry_accepted']['native'], 0)
        self.assertIn('G1/a', (self.folder/'geometry_final.svg').read_text())
        self.assertEqual(scene['objects']['G1']['state'], 'rejected_by_model')

    def test_final_export_does_not_invent_geometry_when_no_revision_succeeded(self):
        result = self.run_object(FakeRuntime(self.config, lambda *args: observed('G1', 'unsupported')))
        self.assertIsNone(result['selected_revision'])
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': result}, {}, [])
        counts = export(scene, self.folder, self.source)
        self.assertEqual(counts['geometry_final']['native'], 0)
        self.assertEqual(counts['geometry_final']['custom'], 0)
        self.assertEqual(scene['objects']['G1']['state'], 'unsupported')

    def test_context_is_replanned_and_points_reobserved_in_new_frame(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                return observed('G1', 'needs_context' if count == 1 else 'ok')
            if tag == 'plan':
                return dict(bbox_2d=[0, 0, 2047, 1023], reason='full context')
            return reviewed(prompt)
        runtime = FakeRuntime(self.config, responder); result = self.run_object(runtime)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'plan', 'observe', 'check'])
        self.assertEqual(result['context_rounds'], 1)
        old = read(self.folder/'objects/G1/revision_00/frame.json'); new = read(self.folder/'objects/G1/revision_01/frame.json')
        self.assertNotEqual(old['source_crop_half_open'], new['source_crop_half_open'])
        self.assertEqual(new['source_crop_half_open'], [0, 0, 400, 200])

    def test_native_affine_ellipse_and_noncircular_corners(self):
        frame = FitFrame((1000, 500), (2048, 1024), (1000, 800), (0, 0, 2048, 1024)).metadata()
        params = dict(center=[200, 200], width=100, height=50, angle_radians=.7, radius=10)
        part = dict(kind='ellipse', parameters_local=params)
        out = native_parameters(part, frame)
        a = .7; rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        axes = np.diag([1., 1.6]) @ rot @ np.diag([100., 50.])
        b = out['angle_radians']; basis = np.array([[math.cos(b), -math.sin(b)], [math.sin(b), math.cos(b)]])
        expected = basis @ np.diag([out['width']**2, out['height']**2]) @ basis.T
        np.testing.assert_allclose(axes @ axes.T, expected, atol=1e-9)
        self.assertIsNone(native_parameters(dict(kind='rounded_rectangle', parameters_local=params), frame))
        self.assertIsNone(native_parameters(dict(kind='rectangle', parameters_local=params), frame))

    def test_scene_export_preserves_shared_points_and_compound_group(self):
        runtime = FakeRuntime(self.config); record = self.run_object(runtime)
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': record}, {}, [])
        self.assertEqual(len(scene['objects']), 1); self.assertEqual(len(scene['parts']), 2)
        self.assertEqual(len(scene['points']), 6)
        counts = export(scene, self.folder, self.source)
        self.assertEqual(counts['geometry_diagnostic']['groups'], 1)
        with ZipFile(self.folder/'geometry_diagnostic.pptx') as z:
            xml = z.read('ppt/slides/slide1.xml').decode()
        self.assertIn('G1/a', xml); self.assertIn('G1/b', xml)
        self.assertIn('p:grpSp', xml)

    def test_batch_reuse_does_not_relocalize_and_discovers_through_image_gate(self):
        source = self.work/'old'; case = source/'cases/case'
        save(source/'manifest.json', {'images': [self.item]})
        save(case/'geometry.json', {'image_size': self.item['source_size'], 'coordinate_system': 'image_pixel_centers', 'objects': [self.candidate]})
        save(case/'summary.json', {'image_coverage_status': 'complete', 'errors': {}, 'local_failures': 0})
        save(case/'dedup/image_boxes_original.json', {'assets': [{'id': 'I1', 'bbox_2d': [10, 10, 40, 40]}]})
        prov = source_manifest(source, [self.item]); self.config['fitting']['coverage_rounds'] = 1
        def responder(tag, prompt, count):
            if tag == 'observe': return observed(prompt['target']['id'])
            if tag == 'check': return reviewed(prompt)
            if tag.startswith('batch_'):
                return {'decisions': [{'shape_id': c['shape_id'], 'decision': 'delete',
                    'image_id': c['image_ids'][0], 'reason': 'Same detailed image asset.'} for c in prompt['candidates']]}
            def box(b): return [(v+.5)*self.item['input_size'][i%2]/self.item['source_size'][i%2]-.5 for i,v in enumerate(b)]
            return dict(status='uncertain', new_objects=[dict(label='icon', kind='compound_symbol', bbox_2d=box([10, 10, 40, 40]), reason='new'),
                                                         dict(label='cells', kind='compound_symbol', bbox_2d=box([220, 50, 340, 150]), reason='new')],
                        possible_duplicates=[], uncertain=['coverage requires another pass'], repair_existing=[])
        runtime = FakeRuntime(self.config, responder)
        with ThreadPoolExecutor(max_workers=2) as pool, patch('shape_integration.batch.case', side_effect=AssertionError('Must not redetect')):
            summary = run_case(self.work, self.work/'new', self.item, runtime, self.config, pool, source, prov, False)
        self.assertEqual(summary['candidates'], 2)
        self.assertEqual([t for t, _ in runtime.events].count('observe'), 2)
        self.assertEqual(len(summary['coverage'][0]['image_removed_ids']), 1)
        self.assertTrue(summary['coverage'][0]['image_removed_ids'][0].startswith('F1_N001_'))
        self.assertEqual(sum(tag.startswith('batch_') for tag, _ in runtime.events), 1)
        self.assertEqual(summary['state'], 'completed_with_unresolved')
        self.assertEqual(read(self.work/'new/cases/case/geometry.json')['objects'], [self.candidate])

    def test_coverage_view_contains_no_source_pixels_and_discovery_ids_do_not_collide(self):
        from shape_integration.fitting.coverage import review
        candidate = {**self.candidate, 'id': 'F1_N001'}
        record = {'state': 'unsupported', 'selected_revision': None, 'history': []}
        scene = assemble(self.folder, self.item, [candidate], {'F1_N001': record}, {}, [])
        Image.new('RGB', tuple(self.item['input_size']), 'black').save(self.input)
        Image.new('RGB', tuple(self.item['source_size']), 'black').save(self.source)
        def response(tag, prompt, count):
            self.assertEqual(tag, 'coverage')
            return dict(status='complete', possible_duplicates=[], uncertain=[], repair_existing=[], new_objects=[
                dict(label='missing nodes', kind='compound_symbol', bbox_2d=[100, 100, 200, 200], reason='source-visible'),
                dict(label='missing frame', kind='rectangle', bbox_2d=[400, 100, 500, 200], reason='source-visible')])
        result = review(scene, self.item, self.work, self.work/'coverage', FakeRuntime(self.config, response),
                        self.options, [candidate], [], .5, 0)
        with Image.open(self.work/'coverage/fit.png') as fitted:
            self.assertTrue(all(low == high == 255 for low, high in fitted.getextrema()))
        self.assertEqual(len(set(result['new_candidate_ids'])), 2)
        self.assertNotIn(candidate['id'], result['new_candidate_ids'])
        self.assertEqual(result['duplicate_action'], 'repair_explicit_parts')

    def test_fresh_localization_enters_same_fitting_queue(self):
        root = self.work/'fresh'; runtime = FakeRuntime(self.config)
        def detect(run, item, workspace, shared_runtime):
            self.assertIs(shared_runtime, runtime)
            folder = run/'cases'/item['id']
            save(folder/'geometry.json', {'image_size': item['source_size'], 'coordinate_system': 'image_pixel_centers', 'objects': [self.candidate]})
            return {'image_coverage_status': 'complete', 'errors': {}, 'local_failures': 0}
        with ThreadPoolExecutor(max_workers=2) as pool, patch('shape_integration.batch.case', side_effect=detect) as detect_call:
            summary = run_case(self.work, root, self.item, runtime, self.config, pool, None, None, False)
        self.assertEqual(detect_call.call_count, 1)
        self.assertEqual(summary['accepted_by_model'], 1)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check'])

    def test_new_checkpoint_discovery_reuses_objects_and_source_scene_context(self):
        import shutil
        from shape_integration.fitting.context import identity_context
        source=self.work/'localization';case=source/'cases/case'
        save(source/'manifest.json',{'images':[self.item]})
        save(case/'geometry.json',{'image_size':self.item['source_size'],
            'coordinate_system':'image_pixel_centers','objects':[self.candidate]})
        save(case/'summary.json',{'image_coverage_status':'complete','errors':{},'local_failures':0})
        save(case/'dedup/image_boxes_original.json',{'assets':[]})
        provenance=source_manifest(source,[self.item]);old=self.work/'old_checkpoint'
        with ThreadPoolExecutor(max_workers=1) as pool:
            run_case(self.work,old,self.item,FakeRuntime(self.config),self.config,pool,source,provenance,False)
        previous=old/'cases/case/fitting';root=self.work/'repair_checkpoint';fitted=root/'cases/case/fitting'
        shutil.copytree(previous/'objects',fitted/'objects')
        added={**self.candidate,'id':'G2','bbox_2d':[220,50,340,150],'origin':'fitting_coverage'}
        save(fitted/'resume_candidates.json',{'source_sha256':self.item['source_sha256'],
            'localization_sha256':sha(case/'geometry.json'),'objects':[self.candidate,added],
            'previous_scene':str(previous/'scene.json'),'previous_scene_sha256':sha(previous/'scene.json'),
            'repair_existing':[]})
        runtime=FakeRuntime(self.config);old_receipt=sha(previous/'objects/G1/done.json')
        with ThreadPoolExecutor(max_workers=1) as pool, \
                patch('shape_integration.pipeline.identity_context',wraps=identity_context) as context:
            summary=run_case(self.work,root,self.item,runtime,self.config,pool,source,provenance,False)
        self.assertEqual(summary['accepted_by_model'],2)
        self.assertEqual([tag for tag,_ in runtime.events],['observe','check'])
        self.assertTrue(all(call.kwargs['represented_scene']['source_sha256']==self.item['source_sha256']
                            for call in context.call_args_list))
        self.assertEqual(sha(previous/'objects/G1/done.json'),old_receipt)
        self.assertEqual(sha(fitted/'objects/G1/done.json'),old_receipt)
        doc=read(fitted/'resume_candidates.json');doc['previous_scene_sha256']='bad'
        save(fitted/'resume_candidates.json',doc)
        with ThreadPoolExecutor(max_workers=1) as pool, self.assertRaisesRegex(ValueError,'source scene changed'):
            run_case(self.work,root,self.item,runtime,self.config,pool,source,provenance,False)

    def test_transport_resume_keeps_old_coverage_and_fits_new_discovery(self):
        source = self.work/'localization'; case = source/'cases/case'
        save(source/'manifest.json', {'images': [self.item]})
        save(case/'geometry.json', {'image_size': self.item['source_size'],
             'coordinate_system': 'image_pixel_centers', 'objects': [self.candidate]})
        save(case/'summary.json', {'image_coverage_status': 'complete', 'errors': {}, 'local_failures': 0})
        save(case/'dedup/image_boxes_original.json', {'assets': []})
        provenance = source_manifest(source, [self.item]); self.options['coverage_rounds'] = 1
        class ModelTransportError(RuntimeError): pass
        attempt = 0
        def responder(tag, prompt, count):
            if tag == 'observe':
                if attempt == 0 and prompt['target']['id'] == 'G1':
                    raise ModelTransportError('Interrupted stream')
                return observed(prompt['target']['id'])
            if tag == 'check': return reviewed(prompt)
            if tag == 'coverage':
                return dict(status='complete', possible_duplicates=[], uncertain=[], repair_existing=[],
                            new_objects=[dict(label='missing cells', kind='compound_symbol',
                                bbox_2d=[1100, 100 + attempt*300, 1800, 350 + attempt*300],
                                reason='Source instance '+str(attempt))])
            raise AssertionError(tag)
        root = self.work/'new'; runtime = FakeRuntime(self.config, responder)
        def execute():
            with ThreadPoolExecutor(max_workers=2) as pool:
                return run_case(self.work, root, self.item, runtime, self.config, pool, source, provenance, False)
        first = execute(); first_id = first['coverage'][0]['new_candidate_ids'][0]
        fitted = root/'cases/case/fitting'
        old_review = next(fitted.glob('coverage_00_*/record.json'))
        old_hash = sha(old_review); old_fit = sha(fitted/'objects'/first_id/'done.json')
        attempt = 1
        second = execute(); second_id = second['coverage'][0]['new_candidate_ids'][0]
        self.assertEqual(len(first['coverage']), 1)
        self.assertEqual(len(second['coverage']), 1)
        self.assertEqual(second['accepted_by_model'], 2)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(list(fitted.glob('coverage_00_*/record.json'))), 2)
        self.assertEqual(sha(old_review), old_hash)
        self.assertEqual(sha(fitted/'objects'/first_id/'done.json'), old_fit)
        self.assertEqual(read(fitted/'objects'/second_id/'record.json')['candidate']['id'], second_id)


if __name__ == '__main__':
    unittest.main(verbosity=2)
