"""Source/mask counterexamples and scheduler behavior, without model calls."""
import copy
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor'), str(ROOT/'tests')]
from shape_integration.common import read
from shape_integration.fitting.context import identity_context
from shape_integration.fitting.geometry_guard import compare_revision, geometry_signature
from shape_integration.fitting.control import process_object
from shape_integration.fitting.paths import box_commands
from shape_integration.sam_guidance import GuidedSamEvidence
import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed


def fitted(box, radius=20, kind='rounded_rectangle', ident='arbitrary'):
    x0, y0, x1, y1 = box
    return {'coordinate_frame': 'original_pixel_centers', 'fit_status': 'ok', 'observation_status': 'ok',
            'bbox_2d': list(box), 'parts': [{'id': ident, 'status': 'ok', 'kind': kind,
                'commands': box_commands([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0, 0], radius)}]}


class GuardTests(unittest.TestCase):
    def test_context_completion_is_not_a_same_view_primitive_regression(self):
        from shape_integration.fitting.conflict_critic import changed_primitives
        before=fitted([10,10,60,80],0,kind='rectangle')
        after=copy.deepcopy(before);after['parts'][0]['kind']='path'
        before['frame']={'crop_bounds_global_half_open':[0,0,100,100]}
        after['frame']={'crop_bounds_global_half_open':[0,0,300,100]}
        self.assertEqual(changed_primitives(before,after),[])
        after['frame']=copy.deepcopy(before['frame'])
        self.assertEqual(changed_primitives(before,after),['arbitrary'])

    def test_primitive_observation_is_bounded_and_never_discards_added_parts(self):
        from shape_integration.fitting.conflict_critic import changed_primitives,inspect_revision
        before=fitted([10,10,60,80],0,kind='rectangle');before['parts'][0]['bbox_2d']=[10,10,60,80]
        after=copy.deepcopy(before);after['parts'][0].update(kind='path',commands=[['M',10,10],['L',60,0],['L',60,70],['L',10,80],['Z']],bbox_2d=[10,0,60,80])
        self.assertEqual(changed_primitives(before,after),['arbitrary'])
        image=Image.new('RGB',(100,100),'white');ImageDraw.Draw(image).rectangle((10,10,60,80),outline='black',width=2)
        image.save(self.root/'source.png')
        class Observer:
            def for_stage(owner,**kw):return owner
            def call(owner,folder,tag,system,prompt,images,validate,budget):
                self.assertEqual(budget,8192)
                self.assertEqual(len(images),3)
                return validate({'choice':'before','source_evidence':'The source top edge is horizontal.'})
        observed=inspect_revision(self.root/'source.png',before,after,self.root/'inspect',Observer(),{'check_max_tokens':100000})
        self.assertEqual(observed['choice'],'before')
        after['parts'].append({**copy.deepcopy(after['parts'][0]),'id':'new_face'})
        self.assertEqual(changed_primitives(before,after),[])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_critic_retains_unaccepted_revision_without_a_second_veto_agent(self):
        fixture=fixtures.PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        # Retyping an ordinary primitive needs the same protection as a 3D
        # face; a dashed ellipse must not become a scalloped freeform path.
        fixture.candidate['kind']='ellipse'
        fixture.options.update(geometry_guard_enabled=True,conditional_geometry_critic=True,max_revisions=1)
        def respond(tag,p,count):
            if p.get('protocol')=='shape_integration.primitive_change_observation.1':
                return {'choice':'before','source_evidence':'The original top edge is horizontal; the changed face tilts away from it.'}
            if tag=='observe':
                raw=observed('G1')
                if 'revision_task' in p:
                    raw['points'][1]['xy'][1]=80
                    raw['parts'][0]={'id':'a','kind':'path','closed':True,'segments':[
                        {'kind':'line','point_ids':[a,b]} for a,b in [('0','1'),('1','2'),('2','3'),('3','0')]]}
                return raw
            return {'decision':'revise','issues':[{'kind':'geometry','parts':['F1'],
                'evidence':'The top edge should be tilted.','action':'Retype this face as a parallelogram.'}]}
        runtime=FakeRuntime(fixture.config,respond)
        result=process_object(fixture.folder/'objects/G1',fixture.candidate,fixture.item,fixture.work,
                              runtime,fixture.options,namespace='test/case')
        self.assertEqual(result['selected_revision'],'revision_00')
        self.assertEqual(result['stop_reason'],'revision_budget_exhausted')
        self.assertEqual(sum(p.get('protocol')=='shape_integration.primitive_change_observation.1' for _,p in runtime.events),0)
        self.assertEqual(len([p for _,p in runtime.events if 'fitted_parts' in p]),2)
        self.assertEqual(sum('target' in p and tag=='observe' for tag,p in runtime.events),2)

    def fixture(self, *, scale=1, filled=False, fragmented=False, mask_shift=0):
        size = (int(500*scale), int(380*scale))
        box = np.array([85, 55, 395, 320])*scale
        source = Image.new('RGB', size, 'white'); draw = ImageDraw.Draw(source)
        draw.rounded_rectangle(box.tolist(), radius=30*scale, outline='#245a96', width=max(1, round(3*scale)),
                               fill='#d5e2f0' if filled else None)
        # A real external leader must not be mistaken for the target side.
        draw.line([(box[0], box[1]+50*scale), (box[0]-55*scale, box[1]+130*scale),
                   (box[0], box[3]-30*scale)], fill='black', width=max(1, round(scale)))
        source.save(self.root/'source.png')
        mask = Image.new('L', size); md = ImageDraw.Draw(mask)
        shifted = (box+np.array([mask_shift, 0, mask_shift, 0])*scale).tolist()
        md.rounded_rectangle(shifted, radius=30*scale, outline=255, width=max(1, round(3*scale)),
                             fill=255 if filled else None)
        if fragmented:
            arr = np.array(mask); yy, xx = np.indices(arr.shape)
            arr[((xx+yy)//max(1, round(9*scale))) % 2 == 0] = 0; mask = Image.fromarray(arr)
        mask.save(self.root/'mask.png')
        group = {'id': 'sam', 'representative': {'bbox': shifted, 'category': 'rectangle', 'score': .9,
                 'mask_quality_ok': True, 'mask_path': str(self.root/'mask.png'),
                 'frame': {'box': [0, 0, *size], 'size': list(size)}}}
        provider = GuidedSamEvidence([group], min_mask_short_side=0)
        candidate = {'id': 'target', 'kind': 'rounded_rectangle', 'label': 'generic', 'bbox_2d': box.tolist()}
        provider.set_inventory([candidate])
        return box, candidate, provider

    def test_shifted_side_is_blocked_for_strokes_fills_dashes_and_scales(self):
        for scale in [.7, 1., 2.3]:
            for filled, fragmented in [(False, False), (True, False), (False, True)]:
                with self.subTest(scale=scale, filled=filled, fragmented=fragmented):
                    box, candidate, provider = self.fixture(scale=scale, filled=filled, fragmented=fragmented)
                    before = fitted(box, 30*scale)
                    after = fitted(box+np.array([0, 0, 25*scale, 0]), 30*scale)
                    result = compare_revision(before, after, self.root/'source.png', provider, candidate)
                    self.assertEqual(result['decision'], 'rollback', result)

    def test_external_leader_cannot_replace_an_intact_side(self):
        box, candidate, provider = self.fixture(fragmented=True)
        before = fitted(box, 30)
        after = fitted(box, 30, kind='path')
        after['parts'][0]['commands'] = [['M', 85, 55], ['L', 395, 55], ['L', 395, 320],
            ['L', 85, 320], ['L', 30, 185], ['Z']]
        after['bbox_2d'] = [30, 55, 395, 320]
        result = compare_revision(before, after, self.root/'source.png', provider, candidate)
        self.assertEqual(result['decision'], 'rollback', result)

    def test_source_supported_improvement_and_subpixel_changes_are_allowed(self):
        box, candidate, provider = self.fixture(filled=True)
        before = fitted(box+np.array([0, 0, -6, 0]), 30)
        good = fitted(box, 30)
        result = compare_revision(before, good, self.root/'source.png', provider, candidate)
        self.assertNotIn(result['decision'], ['rollback', 'stop'], result)
        tiny = fitted(box+np.array([.2, .1, .2, .1]), 30)
        self.assertNotEqual(compare_revision(good, tiny, self.root/'source.png', provider, candidate)['decision'], 'rollback')

    def test_bad_or_missing_sam_abstains_instead_of_forcing_a_box(self):
        box, candidate, provider = self.fixture(mask_shift=40)
        before, after = fitted(box, 30), fitted(box+np.array([0, 0, 20, 0]), 30)
        self.assertNotEqual(compare_revision(before, after, self.root/'source.png', provider, candidate)['decision'], 'rollback')
        self.assertEqual(compare_revision(before, after, self.root/'source.png', None, candidate)['decision'], 'abstain')
        blank = Image.new('RGB', (500, 380), 'white'); blank.save(self.root/'source.png')
        self.assertEqual(compare_revision(before, after, self.root/'source.png', provider, candidate)['decision'], 'abstain')

    def test_parent_mask_does_not_constrain_compound_children(self):
        box, candidate, provider = self.fixture(filled=True)
        before, after = fitted(box, 30), fitted(box+np.array([0, 0, 20, 0]), 30)
        before['parts'].append(copy.deepcopy(before['parts'][0]))
        self.assertEqual(compare_revision(before, after, self.root/'source.png', provider, candidate)['decision'], 'abstain')

    def test_low_score_mask_requires_the_same_independent_source_verification(self):
        box, candidate, provider = self.fixture(fragmented=True)
        provider.groups[0]['representative']['score'] = .35
        self.assertIsNone(provider.support(candidate))
        before = fitted(box, 30)
        after = fitted(box+np.array([0, 0, 25, 0]), 30)
        result = compare_revision(before, after, self.root/'source.png', provider, candidate)
        self.assertEqual(result['decision'], 'rollback', result)
        self.assertEqual(result['sam_score'], .35)
        Image.new('RGB', (500, 380), 'white').save(self.root/'source.png')
        self.assertEqual(compare_revision(before, after, self.root/'source.png', provider, candidate)['decision'], 'abstain')

    def test_crop_review_reason_does_not_redefine_fitting_identity(self):
        candidate = {'id': 'symbol', 'label': 'layered icon', 'kind': 'compound_symbol', 'bbox_2d': [2, 3, 50, 60]}
        histories = [[{'id': 'symbol', 'action': 'keep', 'reason': text}] for text in ['Box fully locates icon.', 'Icon is inside box.']]
        contexts = [identity_context(candidate, [candidate], [], h, .2) for h in histories]
        self.assertEqual(contexts[0], contexts[1])
        changed = identity_context(candidate, [candidate], [], [{'id': 'symbol', 'action': 'update',
                                    'reason': 'The enclosing panel also belongs to this group.'}], .2)
        self.assertEqual(changed, contexts[0])

    def test_geometry_signature_ignores_only_metadata_and_tracks_topology(self):
        a = fitted([10, 20, 80, 90]); b = copy.deepcopy(a)
        b['parts'][0]['id'] = 'changed'; b['parts'][0]['key'] = ['namespace', 'id']
        self.assertEqual(geometry_signature(a), geometry_signature(b))
        b['parts'][0]['commands'].pop()
        self.assertNotEqual(geometry_signature(a), geometry_signature(b))

    def test_partial_geometry_is_not_stopped_as_a_completed_noop(self):
        a = fitted([10, 20, 80, 90]); a.update(fit_status='partial', observation_status='partial')
        result = compare_revision(a, copy.deepcopy(a), self.root/'missing.png', None, {})
        self.assertEqual(result['decision'], 'abstain')

    def test_neighbor_colored_edge_in_bad_mask_cannot_freeze_wrong_extent(self):
        box, candidate, provider = self.fixture()
        # A common diagram layout: the target is dark blue, and an adjacent
        # orange panel has a nearby parallel side accidentally included by SAM.
        with Image.open(self.root/'source.png') as image:
            source = image.convert('RGB')
        ImageDraw.Draw(source).line([(415, 85), (415, 290)], fill='#ed8b22', width=3)
        source.save(self.root/'source.png')
        with Image.open(self.root/'mask.png') as image:
            mask = image.convert('L')
        md = ImageDraw.Draw(mask)
        md.rectangle((388, 85, 402, 290), fill=0)
        md.line([(415, 85), (415, 290)], fill=255, width=3)
        mask.save(self.root/'mask.png')
        provider.groups[0]['representative']['bbox'][2] = 415
        before = fitted([85, 55, 415, 320], 30)
        after = fitted(box, 30)
        result = compare_revision(before, after, self.root/'source.png', provider, candidate)
        self.assertNotEqual(result['decision'], 'rollback', result)


class GuardControlTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def test_source_disproved_repeated_closure_can_use_the_existing_context_budget(self):
        self.options.update(max_revisions=1,max_context_rounds=1,geometry_guard_enabled=True)
        checks=0
        def responder(tag,prompt,count):
            nonlocal checks
            if tag=='observe':return observed('G1')
            if tag=='plan':
                self.assertIn('COMPLETE connected source outline',prompt['context_request'])
                return {'bbox_2d':[0,0,2047,1023],'reason':'Include the clipped connected outline'}
            checks+=1
            raw=reviewed(prompt,'revise' if checks==1 else 'accept')
            if checks==1:raw['independent_source_observations']=[{'visible_line':'no','source_evidence':'The closed side has no source pixels.'}]
            return raw
        # Inject the already validated independent observation at the critic
        # boundary; the real source-image gate is tested in test_source_spans.
        from unittest.mock import patch
        from shape_integration.fitting.critic import part_inventory
        def checker(job,folder,runtime,options):
            local=read(folder/'fit_local.json');mapping,inventory,_=part_inventory(local)
            raw=responder('check',{'fitted_parts':inventory},0)
            return {**raw,'reason':'Missing source side' if raw['decision']=='revise' else 'Correct',
                    'issues':[{**i,'part_ids':[mapping[k] for k in i['parts']]} for i in raw['issues']]}
        runtime=FakeRuntime(self.config,responder)
        with patch('shape_integration.fitting.control.check_object',side_effect=checker):
            result=self.run_object(runtime)
        self.assertEqual(result['state'],'accepted_by_model')
        self.assertEqual(result['context_rounds'],1);self.assertEqual(result['corrections'],1)
        self.assertEqual([t for t,_ in runtime.events],['observe','observe','plan','observe'])
        self.assertEqual(result['history'][1]['geometry_guard']['reason'],'repeated_source_conflict')

    def test_no_progress_stops_before_another_critic_and_preserves_status(self):
        self.options.update(max_revisions=3, geometry_guard_enabled=True)
        def responder(tag, prompt, count):
            return observed('G1') if tag == 'observe' else reviewed(prompt, 'revise')
        runtime = FakeRuntime(self.config, responder)
        result = self.run_object(runtime)
        self.assertEqual([t for t, _ in runtime.events], ['observe', 'check', 'observe'])
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['stop_reason'], 'geometry_stalled')
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(self.run_object(runtime), result)

    def test_cycle_stops_without_reopening_the_old_critic_loop(self):
        self.options.update(max_revisions=4, geometry_guard_enabled=True)
        observations = 0
        def responder(tag, prompt, count):
            nonlocal observations
            if tag == 'observe':
                raw = observed('G1')
                offset = 30 if observations % 2 else 0
                for point in raw['points']:
                    point['xy'][0] += offset
                observations += 1
                return raw
            return reviewed(prompt, 'revise')
        runtime = FakeRuntime(self.config, responder)
        result = self.run_object(runtime)
        self.assertEqual(observations, 3)
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['stop_reason'], 'geometry_stalled')

    def test_duplicate_writer_is_rejected_before_any_model_call(self):
        folder = self.folder/'objects/G1'; folder.mkdir(parents=True)
        runtime = FakeRuntime(self.config)
        with (folder/'.object.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another worker owns'):
                self.run_object(runtime)
        self.assertFalse(runtime.events)
        self.assertEqual(self.run_object(runtime)['state'], 'accepted_by_model')

    def test_real_pixel_veto_runs_before_critic_and_preserves_previous_revision(self):
        from shape_integration.fitting.frames import FitFrame
        # Isolate the regression veto: the source fitter would now repair this
        # deliberately displaced rectangle before it can reach the guard.
        self.config.setdefault('fitting',{})['source_frame_refinement']=False
        self.options.update(max_revisions=3, geometry_guard_enabled=True)
        self.candidate.update(kind='rectangle', label='panel')
        box = self.candidate['bbox_2d']
        with Image.open(self.source) as im:
            source = im.convert('RGB')
        ImageDraw.Draw(source).rectangle(box, outline='#285b91', width=3)
        source.save(self.source)
        mask = Image.new('L', source.size); ImageDraw.Draw(mask).rectangle(box, outline=255, width=3)
        mask_path = self.work/'mask.png'; mask.save(mask_path)
        group = {'id': 'support', 'representative': {'bbox': list(box), 'category': 'rectangle', 'score': .9,
            'mask_path': str(mask_path), 'mask_quality_ok': True, 'frame': {'box': [0, 0, *source.size], 'size': list(source.size)}}}
        provider = GuidedSamEvidence([group]); provider.set_inventory([self.candidate])
        observations = 0
        def responder(tag, prompt, count):
            nonlocal observations
            if tag == 'check':
                return reviewed(prompt, 'revise')
            folder = self.folder/'objects/G1'/f'revision_{observations:02d}'
            frame = FitFrame.from_metadata(read(folder/'frame.json'))
            x0, y0, x1, y1 = box
            x1 += observations*15  # Keep the deliberately wrong edit inside the prepared crop.
            points = [frame.to_local(p) for p in [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]
            observations += 1
            return {'object_id': 'G1', 'status': 'ok', 'reason': 'visible panel', 'uncertain': [],
                    'points': [{'id': str(i), 'xy': p} for i, p in enumerate(points)],
                    'parts': [{'id': 'box', 'kind': 'rectangle', 'point_ids': ['0', '1', '2', '3']}]}
        runtime = FakeRuntime(self.config, responder); runtime.spatial_evidence = provider
        result = self.run_object(runtime)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check', 'observe'])
        self.assertEqual(result['stop_reason'], 'geometry_regression_blocked')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(read(self.folder/'objects/G1/revision_01/geometry_guard.json')['decision'], 'rollback')

    def test_pipeline_does_not_publish_an_error_into_another_workers_object(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from shape_integration.common import save
        from shape_integration.pipeline import run_case
        from shape_integration.fitting.control import ObjectOwnershipError
        root = self.work/'concurrent'; target = root/'cases/case/fitting/objects/G1'
        target.mkdir(parents=True); sentinel = target/'owner.json'; sentinel.write_text('{"owner": "other worker"}')
        runtime = FakeRuntime(self.config)
        def detect(run, item, workspace, shared_runtime):
            save(run/'cases/case/geometry.json', {'image_size': item['source_size'],
                'coordinate_system': 'image_pixel_centers', 'objects': [self.candidate]})
            return {'image_coverage_status': 'complete', 'errors': {}, 'local_failures': 0}
        with (target/'.object.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=1) as pool, patch('shape_integration.batch.case', side_effect=detect):
                with self.assertRaises(ObjectOwnershipError):
                    run_case(self.work, root, self.item, runtime, self.config, pool, None, None, False)
        self.assertEqual(sentinel.read_text(), '{"owner": "other worker"}')
        self.assertFalse((target/'runner_error.json').exists())
        self.assertFalse(runtime.events)


class EntrySourceTests(unittest.TestCase):
    def test_direct_entry_uses_this_project_and_rejects_invalid_source(self):
        import os
        import subprocess
        code = ("import runpy; from pathlib import Path; "
                "runpy.run_path('experiment/shape_sam_guided/run.py', run_name='import_only'); "
                "import shape_integration.fitting.control as c; "
                "assert Path(c.__file__).resolve() == Path('src/shape_integration/fitting/control.py').resolve()")
        environment = {**os.environ, 'PYTHONPATH': str(ROOT/'.runtime/shape_layers_py311')}
        environment.pop('SHAPE_REFERENCE_ROOT', None)
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        environment['SHAPE_REFERENCE_ROOT'] = '/nonexistent/shape_reference'
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=45)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid SHAPE_REFERENCE_ROOT', result.stderr)


if __name__ == '__main__':
    unittest.main()
