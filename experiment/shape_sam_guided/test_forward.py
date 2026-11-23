import copy
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch, Mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from core import read, save, sha
from shape_integration.sam_evidence import SamEvidence, project_mask, local_box
from shape_integration.fitting.local import prepare_candidate, observe_and_fit


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.mask=Image.new('L',(120,80));ImageDraw.Draw(self.mask).rectangle((20,10,80,60),fill=255)
        self.mask.save(self.root/'mask.png')
        self.pred={'bbox':[20.,10.,80.,60.],'category':'rectangle','score':.75,
                   'mask_quality_ok':True,'mask_path':str(self.root/'mask.png'),
                   'frame':{'box':[0,0,120,80],'size':[120,80]}}
        self.group={'id':'S0001','representative':self.pred}
        self.source=Image.new('RGB',(120,80),'white');self.source.save(self.root/'source.png')

    def tearDown(self):self.tmp.cleanup()

    def test_global_response_parser_keeps_a_complete_array_root(self):
        from runtime import parse_response
        rows=[{'id':'a','bbox_2d':[1,2,3,4]},{'id':'b','bbox_2d':[5,6,7,8]}]
        result=parse_response('```json\n'+json.dumps(rows)+'\n```','global')
        self.assertEqual(result['objects'],rows)
        self.assertTrue(result['more_pending'])
        self.assertEqual(parse_response('```json\n{"ok":true}\n```','review'),{'ok':True})

    def test_request_replay_requires_same_prompt_image_and_budget(self):
        import threading
        from runtime import ForwardRuntime
        class Models:
            config = {'glm_max_tokens': 1000, 'seed': 42, 'glm_url': 'http://unused', 'glm_model': 'fixture'}
            slots = threading.Semaphore(1)
            count = 0
            def post(inner, *args, **kwargs):
                inner.count += 1
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}],
                        'usage': {'total_tokens': 100}}
        models = Models()
        reference, output = self.root/'reference', self.root/'output'
        pipe = {'shape_schema_retries': 0}
        args = ('observe', 'system', {'task': 'original'}, [('source', self.root/'source.png')], lambda x: x, 1000)
        ForwardRuntime(models, pipe, self.root).call(reference/'cases/C/calls', *args)
        models.config.update(reuse_request_root=str(reference), reuse_output_root=str(output))
        runtime = ForwardRuntime(models, pipe, self.root)
        runtime.call(output/'cases/C/calls', *args)
        self.assertEqual(models.count, 1)
        self.assertTrue((output/'cases/C/calls/observe_00/replay.json').exists())
        transitive = self.root/'transitive'
        models.config.update(reuse_request_roots=[str(self.root/'missing'), str(output)],
                             reuse_output_root=str(transitive))
        runtime.call(transitive/'cases/C/calls', *args)
        self.assertEqual(models.count, 1)
        self.assertEqual(read(transitive/'cases/C/calls/observe_00/replay.json')['reference_usage']['total_tokens'], 100)
        # New output directory, same corresponding reference, changed budget.
        other = self.root/'other'
        models.config['reuse_output_root'] = str(other)
        runtime.call(other/'cases/C/calls', *args[:-1], 999)
        self.assertEqual(models.count, 2)
        self.assertFalse((other/'cases/C/calls/observe_00/replay.json').exists())

    def test_crop_and_resize_mask_matches_direct_pixels(self):
        crop=[10,5,110,75];size=[400,210]
        expected=self.mask.crop(crop).resize(size,Image.Resampling.NEAREST)
        np.testing.assert_array_equal(project_mask(self.pred,crop,size),expected)
        self.assertEqual(local_box([20,10,80,60],crop,size),[41.5,16.,281.5,166.])

    def test_controlled_replay_preserves_completed_schema_failure_and_retry(self):
        import threading
        from runtime import ForwardRuntime
        class Model:
            config = {'glm_max_tokens': 1000, 'seed': 42, 'glm_url': 'http://unused'}
            slots = threading.Semaphore(1)
            calls = 0
            def post(inner, *args, **kwargs):
                inner.calls += 1
                text = '{"ok":false}' if inner.calls == 1 else '{"ok":true}'
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                        'usage': {'total_tokens': 100}}
        def validator(raw):
            if not raw['ok']:
                raise ValueError('Fixture schema failure')
            return raw
        model = Model(); base, output = self.root/'base', self.root/'output'
        args = ('observe', 'system', {}, [], validator, 1000)
        def retry(folder, error):
            return {'system': 'system', 'prompt': {'previous': (folder/'content.txt').read_text()},
                    'images': [], 'validator': validator}
        ForwardRuntime(model, {'shape_schema_retries': 1}, self.root).call(base/'case', *args, retry_builder=retry)
        model.config.update(reuse_request_root=str(base), reuse_output_root=str(output), replay_schema_failures=True)
        result = ForwardRuntime(model, {'shape_schema_retries': 1}, self.root).call(output/'case', *args, retry_builder=retry)
        self.assertTrue(result['ok'])
        self.assertEqual(model.calls, 2)
        self.assertTrue(read(output/'case/observe_00/replay.json')['schema_failure_replayed'])
        self.assertTrue((output/'case/observe_01/replay.json').exists())
        third = self.root/'third'
        model.config['reuse_request_map'] = {str(third): [str(output)]}
        result = ForwardRuntime(model, {'shape_schema_retries': 1}, self.root).call(third/'case', *args, retry_builder=retry)
        self.assertTrue(result['ok'])
        self.assertEqual(model.calls, 2)
        self.assertTrue(read(third/'case/observe_00/replay.json')['schema_failure_replayed'])

    def test_mask_transport_from_offset_crop(self):
        pred=copy.deepcopy(self.pred);pred['frame']={'box':[20,30,260,190],'size':[120,80]}
        np.testing.assert_array_equal(project_mask(pred,[20,30,260,190],[120,80]),self.mask)

    def test_assets_suppress_interior_but_not_enclosing_container(self):
        evidence=SamEvidence([self.group],[{'bbox_2d':[10,0,100,70]}])
        self.assertFalse(evidence.eligible(self.group))
        bigger=copy.deepcopy(self.group);bigger['representative']['bbox']=[0,0,120,80]
        self.assertTrue(evidence.eligible(bigger))

    def test_child_mask_does_not_replace_container_evidence(self):
        child=copy.deepcopy(self.group);child['representative']['bbox']=[30,20,40,30]
        evidence=SamEvidence([child])
        self.assertEqual(evidence.select_object({'id':'O','bbox_2d':[10,0,110,75]}),[])

    def test_region_does_not_mutate_prompt_or_original_candidates(self):
        evidence=SamEvidence([self.group]);prompt={'task':'original'};images=[('clean',self.root/'source.png')]
        candidate={'id':'G1','bbox_2d':[20,10,80,60]}
        actual,views=evidence.region(prompt,images,{'source_crop_half_open':[0,0,120,80]},
                                    {'candidates':[candidate]},self.root)
        self.assertEqual(prompt,{'task':'original'});self.assertEqual(len(images),1)
        self.assertEqual(len(views),2);self.assertIn('sam_auxiliary_evidence',actual)
        self.assertEqual(candidate['bbox_2d'],[20,10,80,60])

    def test_incomplete_mask_does_not_add_redundant_object_image(self):
        self.pred['mask_quality_ok']=False
        evidence=SamEvidence([self.group]);folder=self.root/'obj'
        job=prepare_candidate(self.source,{'id':'G1','label':'box','kind':'rectangle','bbox_2d':[20,10,80,60]},[120,80],folder)
        prompt,images=evidence.observe({},[('clean',folder/'clean.png')],job,folder)
        self.assertEqual(read(folder/'sam_evidence.json')['shown'],[])
        self.assertEqual(len(images),1);self.assertEqual(prompt,{})

    def test_low_resolution_masks_remain_regional_bbox_candidates(self):
        self.pred['frame']['box']=[0,0,2400,1600]
        evidence=SamEvidence([self.group])
        candidate={'id':'G1','bbox_2d':self.pred['bbox']}
        self.assertEqual(evidence.select_object(candidate),[])
        self.assertEqual(evidence.select_region([0,0,120,80],[candidate])[1],1)

    def test_real_observation_hook_and_fitter_preserve_compound_parts(self):
        evidence=SamEvidence([self.group]);folder=self.root/'obj'
        job=prepare_candidate(self.source,{'id':'G1','label':'box pair','kind':'compound_symbol','bbox_2d':[20,10,80,60]},[120,80],folder)
        calls=[]
        class Fake:
            spatial_evidence=evidence
            def call(self,root,tag,system,prompt,images,validator,budget):
                calls.append((prompt,images))
                raw={'status':'ok','points':[{'id':f'p{i}','xy':xy} for i,xy in enumerate([
                     [200,200],[500,200],[500,500],[200,500],
                     [600,200],[900,200],[900,500],[600,500]])],
                     'parts':[{'id':'left','kind':'rectangle','point_ids':['p0','p1','p2','p3']},
                              {'id':'right','kind':'rectangle','point_ids':['p4','p5','p6','p7']}]}
                result=validator(raw);save(Path(root)/'observe_00/result.json',raw);return result
        status=observe_and_fit(job,folder,Fake(),namespace='test',max_tokens=1000)
        self.assertEqual(status['part_counts']['succeeded'],2)
        self.assertEqual(len(calls[0][1]),3)
        self.assertIn('sam_auxiliary_evidence',calls[0][0])
        self.assertNotIn('sam_auxiliary_evidence',read(folder/'prompt.json')['prompt'])

    def test_complete_paired_forward_and_resume_with_stubbed_model(self):
        import run
        from runtime import ForwardRuntime
        from shape_integration.fitting.frames import FitFrame
        base=self.root/'base';old=base/'runs/frozen/cases/fixture';output=self.root/'run'
        source=base/'inputs/original.png';source.parent.mkdir(parents=True)
        self.source.save(source)
        item={'id':'fixture','original':'inputs/original.png','input':'inputs/original.png',
              'source_size':[120,80],'input_size':[120,80],'source_sha256':sha(source),'input_sha256':sha(source)}
        save(old/'shapes/global_raw.json',{'objects':[{'id':'box','label':'box','kind':'rectangle','bbox_2d':[20,10,80,60]}],
                                         'uncertain':[],'more_pending':False})
        save(old/'shapes/plan.json',{'regions':[{'id':'R1','bbox_2d':[0,0,119,79],'description':'whole'}],'uncertain':[]})
        save(old/'dedup/image_boxes_original.json',{'assets':[{'id':'A1','bbox_2d':[50,10,90,60]}]})
        save(old/'summary.json',{'image_coverage_status':'complete'})
        pipe={'global_max_tokens':1000,'plan_max_tokens':1000,'local_max_tokens':1000,'crop_padding':.05,
              'local_workers':1,'iou_threshold':.85,'model_conflict_review':True,'shape_schema_retries':1,
              'image_conflict_review':{'candidate_iou_threshold':.4,'batch_size':32,'max_tokens':1000},
              'fitting':{'crop_padding':.2,'crop_scale':3,'crop_min_long_edge':1080,'crop_max_long_edge':2048,
                         'max_tokens':1000,'check_max_tokens':1000,'plan_max_tokens':1000,'coverage_max_tokens':1000,
                         'max_revisions':1,'max_context_rounds':1,'coverage_rounds':1,'check_enabled':True,
                         'max_new_candidates_per_round':8}}
        config={'baseline_root':str(base),'baseline_run':'frozen','evidence':{}}
        index={'groups':[self.group],'elapsed_seconds':.1}
        calls=[];fail_critic=False
        class Stub(ForwardRuntime):
            def call(self,root,tag,system,prompt,images,validator,budget,**kwargs):
                if tag in {'global','layout'}:return super().call(root,tag,system,prompt,images,validator,budget,**kwargs)
                calls.append((tag,'sam_auxiliary_evidence' in prompt))
                if tag.startswith('review_'):
                    raw={'reviews':[{'id':c['id'],'action':'keep','bbox_2d':None,'reason':'visible box'}
                                    for c in prompt['candidates'] if c['role']=='assigned'],
                         'new_objects':[],'uncertain':[],'more_pending':False}
                elif tag.startswith('batch_'):
                    raw={'decisions':[{'shape_id':c['shape_id'],'decision':'keep','image_id':None,
                         'reason':'The visible geometric object is distinct from the overlapping image asset.'}
                         for c in prompt['candidates']]}
                elif tag=='observe':
                    job=read(Path(root).parent/'job.json');frame=FitFrame.from_metadata(job['frame'])
                    points=[frame.to_local(p) for p in [[20,10],[80,10],[80,60],[20,60]]]
                    raw={'status':'ok','points':[{'id':f'p{i}','xy':p} for i,p in enumerate(points)],
                         'parts':[{'id':'box','kind':'rectangle','point_ids':['p0','p1','p2','p3']}]}
                elif tag=='check':
                    if fail_critic:
                        from request_io import ModelTransportError
                        raise ModelTransportError('Fixture: critic service unavailable')
                    raw={'decision':'accept','issues':[]}
                elif tag=='coverage':raw={'status':'complete','new_objects':[],'possible_duplicates':[],'uncertain':[]}
                else:raise AssertionError('Unexpected model call '+tag)
                result=validator(raw);save(Path(root)/(tag+'_00')/'result.json',raw);return result
        for arm in ['glm_control','sam_forward']:save(output/arm/'localization/manifest.json',{'images':[item]})
        with patch.object(run,'ForwardRuntime',Stub),ThreadPoolExecutor(max_workers=1) as pool:
            for arm in ['glm_control','sam_forward']:
                result=run.case_arm(item,arm,output,config,pipe,None,index,pool)
                self.assertEqual(result['summary']['successful_parts'],1)
                before=len(calls)
                self.assertEqual(result,run.case_arm(item,arm,output,config,pipe,None,index,pool))
                self.assertEqual(before,len(calls))
            # A failed critic must leave already fitted shape geometry usable.
            fail_critic=True;partial=self.root/'critic_failure_run'
            save(partial/'sam_forward/localization/manifest.json',{'images':[item]})
            retained=run.case_arm(item,'sam_forward',partial,config,pipe,None,index,pool)
            self.assertEqual(retained['summary']['successful_parts'],1)
            self.assertEqual(retained['summary']['unresolved_objects'],1)
            fitted=read(partial/'sam_forward/cases/fixture/fitting/scene.json')
            self.assertTrue(all(obj['selected_revision'] for obj in fitted['objects'].values()))
        self.assertIn(('observe',True),calls);self.assertIn(('observe',False),calls)
        self.assertTrue(any(tag.startswith('batch_') for tag, _ in calls))
        self.assertTrue(all(not auxiliary for tag,auxiliary in calls if tag in {'check','coverage'}))



class ReliabilityTests(unittest.TestCase):
    def test_transient_http_error_retries_identical_request(self):
        from request_io import request_json
        opener=Mock()
        opener.open.side_effect=[urllib.error.HTTPError('http://local',503,'busy',{},None),
                                 io.BytesIO(b'{"ok":true}')]
        with tempfile.TemporaryDirectory() as d, patch('request_io.time.sleep') as sleep:
            trace=Path(d)/'trace.json'
            self.assertEqual(request_json(opener,'http://local',{'unchanged':True},trace_path=trace),{'ok':True})
            self.assertEqual(opener.open.call_count,2)
            self.assertIs(opener.open.call_args_list[0].args[0],opener.open.call_args_list[1].args[0])
            self.assertEqual(read(trace)['attempts'][0]['http_status'],503)
            sleep.assert_called_once()

    def test_permanent_http_error_and_retry_exhaustion_are_bounded(self):
        from request_io import request_json, ModelTransportError
        with patch('request_io.time.sleep'):
            for failure,expected in [(urllib.error.HTTPError('http://local',400,'invalid',{},None),1),
                                     (urllib.error.URLError('disconnected'),3)]:
                opener=Mock();opener.open.side_effect=failure
                with self.assertRaises(ModelTransportError):request_json(opener,'http://local',{})
                self.assertEqual(opener.open.call_count,expected)

    def test_transport_error_does_not_poison_forward_schema_cache(self):
        from request_io import ModelTransportError
        from runtime import ForwardRuntime
        model=Mock();model.config={'glm_max_tokens':100,'seed':42,'glm_url':'http://local'}
        model.slots=__import__('threading').BoundedSemaphore(1)
        model.post.side_effect=[ModelTransportError('temporary failure'),
            {'choices':[{'finish_reason':'stop','message':{'content':'{"ok": true}'}}]}]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);runtime=ForwardRuntime(model,{'shape_schema_retries':1},folder)
            args=(folder,'observe','system',{'task':'unchanged'},[],lambda x:x,100)
            with self.assertRaises(ModelTransportError):runtime.call(*args)
            self.assertFalse((folder/'observe_00/error_0.json').exists())
            self.assertEqual(runtime.call(*args),{'ok':True})
            self.assertEqual(model.post.call_args_list[0].args,model.post.call_args_list[1].args)
            self.assertFalse((folder/'observe_01').exists())

    def test_transport_timeout_budget_prevents_another_attempt(self):
        from request_io import request_json, ModelTransportError
        opener=Mock();opener.open.side_effect=urllib.error.URLError('slow disconnect')
        with patch('request_io.time.monotonic',side_effect=[0.,0.,1199.5,1199.5]), \
             patch('request_io.time.sleep') as sleep:
            with self.assertRaises(ModelTransportError):request_json(opener,'http://local',{})
        self.assertEqual(opener.open.call_count,1);sleep.assert_not_called()

    def test_evaluation_failure_keeps_other_progress_and_records_error(self):
        import report
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'config.json'
            save(config,{'glm_concurrency':2});save(root/'manifest.json',{'images':[{'id':x} for x in ['a','b','c']]})
            def evaluate(root,item,*unused):
                if item['id']=='b':raise RuntimeError('fixture failure')
                return {'id':item['id']}
            with patch.object(sys,'argv',['report','--run-dir',str(root),'--config',str(config)]), \
                 patch.object(report,'Models'),patch.object(report,'case_report',side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError,'1 evaluations failed'):report.main()
            self.assertEqual(read(root/'evaluation_progress.json')['completed'],2)
            self.assertEqual(read(root/'evaluation_progress.json')['failed'],1)
            self.assertEqual(read(root/'evaluation_errors.json')[0]['id'],'b')
            self.assertFalse((root/'aggregate.json').exists())

    def test_worker_detects_service_death_during_evaluation(self):
        from worker import wait_workload
        child=Mock();child.poll.return_value=None
        service=Mock();service.poll.side_effect=[None,1]
        with tempfile.TemporaryDirectory() as d,patch('worker.time.sleep'):
            root=Path(d)
            with self.assertRaisesRegex(RuntimeError,'paired_evaluation'):
                wait_workload(child,[service],root,{'phase':'paired_evaluation','active_code':'fixture'})
            self.assertIn('heartbeat_at',read(root/'status.json'))

    def test_resume_progress_counts_existing_receipts_and_partial_pairs(self):
        from worker import completion_counts
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save(root/'manifest.json',{'images':[{'id':x} for x in ['a','b','c']]})
            for arm,ident in [('glm_control','a'),('sam_forward','a'),('sam_forward','b')]:
                save(root/arm/'cases'/ident/'forward_complete.json',{})
            self.assertEqual(completion_counts(root),{'forward_pairs_complete':1,
                'forward_arms_complete':3,'forward_pairs_expected':3})


if __name__=='__main__':unittest.main()
