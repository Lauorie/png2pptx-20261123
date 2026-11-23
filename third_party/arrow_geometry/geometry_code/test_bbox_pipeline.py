import copy,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import bbox_geometry as bg,bbox_pipeline as bp,bbox_inputs as bi,reconcile_stage as rs
from lib import fit
from transport.common import write_json as write,read_json as read,sha256
BASE={'bbox_min_extent':25,'bbox_display_side':400,'bbox_context_scale':2.,'bbox_max_expansions':1,
      'context_padding':12,'context_side':250,'geometry_critic_enabled':False,'geometry_preview_width_original_px':1.2,
      'coverage_rounds':0,'scope_review':False,'pipeline_mode':'geometry','reconcile_enabled':True,
      'task_workers':2,'bbox_block_mask_evidence':False,'bbox_block_sam_fallback':True,'native_render':True}
def locks():return {'cpu':threading.Semaphore(2),'sam':threading.Semaphore(1)}
def obj(id='A'):
    return dict(id=id,target_id=id,bbox_native=[20.,20.,80.,60.],locator='visible arrow',readonly=[])
class Model:
    def __init__(self,mode='draw'):self.mode=mode;self.calls=[]
    def ask(self,path,prompt,images,validate):
        self.calls.append((str(path),prompt));mode=self.mode
        if mode=='expand' and len(self.calls)==1:return validate({'status':'needs_context','reason':'target clipped'})
        if mode=='block':return validate({'status':'block','reason':'closed arrow'})
        if mode=='forbidden':raise AssertionError('prepare must never ask model')
        if 'regions' in prompt.get('output_schema',{}):return validate({'regions':[]})
        b=prompt['targets'][0]['bbox'];y=(b[1]+b[3])/2
        return validate({'status':'draw','nodes':[['a',b[0],y],['b',b[2],y]],
            'routes':[{'id':'R1','paths':[{'id':'p','heads':['none','triangle'],'segments':[{'kind':'line','nodes':['a','b']}]}]}]})
class PipelineContract(unittest.TestCase):
    def test_prepare_has_no_model_or_sam(self):
        with tempfile.TemporaryDirectory() as t,patch('bbox_geometry.block_mask',side_effect=AssertionError('no SAM')):
            r=bg.recover_target(Image.new('RGB',(100,80),'white'),obj(),Path(t),Model('forbidden'),BASE,locks(),True)
            self.assertEqual(r['state'],'prepared')
    def test_geometry_expansion_and_native_frame(self):
        with tempfile.TemporaryDirectory() as t:
            model=Model('expand');r=bg.recover_target(Image.new('RGB',(100,80),'white'),obj(),Path(t),model,BASE,locks())
            self.assertEqual(len(model.calls),2);self.assertTrue(np.allclose(r['rows'][0]['start'],[20,40],atol=.01))
            self.assertTrue(np.allclose(r['rows'][0]['end'],[80,40],atol=.01));self.assertEqual(r['rows'][0]['style']['end']['length'],'med')
    def test_block_shape_failure_uses_selected_mask_not_bbox(self):
        with tempfile.TemporaryDirectory() as t:
            def mask(crop,*args):
                m=np.zeros((crop.height,crop.width),bool);m[30:50,40:100]=True;return m,[]
            with patch('block_shapes.recover',side_effect=ValueError('shape rejected')),patch('bbox_geometry.block_mask',side_effect=mask):
                result=bg.recover_target(Image.new('RGB',(100,80),'white'),dict(obj(),render_type='block_arrow'),Path(t),Model('forbidden'),BASE,locks())
            self.assertEqual(result['rows'][0]['kind'],'block_arrow');self.assertTrue(result['rows'][0]['mask_shape_parts'])
            self.assertEqual(result['rows'][0]['geometry_method'],'model_selected_mask_shape')
    def test_block_failure_without_mask_does_not_draw_locator_rectangle(self):
        with tempfile.TemporaryDirectory() as t,patch('block_shapes.recover',side_effect=ValueError('shape rejected')),patch('bbox_geometry.block_mask',return_value=(None,[])):
            with self.assertRaises(ValueError):bg.recover_target(Image.new('RGB',(100,80),'white'),dict(obj(),render_type='block_arrow'),Path(t),Model('forbidden'),BASE,locks())
    def fixture(self,root):
        source=root/'source';run=root/'run';run.mkdir();(source/'inputs/x').mkdir(parents=True);(source/'cases/x').mkdir(parents=True)
        Image.new('RGB',(100,80),'white').save(source/'inputs/x/source.png')
        write(source/'inputs/x/views.json',dict(native_size=[100,80],source_sha256=sha256(source/'inputs/x/source.png')))
        write(source/'cases.json',['x']);write(source/'cases/x/finished.json',{});write(source/'cases/x/result.json',{'state':'completed'})
        write(source/'cases/x/reconstruction_handoff.json',{'case_id':'x','tasks':[dict(obj(),identity_hint='line',kind='single',readonly_identities=[])]})
        bi.freeze(run,{'bbox_discovery_run':str(source)},['x']);return run
    def test_main_dispatch_through_native_export_and_style_hook(self):
        import pipeline
        with tempfile.TemporaryDirectory() as t:
            run=self.fixture(Path(t));cfg={**BASE,'discovery_provider':'bbox_handoff','pipeline_mode':'full','coverage_rounds':1};model=Model()
            events=[]
            with patch('style_pipeline.finish_case',side_effect=lambda clean,results,*a:(results,[])) as style:
                summary=pipeline.run_case('x',cfg,run/'cases/x',model,locks(),lambda c,v:events.append(v['stage']))
            self.assertEqual(summary['draw_paths'],1);self.assertFalse(summary['mask_identity_executed'])
            self.assertEqual(len(model.calls),2);style.assert_called_once()
            self.assertEqual(read(run/'cases/x/missing_regions.json'),{'regions':[]})
            self.assertLess(events.index('reconciliation'),events.index('style_recovery'))
            self.assertTrue((run/'cases/x/native/styled.png').is_file())
            scene=read(run/'cases/x/scene.json');self.assertEqual(scene['discovery_provider'],'bbox_handoff')
            self.assertTrue((run/'cases/x/geometry_results.json').is_file())
    def test_reconcile_preserves_unedited_paths(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=Image.new('RGB',(100,80),'white')
            a=bg.recover_target(source,obj('A'),root/'a',Model(),BASE,locks())['rows'][0]
            b=copy.deepcopy(a);b.update(id='B',parent_id='B',source_discovery_id='B')
            c=copy.deepcopy(a);c.update(id='C',parent_id='C',source_discovery_id='C')
            # Simple unmodelled global geometry exercises the mask-compatible interface.
            for key in ['model_nodes','model_segments','geometry_frame']:c.pop(key,None)
            c['start']=[5,5];c['end']=[5,70];c['via']=[]
            class Reconcile:
                def ask(self,path,q,images,validate):
                    g=copy.deepcopy(q['current']);g['routes'][0]['paths']=g['routes'][0]['paths'][:1]
                    return validate({'decision':'refine','reason':'same visible line','geometry':g})
            original=[dict(group='g',rows=[a,b,c],state='completed')]
            result,issues=rs.run(source,original,root/'review',BASE,Reconcile(),locks())
            output=[r for g in result for r in g['rows']]
            self.assertEqual(len(output),2);self.assertEqual(next(r for r in output if r['id']=='C'),c)
            self.assertEqual(original[0]['rows'],[a,b,c]);self.assertFalse(issues)
if __name__=='__main__':unittest.main()

