import copy,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]
import numpy as np
from PIL import Image
from lxml import etree
import evidence,scene_snapshot,style_contract,style_pipeline,observation_identity,semantic_geometry
from lib import fit,native


def line():
    return dict(id='G1:R1/P1',parent_id='G1:T1',kind='straight',start=[20.,40.],via=[],end=[180.,40.],
                beziers=[],heads=['none','triangle'],start_head=False,end_head=True,end_head_type='triangle',
                coordinate_source='glm_segmented_points',node_ids=['n1','n2'],
                style=dict(color='#123456',width_px=4.,dash=dict(type='solid',preset='solid'),start=None,end=dict(type='triangle',width='lg',length='sm')))


class Contracts(unittest.TestCase):
    def test_canonical_handoff_without_labels(self):
        rows=[{'rows':[line()]}];before=copy.deepcopy(rows)
        a=scene_snapshot.canonical_rows(rows)
        self.assertEqual(a[0]['label'],'A1');self.assertEqual(rows,before)
        with self.assertRaises(ValueError):scene_snapshot.canonical_rows(rows+rows)

    def test_style_keeps_physical_end_order_and_geometry(self):
        row=line();row['label']='A7'
        answer={'styles':[dict(id=row['id'],color='#abcdef',width_px=5,dash_px=[12,6,2,6],
                              start=dict(type='arrow',width='sm',length='lg'),end=None)]}
        style_contract.validate(answer,[row])
        new=style_pipeline.apply_styles([row],answer)
        self.assertEqual(style_contract.geometry_signature(new),style_contract.geometry_signature([row]))
        self.assertEqual(new[0]['heads'],['arrow','none'])
        self.assertEqual(new[0]['style']['dash']['pattern_px'],[12,6,2,6])

    def test_dimensions_scale_once_and_roundtrip(self):
        row=line();row['style']['dash']['pattern_px']=[20.,8.,3.,8.]
        view=evidence.identity_view((200,100))
        view['view_to_global']=[[.2,0,10],[0,.2,20],[0,0,1]]
        view['global_to_view']=np.linalg.inv(view['view_to_global']).tolist()
        global_row=fit.transform(row,view);self.assertAlmostEqual(global_row['style']['width_px'],.8)
        self.assertEqual(global_row['style']['dash']['pattern_px'],[4.,1.6,.6000000000000001,1.6])
        restored=fit.transform(global_row,view,False)
        np.testing.assert_allclose(restored['style']['dash']['pattern_px'],row['style']['dash']['pattern_px'])
        np.testing.assert_allclose(restored['start'],row['start'])
        self.assertEqual(restored['style']['end'],row['style']['end'])

    def test_invalid_style_cannot_delete_or_move_path(self):
        value=dict(id='wrong',color='#123456',width_px=2,dash_px=[],start=None,end=None)
        with self.assertRaises(ValueError):style_contract.validate({'styles':[value]},[line()])
        value['id']=line()['id'];value['width_px']=float('nan')
        with self.assertRaises(ValueError):style_contract.validate({'styles':[value]},[line()])

    def test_native_sizes_dash_and_pending_geometry_export(self):
        row=line();row['style']['dash']=dict(type='dashed',preset='dash',pattern_px=[12,6,2,6])
        pending=line();pending['id']='pending';pending['style']['dash']=dict(type='unknown',preset=None)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.pptx';native.write((200,100),[row,pending],[row['style'],pending['style']],path)
            with zipfile.ZipFile(path) as z:xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
            ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
            self.assertEqual(xml.xpath('//a:tailEnd/@w',namespaces=ns)[0],'lg')
            self.assertEqual(xml.xpath('//a:tailEnd/@len',namespaces=ns)[0],'sm')
            self.assertEqual(len(xml.xpath('//a:custDash/a:ds',namespaces=ns)),2)
            audit=json.loads(path.with_suffix('.export.json').read_text())
            self.assertEqual(audit['native_shapes'],2);self.assertEqual(audit['style_pending'],['pending'])

    def test_critic_failure_retains_estimated_style_and_geometry(self):
        row=line();geometry=style_contract.geometry_signature([row])
        class Model:
            def ask(self,folder,prompt,images,validator):
                if folder.name=='critic':raise RuntimeError('simulated transport failure')
                return validator({'styles':[dict(id=row['id'],color='#ff0000',width_px=7,dash_px=[6,4],start=None,end=None)]})
        def render(rows,out,*args):
            out.mkdir(parents=True,exist_ok=True);p=out/'render.png';Image.new('RGB',(200,100),'white').save(p);return p
        with tempfile.TemporaryDirectory() as d,patch.object(style_pipeline,'render',render):
            result,issues=style_pipeline.recover(Image.new('RGB',(200,100)),[row],{},Path(d),{'style_refine_rounds':1},Model(),{})
            self.assertEqual(style_contract.geometry_signature(result),geometry)
            self.assertEqual(result[0]['style']['color'],'#FF0000')
            self.assertTrue(any(x['stage']=='style_revision' for x in issues))
            self.assertTrue((Path(d)/'final/render.png').exists())

    def test_review_saved_style_then_refine_with_corresponding_render(self):
        row=line();calls=[]
        class Model:
            def ask(self,folder,prompt,images,validator):
                calls.append((folder.parent.name,folder.name))
                if folder.name=='critic':
                    verdict='revise' if folder.parent.name=='rev_00' else 'accept'
                    return validator({'checks':[dict(id=row['id'],verdict=verdict,reason='Shaft is too thick',geometry_action='keep')]})
                self_test.assertTrue(any('previous revision' in label for label,_ in images))
                return validator({'styles':[dict(id=row['id'],color='#123456',width_px=2,dash_px=[],start=None,end=None)]})
        self_test=self
        def render(rows,out,*args):
            out.mkdir(parents=True,exist_ok=True);p=out/'render.png';Image.new('RGB',(200,100),'white').save(p);return p
        with tempfile.TemporaryDirectory() as d,patch.object(style_pipeline,'render',render):
            result,issues=style_pipeline.recover(Image.new('RGB',(200,100)),[row],{},Path(d),
                {'style_refine_rounds':1,'style_entry':'review_existing'},Model(),{})
            self.assertEqual(calls,[('rev_00','critic'),('rev_01','estimate'),('rev_01','critic')])
            self.assertEqual(result[0]['style']['width_px'],2)
            self.assertEqual(issues,[])

    def test_observation_sources_do_not_become_pixel_attribution(self):
        raw=dict(nodes=[],routes=[dict(id='R1',kind='polyline',geometry='glm',masks=[],locator='visible fan')],status='complete')
        a=observation_identity.answer(raw,['G1','G2'])
        value=semantic_geometry.validate(a,(100,100),['G1','G2'],{},[])
        self.assertEqual(value['routes'][0]['source_linking'],'observation_context_only')
        self.assertEqual(value['routes'][0]['masks'],[])
        self.assertEqual(raw['routes'][0].get('sources'),None)

    def test_only_model_can_select_mask_shape_and_style_is_reestimated(self):
        row=line();row.update(kind='block_arrow',heads=['none','none'],start=[10,10],via=[[60,10]],end=[40,60],
            shape_parts=[dict(id='shape',commands=[['M',10,10],['L',60,10],['L',40,60],['Z']])],
            mask_shape_parts=[dict(id='outline',commands=[['M',12,10],['L',62,10],['L',42,60],['Z']])])
        original=copy.deepcopy(row)
        class Model:
            def ask(self,folder,prompt,images,validator):
                if folder.name=='critic':
                    first=folder.parent.name=='rev_00'
                    return validator({'checks':[dict(id=row['id'],verdict='geometry_issue' if first else 'accept',
                        reason='The supplied mask follows the visible outline',geometry_action='mask' if first else 'keep')]})
                ids=prompt['targets'][0]['part_ids']
                return validator({'styles':[dict(id=row['id'],parts=[dict(id=i,fill='#FF0000',stroke='#000000',width_px=2,dash='solid') for i in ids])]})
        def render(rows,out,*args):
            out.mkdir(parents=True,exist_ok=True);p=out/'render.png';Image.new('RGB',(200,100),'white').save(p);return p
        with tempfile.TemporaryDirectory() as d,patch.object(style_pipeline,'render',render):
            result,issues=style_pipeline.recover(Image.new('RGB',(200,100)),[row],{},Path(d),
                {'style_refine_rounds':1},Model(),{})
            self.assertEqual(row,original)
            self.assertEqual(result[0]['geometry_method'],'model_selected_mask_shape')
            self.assertEqual(set(result[0]['style']['shape_parts']),{'outline'})
            self.assertEqual(result[0]['style']['start'],None)
            self.assertEqual(issues,[])
            self.assertTrue((Path(d)/'rev_00/mask_geometry_input/geometry.json').exists())

    def test_sam_cache_only_missing_evidence_does_not_call_provider(self):
        import local_sampling,threading
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'input').mkdir()
            crop=Image.new('RGB',(100,100),'white');crop.save(out/'input/clean.png')
            with patch.object(local_sampling.SamClient,'segment',side_effect=AssertionError('Unexpected SAM request')):
                selected,all_masks=local_sampling.sample(crop,evidence.identity_view(crop.size),[],[],{},'G1',out,
                    {'sam_evidence_mode':'cache_only'},threading.Semaphore(1))
            self.assertEqual((selected,all_masks),([],[]))
            self.assertEqual(json.loads((out/'sam_summary.json').read_text())['state'],'no_matching_cache')


if __name__=='__main__':unittest.main()
