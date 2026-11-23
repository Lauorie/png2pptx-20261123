"""Arrow semantics, coordinate units, native export, and revision input binding."""
import copy
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read,save,sha
from shape_integration.arrows import options,contracts,geometry,prompts,export,controller,model
from shape_integration.styling.export import NS,render_native,pattern
from shape_integration.transport import RequestTimeout


def network():
    return {'status':'ready','points':{'a':[30,100],'j':[100,100],'b':[170,45],'c':[180,135]},
            'paths':[{'id':'trunk','segments':[{'kind':'line','points':['a','j']}],'start_head':False,'end_head':False},
                     {'id':'up','segments':[{'kind':'curve','points':['j','b']}],'start_head':False,'end_head':True},
                     {'id':'down','segments':[{'kind':'line','points':['j','c']}],'start_head':False,'end_head':True}]}


def style(g):
    return {p['id']:{'stroke':'#173E61','width_px':2,'dash':'solid','start':{'type':'triangle','width':'med','length':'med'} if p['start_head'] else None,
                    'end':{'type':'triangle','width':'med','length':'med'} if p['end_head'] else None} for p in g['paths']}


def geometry_verdict(verdict='accept'):
    return {'off_points':[{'id':'a','visible':'source stroke lies above the fitted sample','move':'up'}] if verdict=='revise' else [],
            'endpoint_issues':[],'missing_sections':[],
            'reason':'Compared samples, endpoints and all visible branches against source.', 'verdict':verdict}


class ArrowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.opt=options({});self.opt.update(crop_min_long_edge=180,crop_max_long_edge=320)
        self.source=self.root/'source.png';Image.new('RGB',(240,180),'white').save(self.source)
        self.g=network();self.candidate={'id':'A001','description':'one branching thin arrow network','bbox_2d':[25,40,185,140]}

    def test_shared_headless_trunk_and_two_heads_are_valid(self):
        self.assertEqual(len(contracts.geometry(self.g,(240,180))['paths']),3)
        headless=copy.deepcopy(self.g)
        for path in headless['paths']:path['end_head']=False
        self.assertEqual(len(contracts.geometry(headless,(240,180))['paths']),3)

    def test_prompt_echo_envelope_is_audited_without_weakening_validation(self):
        raw={'task':'echoed instruction','output':self.g}
        valid=lambda v:contracts.geometry(v,(240,180))
        result=model.validate_response(raw,valid,self.root)
        self.assertEqual(result['points'],self.g['points'])
        self.assertEqual(read(self.root/'response_normalization.json')['operation'],'unwrap_output')
        bad=copy.deepcopy(raw);bad['output']['points']['a']=[-20,10]
        with self.assertRaisesRegex(ValueError,'outside'):model.validate_response(bad,valid,self.root)
        # A conflicting top-level response must not be silently replaced by its envelope.
        with self.assertRaises(ValueError):model.validate_response({'points':{},'paths':[],'output':self.g},valid,self.root)

    def test_bad_coordinates_and_head_flags_are_rejected(self):
        bad=copy.deepcopy(self.g);bad['points']['a']=[float('nan'),5]
        with self.assertRaises(ValueError):contracts.geometry(bad,(240,180))
        bad=copy.deepcopy(self.g);bad['points']['a']=[300,5]
        with self.assertRaises(ValueError):contracts.geometry(bad,(240,180))
        bad=copy.deepcopy(self.g);bad['paths'][1]['end_head']='true'
        with self.assertRaises(ValueError):contracts.geometry(bad,(240,180))

    def test_frame_roundtrip_and_one_style_length_scale(self):
        f=geometry.frame(self.source,[25,40,185,140],self.root/'frame',self.opt)
        g=geometry.mapped(self.g,f);back=geometry.mapped(g,f,True)
        for key in self.g['points']:np.testing.assert_allclose(back['points'][key],self.g['points'][key],atol=1e-12)
        rows=[{'id':'trunk','stroke':'#123456','width_px':7.3,'dash':'dash','dash_pattern_px':[25,12]}]
        original=geometry.source_styles(rows,f,0);local=geometry.local_styles(original,f)
        self.assertAlmostEqual(local['trunk']['width_px'],7.3);np.testing.assert_allclose(local['trunk']['dash_pattern_px'],[25,12])

    def test_curves_keep_samples_tips_and_sharp_corner_segments(self):
        g={'points':{'a':[10,10],'b':[35,18],'c':[55,60],'d':[100,80],'e':[150,80]},
           'paths':[{'id':'p0','segments':[{'kind':'curve','points':['a','b','c','d']},{'kind':'line','points':['d','e']}],'start_head':True,'end_head':True}]}
        cmd=geometry.commands(g,g['paths'][0]);self.assertEqual(cmd[0],['M',10.,10.]);self.assertEqual(cmd[-1],['L',150.,80.])
        self.assertEqual([c[-2:] for c in cmd if c[0]=='C'],[[35.,18.],[55.,60.],[100.,80.]])
        self.assertFalse(any(c[0]=='Z' for c in cmd))

    def test_native_connector_freeform_and_heads_without_raster(self):
        g=copy.deepcopy(self.g);g['points']['mid']=[140,75];g['paths'][1]['segments'][0]['points']=['j','mid','b']
        out=self.root/'arrows.pptx';counts=export.write([{'id':'A001','geometry':g,'styles':style(g)}],(240,180),out)
        self.assertEqual(counts,{'connectors':2,'freeforms':1,'heads':2,'networks':1})
        with zipfile.ZipFile(out) as z:
            xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
            self.assertEqual(len(xml.findall('.//p:cxnSp',NS)),2)
            self.assertEqual(len(xml.findall('.//a:cubicBezTo',NS)),2)
            self.assertFalse(xml.findall('.//a:close',NS));self.assertFalse(xml.findall('.//p:pic',NS))
            self.assertEqual(len(xml.xpath('.//a:tailEnd[@type="triangle"]',namespaces=NS)),2)
        render_native(out,self.root/'native_render',(240,180));self.assertTrue((self.root/'native.png').exists())

    def test_combined_ppt_preserves_base_shapes(self):
        base=self.root/'base.pptx';p=Presentation();p.slide_width=Inches(12);p.slide_height=Inches(9)
        slide=p.slides.add_slide(p.slide_layouts[6]);s=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,100,200,300,400);s.name='existing shape';p.save(base)
        digest=sha(base);before=etree.tostring(Presentation(base).slides[0].shapes[0]._element)
        export.write([{'id':'A001','geometry':self.g,'styles':style(self.g)}],(240,180),self.root/'combined.pptx',base)
        self.assertEqual(sha(base),digest);self.assertEqual(etree.tostring(Presentation(self.root/'combined.pptx').slides[0].shapes[0]._element),before)

    def test_head_enums_and_presence_are_enforced(self):
        heads=[{'id':p['id'],'start':None,'end':{'type':'arrow','width':'sm','length':'lg'} if p['end_head'] else None} for p in self.g['paths']]
        self.assertEqual(len(contracts.styles({'heads':heads},self.g['paths'],'heads')['heads']),3)
        heads[0]['start']={'type':'triangle','width':'med','length':'med'}
        with self.assertRaisesRegex(ValueError,'no observed head'):contracts.styles({'heads':heads},self.g['paths'],'heads')

    def test_only_complete_duplicate_networks_are_merged(self):
        a={'geometry':self.g};b=copy.deepcopy(a);self.assertTrue(controller.duplicate(a,b,1.5))
        b['geometry']['points']['b'][1]+=10;self.assertFalse(controller.duplicate(a,b,1.5))
        b=copy.deepcopy(a);b['geometry']['paths']=b['geometry']['paths'][:2];self.assertFalse(controller.duplicate(a,b,1.5))

    def test_discovery_uses_local_pixels_and_keeps_cross_region_scan(self):
        calls=[]
        def ask(runtime,folder,p,images,validator,opt):
            calls.append((folder,p,images))
            if folder.name=='global':return validator({'objects':[{'bbox_2d':[30,20,250,100],'description':'long cross-region arrow'}],
                 'regions':[{'bbox_2d':[0,0,159,239]},{'bbox_2d':[159,0,319,239]}]})
            if folder.name=='R01':return validator({'objects':[],'uncertain':['another route runs outside right edge']})
            return validator({'objects':[]})
        with patch('shape_integration.arrows.model.ask',side_effect=ask),ThreadPoolExecutor(max_workers=2) as pool:
            result=controller.discover(self.source,self.root/'detect',None,self.opt,pool)
        self.assertEqual(len(result['objects']),1);self.assertEqual(len(calls),4)
        self.assertIn('outside right edge',calls[-1][1]['reported_missing_context'][0])
        self.assertFalse(any('bbox_2d' in str(p.get('existing_ids')) for _,p,_ in calls))

    def test_failed_global_discovery_falls_back_to_coarse_regions(self):
        calls=[]
        def ask(rt,folder,p,images,validator,opt):
            calls.append(folder.name)
            if folder.name=='global':raise RequestTimeout('deadline')
            return validator({'objects':[],'uncertain':[]})
        with patch('shape_integration.arrows.model.ask',side_effect=ask),ThreadPoolExecutor(max_workers=4) as pool:
            result=controller.discover(self.source,self.root/'fallback',None,self.opt,pool)
        self.assertEqual(set(calls),{'global','R01','R02','R03','R04','coverage'})
        self.assertEqual(len(result['errors']),1)
        self.assertLess(read(self.root/'fallback/regions/R01/input/frame.json')['box'][2],240)

    def fake_render(self,obj,source,frame,folder,slots):
        folder.mkdir(parents=True,exist_ok=True);save(folder/'scene.json',obj)
        Image.new('RGB',frame['size'],'white').save(folder/'rendered.png');return folder/'rendered.png'

    def test_whole_network_refine_receives_matching_geometry_and_style(self):
        counts={'geometry':0,'style':0};calls=[]
        def ask(runtime,folder,p,images,validator,opt):
            calls.append((folder,p,images))
            if folder.name=='observe':
                f=read(folder.parent/'input/frame.json')
                if counts['geometry']:
                    self.assertEqual(len(p['previous_geometry']['paths']),3)
                    self.assertEqual(images[-1][0],'Previous geometry, aligned with previous_geometry text')
                    self.assertIn('a: up',p['feedback']['suggestion'])
                    self.assertIn('source stroke lies above',p['feedback']['evidence'])
                return validator(geometry.mapped(self.g,f))
            if folder.name=='critic':
                key='style' if 'current_style' in p else 'geometry';counts[key]+=1
                if key=='style':
                    saved=read(images[1][1].parent/'scene.json');f=read(self.root/'object/style/input/frame.json')
                    self.assertEqual(p['current_style'],geometry.local_styles(saved['styles'],f))
                else:
                    self.assertNotIn('geometry',p)
                    return validator(geometry_verdict('revise' if counts[key]==1 else 'accept'))
                return validator({'verdict':'revise' if counts[key]==1 else 'accept','evidence':'source difference','suggestion':'match the source'})
            if folder.name=='shaft':
                if 'feedback' in p:
                    self.assertEqual(len(p['previous_style']),3)
                    saved=read(images[-1][1].parent/'scene.json');f=read(self.root/'object/style/input/frame.json')
                    self.assertEqual(p['previous_style'],geometry.local_styles(saved['styles'],f))
                self.assertNotIn('geometry',p)
                return validator({'styles':[{'id':q['id'],'stroke':'#173E61','width_px':4,'dash':'dash','dash_pattern_px':[15,8]} for q in p['paths']]})
            if folder.name=='heads':
                f=read(self.root/'object/style/input/frame.json');saved=read(images[2][1].parent/'scene.json')
                self.assertEqual(p.get('previous_style',p.get('shaft_style')),geometry.local_styles(saved['styles'],f))
                self.assertEqual(p['image_size'],f['size']);self.assertNotIn('geometry',p)
                self.assertEqual(images[-1][0],prompts.HEAD_GUIDE_LABEL)
                self.assertTrue(images[-1][1].exists())
                return validator({'heads':[{'id':q['id'],'start':None,'end':{'type':'triangle','width':'sm','length':'med'} if q['end_head'] else None} for q in p['paths']]})
            raise AssertionError(folder)
        with patch('shape_integration.arrows.model.ask',side_effect=ask),patch('shape_integration.arrows.export.render_object',side_effect=self.fake_render):
            r=controller.fit_object(self.source,self.candidate,self.root/'object',None,self.opt,threading.BoundedSemaphore(1))
        self.assertEqual(r['state'],'completed');self.assertEqual(counts,{'geometry':2,'style':2})
        self.assertEqual(len(r['styles']),3)

    def test_geometry_critic_failure_keeps_usable_geometry(self):
        def ask(rt,folder,p,images,validator,opt):
            if folder.name=='observe':return validator(geometry.mapped(self.g,read(folder.parent/'input/frame.json')))
            raise RequestTimeout('test deadline')
        with patch('shape_integration.arrows.model.ask',side_effect=ask):
            r=controller.fit_object(self.source,self.candidate,self.root/'failed',None,self.opt,threading.BoundedSemaphore(1))
        self.assertIn('geometry',r);self.assertIn('styles',r);self.assertEqual(r['state'],'completed_with_unresolved')
        self.assertEqual(r['geometry_state'],'unresolved')

    def test_headless_connectors_skip_head_requests_and_render_without_heads(self):
        g=copy.deepcopy(self.g)
        for path in g['paths']:path['end_head']=False
        def ask(rt,folder,p,images,validator,opt):
            if folder.name=='observe':return validator(geometry.mapped(g,read(folder.parent/'input/frame.json')))
            if folder.name=='critic':return validator({'verdict':'accept'} if 'current_style' in p else geometry_verdict())
            if folder.name=='shaft':return validator({'styles':[{'id':x['id'],'stroke':'#000000','width_px':3} for x in g['paths']]})
            raise AssertionError('Unexpected head request for headless connectors')
        with patch('shape_integration.arrows.model.ask',side_effect=ask),patch('shape_integration.arrows.export.render_object',side_effect=self.fake_render):
            result=controller.fit_object(self.source,self.candidate,self.root/'headless',None,self.opt,threading.BoundedSemaphore(1))
        self.assertEqual(result['state'],'completed')
        self.assertTrue(all(s['start'] is None and s['end'] is None for s in result['styles'].values()))
        self.assertEqual(export.write([result],(240,180),self.root/'headless.pptx')['heads'],0)

    def test_style_replay_freezes_geometry_and_never_rediscovers(self):
        original=self.root/'original';record={'id':'A001','candidate':self.candidate,'geometry':self.g,
            'geometry_state':'accept','style_state':'accept','styles':style(self.g),'history':[], 'state':'completed'}
        save(original/'scene.json',{'source_sha256':sha(self.source),'records':{'A001':record},'objects':[record],'state':'finished'})
        save(original/'summary.json',{'state':'completed'})
        save(original/'objects/A001/record.json',record)
        save(original/'detection/candidates.json',{'objects':[self.candidate],'errors':[],'uncertain':[],'history':[]})
        digest=sha(original/'scene.json');seen=[]
        def recover(source,folder,rt,opt,slots,seed):
            self.assertEqual(seed['geometry'],self.g);seen.append(seed)
            seed.update(state='completed',style_state='accept',styles=style(self.g));return seed
        counts={'connectors':3,'freeforms':0,'heads':2,'networks':1}
        with patch('shape_integration.arrows.controller.discover',side_effect=AssertionError('Must reuse detection')),patch('shape_integration.arrows.controller.fit_object',side_effect=AssertionError('Must reuse geometry')),patch('shape_integration.arrows.controller.fit_style',side_effect=recover),patch('shape_integration.arrows.export.final',return_value=counts),ThreadPoolExecutor(max_workers=2) as pool:
            result=controller.run_case(self.source,self.root/'replayed',SimpleNamespace(config={'model':'test'}),self.opt,pool,threading.BoundedSemaphore(1),frozen=original)
        self.assertEqual(len(seen),1);self.assertEqual(result['objects'],1);self.assertEqual(result['geometry_source'],str(original))
        self.assertEqual(sha(original/'scene.json'),digest)
        self.assertEqual(read(self.root/'replayed/objects/A001/source_geometry.json'),self.g)

    def test_missing_route_context_expands_to_full_image_once(self):
        frames=[]
        def ask(rt,folder,p,images,validator,opt):
            if folder.name=='observe':
                f=read(folder.parent/'input/frame.json');frames.append(f)
                if len(frames)==1:return validator({'status':'needs_context','reason':'shaft continues outside previous crop near x=9999'})
                if len(frames)==2:
                    self.assertEqual(f['box'],[0,0,240,180]);self.assertIn('full figure',p['feedback'])
                else:
                    self.assertNotEqual(f['box'],[0,0,240,180])
                    self.assertEqual(p['previous_geometry']['points'],geometry.mapped(self.g,f)['points'])
                self.assertNotIn('9999',str(p))
                return validator(geometry.mapped(self.g,f))
            if folder.name=='critic':return validator({'verdict':'accept'} if 'current_style' in p else geometry_verdict())
            if folder.name=='shaft':return validator({'styles':[{'id':x['id'],'stroke':'#000000','width_px':3} for x in self.g['paths']]})
            if folder.name=='heads':return validator({'heads':[{'id':x['id'],'start':None,'end':{'type':'triangle','width':'med','length':'med'} if x['end_head'] else None} for x in self.g['paths']]})
            raise AssertionError(folder)
        candidate={**self.candidate,'bbox_2d':[50,90,60,105]}
        with patch('shape_integration.arrows.model.ask',side_effect=ask),patch('shape_integration.arrows.export.render_object',side_effect=self.fake_render):
            result=controller.fit_object(self.source,candidate,self.root/'expanded',None,self.opt,threading.BoundedSemaphore(1))
        self.assertEqual(len(frames),3);self.assertNotEqual(frames[0]['box'],frames[1]['box'])
        self.assertEqual(result['state'],'completed')
        self.assertIn('9999',result['history'][0]['reason'])
        for key in self.g['points']:np.testing.assert_allclose(result['geometry']['points'][key],self.g['points'][key])

    def test_replay_can_retry_missing_geometry_without_touching_source(self):
        original=self.root/'original';record={'id':'A001','candidate':self.candidate,'state':'unresolved','history':[{'stage':'context','reason':'clipped'}]}
        save(original/'scene.json',{'source_sha256':sha(self.source),'records':{'A001':record},'objects':[],'state':'finished'})
        save(original/'summary.json',{'state':'completed_with_unresolved'})
        save(original/'objects/A001/record.json',record)
        save(original/'detection/candidates.json',{'objects':[self.candidate],'errors':[],'uncertain':[],'history':[]})
        digest=sha(original/'scene.json');seen=[]
        def recover(source,candidate,folder,rt,opt,slots,*,initial_full_context=False):
            self.assertTrue(initial_full_context);seen.append(candidate)
            return {**record,'geometry':self.g,'styles':style(self.g),'geometry_state':'accept','style_state':'accept','state':'completed'}
        with patch('shape_integration.arrows.controller.discover',side_effect=AssertionError('No new detection')),patch('shape_integration.arrows.controller.fit_style',side_effect=AssertionError('Missing geometry requires fitting')),patch('shape_integration.arrows.controller.fit_object',side_effect=recover),patch('shape_integration.arrows.export.final',return_value={}),ThreadPoolExecutor(max_workers=2) as pool:
            result=controller.run_case(self.source,self.root/'recovered',SimpleNamespace(config={'model':'test'}),self.opt,pool,threading.BoundedSemaphore(1),frozen=original,retry_missing_geometry=True)
        self.assertEqual(seen,[self.candidate]);self.assertEqual(result['objects'],1)
        self.assertEqual(sha(original/'scene.json'),digest)
        self.assertEqual(read(self.root/'recovered/objects/A001/source_record.json')['state'],'unresolved')

    def test_failed_local_attempt_does_not_relabel_valid_full_geometry(self):
        attempts=[]
        def ask(rt,folder,p,images,validator,opt):
            if folder.name=='observe':
                attempts.append(folder)
                if len(attempts)==1:return validator(geometry.mapped(self.g,read(folder.parent/'input/frame.json')))
                return validator({'status':'needs_context','reason':'The route still leaves this crop'})
            raise RequestTimeout('style unavailable')
        with patch('shape_integration.arrows.model.ask',side_effect=ask):
            result=controller.fit_object(self.source,self.candidate,self.root/'partial',None,self.opt,threading.BoundedSemaphore(1),initial_full_context=True)
        self.assertEqual(result['geometry_attempt'],1)
        self.assertEqual(result['geometry_revision'],0)
        self.assertEqual(result['geometry_source_view'],'full_context')
        self.assertEqual(result['state'],'completed_with_unresolved')
        self.assertEqual(result['geometry'],read(self.root/'partial/geometry/revision_00/geometry_source.json'))
        self.assertFalse((self.root/'partial/geometry/revision_01/geometry_source.json').exists())


if __name__=='__main__':unittest.main()
