"""Behavioral tests of style recovery; no external model calls."""
import copy
import math
import sys
import tempfile
import threading
import unittest
import fitz
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import GeometryCollection,box

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import sha,digest,read,save
from shape_integration.styling import options
from shape_integration.styling import geometry,occlusion,gradient,rules,contracts,model,views,grouping
from shape_integration.styling.export import export,geometry_signature,render_pdf_crop,pattern
from shape_integration.styling.controller import run_case
from shape_integration.fitting.paths import box_commands
from shape_integration.transport import RequestTimeout


def rectangle(oid,ident,cx,cy,w,h,angle=0):
    p={'key':['test/revision_00',oid,ident],'object_id':oid,'id':ident,'kind':'rectangle','status':'ok',
       'commands':box_commands([cx,cy,w,h,angle]),'native_parameters_original':
       {'center':[cx,cy],'width':w,'height':h,'angle_radians':angle,'radius':0}}
    return geometry.part_key(p),p


def scene_of(entries,size=(240,180),source_hash='test'):
    table=dict(entries);objs={}
    for key,p in table.items():
        o=objs.setdefault(p['object_id'],{'id':p['object_id'],'state':'unresolved','selected_revision':'revision_00','part_keys':[]})
        o['part_keys'].append(key)
    return {'schema_version':'shape_integration.scene.1','source_size':list(size),'source_sha256':source_hash,
            'objects':objs,'parts':table,'points':{},'paths':{}}


class FakeRuntime:
    def __init__(self,responder):self.responder=responder;self.events=[];self.config={'model':'fake','request_timeout_seconds':1200}
    def for_stage(self,**kwargs):return self
    def call(self,root,tag,system,prompt,images,validator,budget):
        self.events.append((tag,copy.deepcopy(prompt)))
        raw=self.responder(tag,prompt)
        if isinstance(raw,Exception):raise raw
        return validator(raw)


class StyleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.opt=options({})
        self.opt.update(crop_min_long_edge=160,crop_max_long_edge=320,live_render_interval_seconds=0)
        self.k,self.p=rectangle('A','face',100,85,140,100)
        self.image=Image.new('RGB',(240,180),'white');ImageDraw.Draw(self.image).rectangle([30,35,170,135],fill='#CC5533')
        self.source=self.folder/'source.png';self.image.save(self.source)
        self.scene=scene_of([(self.k,self.p)],source_hash=sha(self.source))

    def fake_export(self,scene,styles,relations,folder,source,opt,slots):
        folder.mkdir(parents=True,exist_ok=True);self.image.save(folder/'native.png')
        with fitz.open() as doc:
            page=doc.new_page(width=self.image.width,height=self.image.height)
            page.draw_rect(fitz.Rect(30,35,170,135),color=(0,0,0))
            doc.save(folder/'native.pdf')
        for name in ('styled.pptx','styled.svg'):(folder/name).write_text(digest(styles))
        for name in ('styled.png','comparison.png'):self.image.save(folder/name)
        save(folder/'export.json',{'native_rendered':True,'files':{n:sha(folder/n) for n in
             ('styled.pptx','styled.svg','styled.png','native.pdf','native.png','comparison.png')}})

    def assert_style_context_matches_snapshot(self,folder):
        jobs=[p for p in folder.rglob('job.json') if p.parent.name in ('critic_000','refine_000')]
        self.assertTrue(jobs)
        for path in jobs:
            job=read(path);snapshot=folder/job['render_snapshot'];saved=read(snapshot/'styles.json')
            frame=read(path.parent/'input/frame.json');scale=frame['source_to_crop_scale']
            field='current_style' if path.parent.name=='critic_000' else 'previous_style'
            self.assertEqual(job['render_sha256'],sha(snapshot/'native.pdf'))
            for target in job['prompt']['targets']:
                for key in job['output_mapping'][target['id']]:
                    with self.subTest(job=str(path),key=key):
                        expected=copy.deepcopy(saved[key]);expected.pop('issue',None)
                        dash=pattern(expected)
                        if dash:expected['dash_pattern_px']=list(dash)
                        expected['width_px']*=scale
                        if 'dash_pattern_px' in expected:
                            expected['dash_pattern_px']=[v*scale for v in expected['dash_pattern_px']]
                        self.assertEqual(target[field],expected)

    def test_curve_bounds_use_extrema_not_control_points(self):
        p={'commands':[['M',0,0],['C',0,100,100,100,100,0]]}
        np.testing.assert_allclose(geometry.bounds(p),[0,0,100,75])

    def test_containment_uses_geometry_not_only_bbox(self):
        k,p=rectangle('B','other',150,120,10,10)
        tri=copy.deepcopy(self.p);tri['commands']=[['M',0,0],['L',200,0],['L',0,150],['Z']]
        _,rel=occlusion.discover({self.k:tri,k:p},self.opt)
        self.assertEqual(rel['edges'],[])

    def test_shared_edge_is_not_partial_overlap(self):
        a=rectangle('a','f',30,30,40,40);b=rectangle('b','f',70,30,40,40)
        _,rel=occlusion.discover(dict([a,b]),self.opt)
        self.assertFalse(rel['pending']);self.assertFalse(rel['edges'])

    def test_relation_crops_separate_disconnected_components(self):
        pairs=[{'a':'a','b':'b'},{'a':'x','b':'y'},{'a':'b','b':'c'}]
        batches=occlusion.relation_batches(pairs,12)
        self.assertEqual(batches,[[pairs[0],pairs[2]],[pairs[1]]])
        self.assertEqual(len(occlusion.relation_batches(pairs,1)),3)

    def test_inner_object_above_outer_and_hollow_mask(self):
        k,p=rectangle('B','inner',100,85,30,30);s=scene_of([(k,p),(self.k,self.p)])
        table=geometry.parts(s);geoms,rel=occlusion.discover(table,self.opt);rel=occlusion.order(s,table,rel)
        self.assertEqual(rel['flat_order'],[self.k,k])
        mask=occlusion.paint(geoms[k],{'fill':None,'stroke':'#000000','width_px':2})
        self.assertFalse(mask.contains(box(97,82,103,88)))
        self.assertLess(mask.area,geoms[k].area)

    def test_cross_group_cycle_is_exposed(self):
        a=rectangle('a','one',30,30,40,40);b=rectangle('b','one',50,50,40,40)
        a2=rectangle('a','two',90,90,40,40);s=scene_of([a,b,a2])
        geoms,rel=occlusion.discover(geometry.parts(s),self.opt)
        rel['edges']=[{'back':'a','front':'b'},{'back':'b','front':'a'}]
        out=occlusion.order(s,geometry.parts(s),rel)
        self.assertTrue(out['order_conflicts']);self.assertTrue(out['unapplied_edges'])
        self.assertEqual(out['effective_edges'],[])
        self.assertEqual(out['part_orders']['a'],[a[0],a2[0]])
        for k in geoms:self.assertTrue(occlusion.exclusions(k,geoms,out,{}).is_empty)

    def test_internal_overlap_and_containment_never_become_relations_or_masks(self):
        entries=[rectangle('stack','back',80,70,70,60),rectangle('stack','front',65,85,70,60),
                 rectangle('stack','detail',65,85,12,12)]
        scene=scene_of(entries);table=geometry.parts(scene)
        geoms,rel=occlusion.discover(table,self.opt);rel=occlusion.order(scene,table,rel)
        self.assertEqual(rel['edges'],[]);self.assertEqual(rel['pending'],[])
        self.assertEqual(rel['part_orders']['stack'],[k for k,_ in entries])
        for k in geoms:self.assertTrue(occlusion.exclusions(k,geoms,rel,{}).is_empty)
        actual=rules.ranked_source_colors(self.image,table,lambda k:occlusion.exclusions(k,geoms,rel,{}))
        expected=rules.ranked_source_colors(self.image,table,lambda k:GeometryCollection())
        self.assertEqual(actual,expected)

    def test_compound_candidates_are_unique_whole_component_pairs(self):
        entries=[rectangle('A','left',60,80,60,60),rectangle('A','right',140,80,60,60),
                 rectangle('B','left',90,80,100,40),rectangle('B','right',170,80,50,50)]
        table=dict(entries);_,rel=occlusion.discover(table,self.opt)
        self.assertEqual(rel['pending'],[{'a':'A','b':'B','kind':'partial_overlap'}])
        self.assertEqual(rel['edges'],[])
        self.assertEqual(rel['component_parts']['A'],[k for k,p in entries if p['object_id']=='A'])

    def test_whole_component_containment_and_union_holes(self):
        # A bbox encloses B, but the disconnected component leaves B in a gap.
        entries=[rectangle('A','left',40,80,20,100),rectangle('A','right',160,80,20,100),
                 rectangle('B','gap',100,80,20,20)]
        _,rel=occlusion.discover(dict(entries),self.opt)
        self.assertEqual(rel['edges'],[]);self.assertEqual(rel['pending'],[])
        # Containment must cover every part of the contained component.
        inner=[rectangle('B','one',70,80,20,20),rectangle('B','two',130,80,20,20)]
        _,rel=occlusion.discover(dict([(self.k,self.p),*inner]),self.opt)
        self.assertEqual(rel['edges'],[{'back':'A','front':'B','origin':'contains'}])

    def test_unknown_order_preserves_pixels_but_confirmed_foreground_masks_all_its_parts(self):
        entries=[rectangle('A','face',80,80,90,80),rectangle('B','one',110,70,70,50),
                 rectangle('B','two',110,110,70,30)]
        scene=scene_of(entries);table=geometry.parts(scene);geoms,rel=occlusion.discover(table,self.opt)
        self.assertEqual(len(rel['pending']),1)
        rel['pending'][0]['reason']='timeout: output discarded'
        unknown=occlusion.order(scene,table,rel)
        for key in geoms:self.assertTrue(occlusion.exclusions(key,geoms,unknown,{}).is_empty)
        rel['pending']=[];rel['edges']=[{'back':'A','front':'B','origin':'model'}]
        known=occlusion.order(scene,table,rel)
        styles={k:{'fill':'#0000FF','stroke':None} for k,p in entries if p['object_id']=='B'}
        mask=occlusion.exclusions(entries[0][0],geoms,known,styles)
        self.assertTrue(mask.contains(box(100,65,105,75)))
        self.assertTrue(mask.contains(box(100,105,105,115)))
        self.assertFalse(mask.contains(box(40,65,45,75)))
        for k in list(table)[1:]:self.assertTrue(occlusion.exclusions(k,geoms,known,styles).is_empty)

    def test_component_overlay_includes_nonoverlapping_members_with_one_label(self):
        entries=[rectangle('A','overlap',60,65,50,50),rectangle('A','far',205,140,20,20),
                 rectangle('B','overlap',80,65,50,50)]
        table=dict(entries);_,rel=occlusion.discover(table,self.opt)
        images,_,frame=views.prepare(self.source,table,{'C0':'A','C1':'B'},self.folder/'components',self.opt,
                                    component_members=rel['component_parts'])
        self.assertEqual(frame['target_scope'],'components')
        self.assertEqual(set(frame['labels']),{'C0','C1'})
        self.assertEqual(frame['component_members']['C0'],[entries[0][0],entries[1][0]])
        self.assertGreater(frame['crop_original_xyxy'][2],215)
        self.assertGreater(frame['crop_original_xyxy'][3],150)
        self.assertIn('whole-component',images[-1][0])

    def test_controller_requests_component_order_and_keeps_compound_style_whole(self):
        entries=[rectangle('A','overlap',60,65,50,50),rectangle('A','far',205,140,20,20),
                 rectangle('B','overlap',80,65,50,50)]
        scene=scene_of(entries,source_hash=sha(self.source));before=digest(scene)
        def respond(tag,p):
            if tag.startswith('relations'):
                self.assertEqual(p['targets'],[{'id':'C0'},{'id':'C1'}])
                self.assertEqual(p['pairs'],[{'id':'R0','parts':['C0','C1']}])
                return {'relations':[{'id':'R0','front':'C1','evidence':'B boundary covers A boundary'}]}
            if tag.startswith('critic'):
                return {'checks':[{'id':t['id'],'verdict':'accept'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':None} for t in p['targets']]}
        rt=FakeRuntime(respond);folder=self.folder/'component_controller'
        with patch('shape_integration.styling.controller.export',side_effect=self.fake_export), ThreadPoolExecutor(max_workers=4) as pool:
            run_case(scene,self.source,folder,rt,{**self.opt,'component_grouping_enabled':False},pool,threading.BoundedSemaphore(1))
        self.assertEqual(sum(tag.startswith('relations') for tag,_ in rt.events),1)
        rel=read(folder/'occlusion.json')
        self.assertEqual(rel['edges'],[{'back':'A','front':'B','origin':'model','evidence':'B boundary covers A boundary'}])
        self.assertEqual(rel['part_orders']['A'],[entries[0][0],entries[1][0]])
        select=read(folder/'objects/A/revision_00/select_000/job.json')
        self.assertEqual(len(select['prompt']['targets']),2)
        self.assertEqual(digest(scene),before)

    def test_style_rows_salvage_and_exact_errors(self):
        p0={'id':'P0','fill':'#CC5533','stroke':None,'width_px':None,'dash':'solid','irrelevant':'extra'}
        p1={'id':'P1','fill':'multi_color','stroke':None}
        res=contracts.rows({'styles':[p0,p1]},['P0','P1'],'refine',{'P0':self.p,'P1':self.p})
        self.assertEqual(len(res['styles']),1);self.assertEqual(res['styles'][0]['width_px'],0)
        self.assertIn('P1',res['_errors'])
        self.assertIn({'id':'P0','ignored_fields':['irrelevant']},res['_discarded'])
        self.assertEqual(res,contracts.rows(res,['P0','P1'],'refine',{'P0':self.p,'P1':self.p}))
        duplicate=contracts.rows({'styles':[p0,p0]},['P0'],'refine',{'P0':self.p})
        self.assertEqual(duplicate['_errors']['P0'],'Duplicate ID')

    def test_gradient_contract_and_no_implicit_open_fill(self):
        f={'type':'linear_gradient','start_color':'#000000','end_color':'#FFFFFF','angle_deg':450}
        self.assertEqual(contracts.fill(f)['angle_deg'],90)
        with self.assertRaises(ValueError):contracts.fill({**f,'angle_deg':float('nan')})
        p=copy.deepcopy(self.p);p['commands']=p['commands'][:-1]
        with self.assertRaises(ValueError):contracts.style({'fill':f,'stroke':None},p)

    def test_gradient_estimator_and_solid_negative_control(self):
        yy,xx=np.mgrid[42:128,37:163];xy=np.c_[xx.ravel(),yy.ravel()]
        t=geometry.gradient_position(self.p,32,xy)
        values=(1-t[:,None])*np.array([40,80,100])+t[:,None]*np.array([170,200,230])
        f,a=gradient.estimate(self.p,xy,values,self.opt)
        self.assertIsNotNone(f);pred=geometry.gradient_position(self.p,f['angle_deg'],xy)
        colors=np.array([gradient.rgb(f['start_color']),gradient.rgb(f['end_color'])])
        np.testing.assert_allclose((1-pred[:,None])*colors[0]+pred[:,None]*colors[1],values,atol=1.5)
        f,_=gradient.estimate(self.p,xy,np.full_like(values,100),self.opt);self.assertIsNone(f)

    def test_masked_foreground_does_not_contaminate_fill(self):
        image=self.image.copy();ImageDraw.Draw(image).rectangle([40,45,145,125],fill='#0000FF')
        result=rules.estimate(image,self.p,box(39,44,146,126),self.opt)
        self.assertEqual(result['style']['fill'],'#CC5533')

    def test_crop_follows_geometry_and_half_pixel_transform(self):
        a,_,frame=views.prepare(self.source,{self.k:self.p},{'P0':self.k},self.folder/'view',self.opt)
        self.assertLess(frame['crop_original_xyxy'][0],30)
        self.assertGreater(frame['crop_original_xyxy'][2],170)
        self.assertEqual(Image.open(a[0][1]).size,Image.open(a[1][1]).size)
        before=geometry.gradient_line(self.p,32)
        views.prepare(self.source,{self.k:self.p},{'P0':self.k},self.folder/'view2',{**self.opt,'crop_padding':.3})
        for p,q in zip(before,geometry.gradient_line(self.p,32)):np.testing.assert_array_equal(p,q)

    def test_minimum_stroke_config_rejects_invalid_lengths(self):
        self.assertEqual(options({})['minimum_stroke_width_px'],.8)
        self.assertEqual(options({'styling':{'minimum_stroke_width_px':0}})['minimum_stroke_width_px'],0)
        for invalid in (True,-1,float('inf'),float('nan'),'0.8',None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError,'minimum_stroke_width_px'):
                options({'styling':{'minimum_stroke_width_px':invalid}})

    def test_stroke_reference_preserves_crop_pixels_and_render_alignment(self):
        native=self.folder/'native';self.fake_export(None,None,None,native,None,None,None)
        args=(self.source,{self.k:self.p},{'P0':self.k})
        plain,_,before=views.prepare(*args,self.folder/'plain',self.opt,native_pdf=native/'native.pdf')
        images,units,frame=views.prepare(*args,self.folder/'reference',self.opt,native_pdf=native/'native.pdf',
                                       with_stroke_reference=True)
        self.assertEqual(len(images),3)
        self.assertEqual(frame['crop_display_size'],before['display_size'])
        self.assertEqual(frame['source_to_crop_scale'],before['source_to_crop_scale'])
        self.assertAlmostEqual(units['minimum_stroke_width_px'],.8*frame['source_to_crop_scale'])
        self.assertEqual(units['main_crop_size_px'],before['display_size'])
        footers=[]
        for (_,baseline),(_,path) in zip(plain,images):
            with Image.open(baseline) as old, Image.open(path) as new:
                self.assertEqual(list(new.size),frame['display_size'])
                np.testing.assert_array_equal(np.asarray(new)[:old.height],np.asarray(old))
                footers.append(np.asarray(new)[old.height:])
        for footer in footers[1:]:np.testing.assert_array_equal(footer,footers[0])
        disabled,disabled_units,disabled_frame=views.prepare(*args,self.folder/'disabled',
                     {**self.opt,'minimum_stroke_width_px':0},native_pdf=native/'native.pdf',with_stroke_reference=True)
        self.assertIsNone(disabled_frame['stroke_reference'])
        self.assertNotIn('minimum_stroke_width_px',disabled_units)
        for (_,a),(_,b) in zip(plain,disabled):self.assertEqual(sha(a),sha(b))

    def test_reference_line_fractional_width_tracks_crop_scale(self):
        for scale in (.5,1,14.953488):
            path=self.folder/f'reference_{scale}.png';Image.new('RGB',(300,100),'white').save(path)
            audit=views.stroke_reference([('clean',path)],(300,100),.8,scale)
            with Image.open(path) as image:
                # Below the label: integrated pixel coverage, including fractional edge pixels.
                profile=1-np.asarray(image,dtype=float)[128:,150,0]/255
            self.assertAlmostEqual(float(profile.sum()),.8*scale,delta=2/255)
            self.assertAlmostEqual(audit['minimum_crop_px'],.8*scale)

    def test_stroke_floor_covers_rules_selection_refine_and_previous_render(self):
        entries=[rectangle('A',str(i),45+i*75,85,25,55) for i in range(3)]
        scene=scene_of(entries,source_hash=sha(self.source))
        def estimate(image,part,excluded,opt):
            style={'fill':None,'stroke':None if part['id']=='1' else '#CC5533',
                   'width_px':0 if part['id']=='1' else .2 if part['id']=='0' else 2,'dash':'solid'}
            return {'style':style,'candidates':{'fills':[None],'strokes':[None,'#CC5533']},
                    'needs_model':True,'reasons':[]}
        revised=[]
        def respond(tag,p):
            minimum=p['units']['minimum_stroke_width_px']
            if tag.startswith('critic'):
                return {'checks':[{'id':t['id'],'verdict':'accept' if revised else 'revise',
                                  'source_evidence':'edge visible','rendered_difference':'edge color differs',
                                  'suggestion':'match source edge color'} for t in p['targets']]}
            if tag.startswith('refine'):
                for t in p['targets']:
                    expected=0 if t['id']=='P1' else minimum if t['id']=='P0' else minimum*3
                    self.assertAlmostEqual(t['previous_style']['width_px'],expected)
                revised.append(True)
            return {'styles':[{'id':t['id'],'fill':None,'stroke':None if t['id']=='P1' else '#CC5533',
                              'width_px':0 if t['id']=='P1' else minimum*(.1 if revised else .25) if t['id']=='P0' else minimum*3,
                              'dash':'solid'} for t in p['targets']]}
        for rules_only in (True,False):
            folder=self.folder/('floor_rules' if rules_only else 'floor_model')
            with patch('shape_integration.styling.rules.estimate',side_effect=estimate), \
                 patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
                 patch('shape_integration.styling.calibration.calibrate',side_effect=AssertionError('disabled')), \
                 ThreadPoolExecutor(max_workers=4) as pool:
                run_case(scene,self.source,folder,FakeRuntime(respond),self.opt,pool,
                         threading.BoundedSemaphore(1),rules_only=rules_only)
            for revision in (['revision_00'] if rules_only else ['revision_00','revision_01']):
                styles=read(folder/'objects/A'/revision/'styles.json')
                self.assertEqual(styles[entries[0][0]]['width_px'],.8)
                self.assertIsNone(styles[entries[1][0]]['stroke'])
                self.assertEqual(styles[entries[1][0]]['width_px'],0)
                self.assertAlmostEqual(styles[entries[2][0]]['width_px'],2 if rules_only else 2.4)
                floor=read(folder/('rules_baseline' if rules_only else 'objects/A/'+revision)/'stroke_width_floor.json')
                self.assertEqual(set(floor['applied']),{entries[0][0]})
            if not rules_only:
                for revision,mode,original in [('revision_00','select',.2),('revision_01','refine',.08)]:
                    valid=read(folder/'objects/A'/revision/(mode+'_000')/'calls/validation.json')
                    self.assertAlmostEqual(valid['valid']['P0']['width_px'],original)

    def test_concentric_part_labels_remain_distinguishable(self):
        entries=[rectangle('nested',str(i),100,85,80-i*4,60-i*3) for i in range(8)]
        mapping={f'P{i}':key for i,(key,_) in enumerate(entries)}
        _,_,frame=views.prepare(self.source,dict(entries),mapping,self.folder/'labels',self.opt)
        rects=[v['label_box_display'] for v in frame['labels'].values()]
        self.assertEqual(len(rects),8)
        for i,a in enumerate(rects):
            for b in rects[i+1:]:self.assertEqual(box(*a).intersection(box(*b)).area,0)

    def test_retry_only_failed_rows_and_timeout_has_no_retry(self):
        n=[0]
        def respond(tag,p):
            n[0]+=1
            return {'styles':[{'id':t['id'],'fill':'bad' if n[0]==1 and t['id']=='P1' else '#CC5533','stroke':None} for t in p['targets']]}
        baseline={'fill':None,'stroke':'#000000','width_px':2.5,'dash':'solid'}
        runtime=FakeRuntime(respond);p=model.prompt('refine',[{'id':i,'previous_style':copy.deepcopy(baseline)} for i in ('P0','P1')],{})
        res=model.ask(runtime,self.folder/'call','refine',p,[],['P0','P1'],{'P0':self.p,'P1':self.p},self.opt,source_to_crop_scale=1)
        self.assertEqual(set(res['valid']),{'P0','P1'})
        self.assertEqual([t['id'] for t in runtime.events[1][1]['targets']],['P0','P1'])
        self.assertEqual(runtime.events[1][1]['format_correction']['repair_ids'],['P1'])
        self.assertTrue(all(t['previous_style']==baseline for t in runtime.events[1][1]['targets']))
        self.assertEqual(runtime.events[1][1]['output']['styles'][0]['id'],'P1')
        rt=FakeRuntime(lambda t,p:RequestTimeout('test deadline'))
        res=model.ask(rt,self.folder/'timeout','refine',p,[],['P0','P1'],{'P0':self.p,'P1':self.p},self.opt,source_to_crop_scale=1)
        self.assertEqual(len(rt.events),1);self.assertFalse(res['valid'])

    def test_native_gradient_export_and_geometry_invariance(self):
        entries=[rectangle('R'+str(i),'face',170+360*(i%2),130+240*(i//2),180,90,
                           math.radians(27) if i==3 else 0) for i in range(4)]
        source=self.folder/'gradient_source.png';Image.new('RGB',(720,480),'white').save(source)
        scene=scene_of(entries,size=(720,480),source_hash=sha(source));table=geometry.parts(scene)
        _,rel=occlusion.discover(table,self.opt);rel=occlusion.order(scene,table,rel)
        styles={key:{'fill':{'type':'linear_gradient','start_color':'#204060','end_color':'#CCDDEE','angle_deg':angle},
                     'stroke':None,'width_px':0,'dash':'solid'} for (key,_),angle in zip(entries,[0,90,45,51])}
        out=self.folder/'native';result=export(scene,styles,rel,out,source,self.opt,threading.BoundedSemaphore(1))
        self.assertTrue(result['geometry_unchanged'])
        a=np.asarray(Image.open(out/'styled.png').convert('RGB'),float)
        b=np.asarray(Image.open(out/'native.png').convert('RGB'),float)
        for key,part in entries:
            geom,_=geometry.footprint(part);mask=geometry.raster(geom.buffer(-5),[0,0,720,480])
            with self.subTest(angle=styles[key]['fill']['angle_deg']):
                self.assertLess(float(np.mean(abs(a[mask]-b[mask]))),8.)

    def test_controller_one_refine_and_preserves_unresolved_geometry(self):
        counts={'critic':0,'refine':0}
        def respond(tag,p):
            mode=tag.split('_')[0]
            if mode=='critic':
                counts['critic']+=1
                return {'checks':[{'id':t['id'],'verdict':'revise' if counts['critic']==1 else 'accept',
                                  'source_evidence':'source has orange fill','rendered_difference':'rendered fill wrong',
                                  'suggestion':'use the visible orange fill'} for t in p['targets']]}
            if mode=='refine':counts['refine']+=1
            return {'styles':[{'id':t['id'],'fill':'#CC5533' if mode=='refine' else None,'stroke':None} for t in p['targets']]}
        rt=FakeRuntime(respond);before=digest(self.scene)
        with ThreadPoolExecutor(max_workers=4) as pool:
            summary=run_case(self.scene,self.source,self.folder/'case',rt,self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(counts,{'critic':2,'refine':1});self.assertEqual(summary['critic_accepted'],1)
        self.assertEqual(before,digest(self.scene))
        data=read(self.folder/'case/styles.json');self.assertEqual(data['parts'][self.k]['geometry_state'],'unresolved')
        pdfs=[]
        for revision,mode in [('revision_00','critic'),('revision_01','refine'),('revision_01','critic')]:
            job=self.folder/'case/objects/A'/revision/(mode+'_000')
            frame=read(job/'input/frame.json')
            pdf=self.folder/'case'/read(job/'job.json')['render_snapshot']/'native.pdf';pdfs.append(pdf)
            self.assertEqual(frame['rendered_source']['path'],str(pdf.resolve()))
            self.assertEqual(frame['rendered_source']['method'],'direct_pdf_crop.1')
            self.assertEqual(frame['rendered_source']['sha256'],sha(pdf))
            self.assertEqual(read(job/'job.json')['render_sha256'],sha(pdf))
            with Image.open(job/'input/rendered.png') as im:self.assertEqual(list(im.size),frame['display_size'])
        self.assertEqual(pdfs[0],pdfs[1]);self.assertNotEqual(pdfs[1],pdfs[2])
        self.assertEqual(sha(self.folder/'case/native.pdf'),sha(pdfs[2]))
        self.assertEqual(read(self.folder/'case/export.json')['files']['native.pdf'],sha(self.folder/'case/native.pdf'))
        self.assert_style_context_matches_snapshot(self.folder/'case')

    def test_off_image_target_does_not_abort_other_styles(self):
        outside=rectangle('outside','face',450,450,60,40)
        scene=scene_of([(self.k,self.p),outside],source_hash=sha(self.source))
        def respond(tag,p):
            if tag.startswith('critic'):
                return {'checks':[{'id':t['id'],'verdict':'accept'} for t in p['targets']]}
            return {'styles':[{'id':t['id'],'fill':'#CC5533','stroke':None} for t in p['targets']]}
        with ThreadPoolExecutor(max_workers=4) as pool:
            summary=run_case(scene,self.source,self.folder/'partial',FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        data=read(self.folder/'partial/styles.json')
        self.assertEqual(summary['parts'],2)
        self.assertEqual(data['parts'][self.k]['state'],'critic_accept')
        self.assertEqual(data['parts'][outside[0]]['state'],'unresolved')
        self.assertIn('input_error',data['parts'][outside[0]]['critic']['error'])
        self.assertTrue((self.folder/'partial/styled.pptx').exists())

    def test_optional_stage_failure_preserves_geometry_status(self):
        from shape_integration.pipeline import append_styling
        original={'state':'completed','objects':1}
        with patch('shape_integration.styling.controller.run_case',side_effect=RuntimeError('render unavailable')):
            result=append_styling(self.folder,self.folder/'pipeline',{'original':'source.png'},self.scene,
                                  original,None,{},None,threading.BoundedSemaphore(1))
        self.assertEqual(result['state'],'completed')
        self.assertEqual(result['styling']['state'],'failed')
        self.assertEqual(original,{'state':'completed','objects':1})

    def test_compound_over_sixteen_parts_stays_whole_even_for_one_part_refine(self):
        entries=[rectangle('A',str(i),20+(i%6)*40,30+(i//6)*45,20,20) for i in range(18)]
        scene=scene_of(entries,source_hash=sha(self.source));refined=[]
        def respond(tag,p):
            mode=tag.split('_')[0]
            example=next(iter(p['output'].values()))[0]['id']
            self.assertEqual(example,p['targets'][0]['id'])
            if mode=='critic':
                return {'checks':[{'id':t['id'],'verdict':'revise' if t['id']=='P17' and not refined else 'accept',
                                  'source_evidence':'P17 is orange','rendered_difference':'P17 is unfilled',
                                  'suggestion':'Give P17 the source orange fill'} for t in p['targets']]}
            if mode=='refine':refined.extend(t['id'] for t in p['targets'])
            return {'styles':[{'id':t['id'],'fill':'#CC5533' if mode=='refine' else None,'stroke':None} for t in p['targets']]}
        rt=FakeRuntime(respond)
        with patch('shape_integration.styling.controller.export',side_effect=self.fake_export), ThreadPoolExecutor(max_workers=4) as pool:
            run_case(scene,self.source,self.folder/'aliases',rt,{**self.opt,'review_batch_parts':2,'selection_batch_parts':2},
                     pool,threading.BoundedSemaphore(1))
        self.assertEqual(refined,[f'P{i}' for i in range(18)])
        self.assertEqual([tag for tag,_ in rt.events],['select_00','critic_00','refine_00','critic_00'])
        self.assertTrue(all(len(p['targets'])==18 for _,p in rt.events))
        final=read(self.folder/'aliases/objects/A/revision_01/refine_000/job.json')
        self.assertEqual(final['mapping'],{f'P{i}':v[0] for i,v in enumerate(entries)})
        self.assertEqual(len(final['prompt']['targets']),18)
        self.assertTrue(all(t['previous_style']=={'fill':None,'stroke':None,'width_px':0,'dash':'solid'}
                            for t in final['prompt']['targets']))
        self.assert_style_context_matches_snapshot(self.folder/'aliases')

    def test_width_and_colors_are_free_while_style_contracts_remain_validated(self):
        candidates={'fills':[None],'strokes':['#000000']}
        row={'fill':None,'stroke':'#000000','width_px':7.73,'dash':'dash','dash_pattern_px':[18.3,6.1]}
        result=contracts.style(row,self.p,candidates)
        self.assertEqual(result['width_px'],7.73)
        self.assertEqual(contracts.style({**row,'fill':'#112233','stroke':'#AABBCC'},self.p,candidates)['stroke'],'#AABBCC')
        for invalid in ({**row,'stroke':'blue'},{**row,'fill':'#1234'},
                        {**row,'width_px':-1},{**row,'dash_pattern_px':[3,0]}):
            with self.assertRaises(ValueError):contracts.style(invalid,self.p,candidates)
        open_part={**self.p,'commands':[['M',0,0],['L',100,50]]}
        with self.assertRaisesRegex(ValueError,'open path'):
            contracts.style({**row,'fill':'#112233'},open_part,candidates)
        for scale in (.5,6.89):
            with self.subTest(scale=scale):
                normalized=contracts.to_source_pixels(result,scale)
                self.assertAlmostEqual(normalized['width_px'],7.73/scale)
                np.testing.assert_allclose(normalized['dash_pattern_px'],[18.3/scale,6.1/scale])
                self.assertEqual(result['width_px'],7.73)

    def test_select_accepts_unlisted_colors_and_gradient_without_format_retry(self):
        gradient_fill={'type':'linear_gradient','start_color':'#123456','end_color':'#ABCDEF','angle_deg':45}
        response={'styles':[{'id':'P0','fill':'#112233','stroke':'#AABBCC','width_px':4},
                            {'id':'P1','fill':gradient_fill,'stroke':None}]}
        rt=FakeRuntime(lambda tag,p:response)
        p=model.prompt('select',[{'id':'P0'},{'id':'P1'}],{},color_candidates=['#000000'])
        result=model.ask(rt,self.folder/'palette','select',p,[],['P0','P1'],
                         {'P0':self.p,'P1':self.p},self.opt,
                         candidates={k:{'fills':[None],'strokes':['#000000']} for k in ('P0','P1')},
                         source_to_crop_scale=2)
        self.assertEqual(len(rt.events),1);self.assertFalse(result['errors'])
        self.assertEqual(result['valid']['P0']['fill'],'#112233')
        self.assertEqual(result['valid']['P0']['stroke'],'#AABBCC')
        self.assertEqual(result['valid']['P1']['fill'],gradient_fill)

    def test_critic_and_refine_receive_effective_implicit_dash_cycle(self):
        counts={'critic':0}
        def respond(tag,p):
            mode=tag.split('_')[0]
            if mode=='critic':
                counts['critic']+=1
                self.assertIn('dash_pattern_px',p['targets'][0]['current_style'])
                return {'checks':[{'id':'P0','verdict':'revise' if counts['critic']==1 else 'accept',
                                   'source_evidence':'longer source dashes','rendered_difference':'short dense dashes',
                                   'suggestion':'lengthen painted segments while retaining gaps'}]}
            if mode=='refine':
                previous=p['targets'][0]['previous_style']
                np.testing.assert_allclose(previous['dash_pattern_px'],[4*previous['width_px'],3*previous['width_px']])
            return {'styles':[{'id':'P0','fill':None,'stroke':'#123456','width_px':4,
                               'dash':'dash' if mode=='select' else 'dash_dot'}]}
        with patch('shape_integration.styling.controller.export',side_effect=self.fake_export), ThreadPoolExecutor(max_workers=4) as pool:
            run_case(self.scene,self.source,self.folder/'implicit_dash',FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(counts['critic'],2)
        self.assert_style_context_matches_snapshot(self.folder/'implicit_dash')

    def test_retry_and_cache_never_double_convert_lengths(self):
        calls=[]
        def respond(tag,p):
            calls.append(tag)
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#000000','width_px':10,
                               'dash':'dash','dash_pattern_px':[30,15]} if t['id']=='P0' or len(calls)>1 else
                              {'id':'P1','fill':'bad','stroke':None} for t in p['targets']]}
        class CachedRuntime(FakeRuntime):
            def __init__(self):super().__init__(respond);self.cache={}
            def call(self,root,tag,system,prompt,images,validator,budget):
                if tag in self.cache:return validator(copy.deepcopy(self.cache[tag]))
                result=super().call(root,tag,system,prompt,images,validator,budget)
                self.cache[tag]=copy.deepcopy(result);return result
        rt=CachedRuntime();p=model.prompt('refine',[{'id':'P0'},{'id':'P1'}],{'length_unit':'pixels of the main CLEAN crop'})
        for _ in range(2):
            out=model.ask(rt,self.folder/'cache','refine',p,[],['P0','P1'],{'P0':self.p,'P1':self.p},self.opt,source_to_crop_scale=5)
            for row in out['valid'].values():
                self.assertEqual(row['width_px'],2);self.assertEqual(row['dash_pattern_px'],[6,3])
        self.assertEqual(len(calls),2)
        self.assertEqual(rt.events[1][1]['format_correction']['previous_response'][0]['width_px'],10)
        audit=read(self.folder/'cache/validation.json')
        self.assertEqual(audit['model_valid']['P0']['width_px'],10)
        self.assertEqual(audit['valid']['P0']['width_px'],2)

    def test_uniform_crop_resampling_preserves_measured_width_and_position(self):
        image=Image.new('RGB',(200,200),'white');draw=ImageDraw.Draw(image)
        draw.rectangle([70,0,73,199],fill='black');draw.rectangle([0,90,199,93],fill='black')
        crop=[20,10,181,139]
        for size in ((1000,801),(101,81)):
            with self.subTest(size=size):
                scale,offset=views.crop_transform(crop,size)
                result=np.asarray(views.resample_crop(image,crop,size,scale,offset),float)[:,:,0]
                def local(x,y):return ((x-crop[0]+.5)*scale-.5+offset[0],(y-crop[1]+.5)*scale-.5+offset[1])
                x,y=local(40,40)
                profile_x=1-result[round(y)]/255;profile_y=1-result[:,round(x)]/255
                # Ignore padded crop edges. Integrals measure actual resampled ink,
                # not just the transform metadata under test.
                profile_x[:10]=profile_x[-10:]=0;profile_y[:10]=profile_y[-10:]=0
                self.assertAlmostEqual(float(profile_x.sum()/scale),4,delta=.25)
                self.assertAlmostEqual(float(profile_y.sum()/scale),4,delta=.25)
                self.assertAlmostEqual(float(np.dot(np.arange(size[0]),profile_x)/profile_x.sum()),local(71.5,40)[0],delta=.6)
                self.assertAlmostEqual(float(np.dot(np.arange(size[1]),profile_y)/profile_y.sum()),local(40,91.5)[1],delta=.6)

    def test_pdf_crop_registration_at_both_zoom_directions_and_page_edges(self):
        pdf=self.folder/'cross.pdf';source_size=(240,180)
        # PDF physical units deliberately differ from original pixels.
        with fitz.open() as doc:
            page=doc.new_page(width=480,height=360)
            page.draw_rect(fitz.Rect(140,0,148,360),color=None,fill=(0,0,0))
            page.draw_rect(fitz.Rect(0,180,480,188),color=None,fill=(0,0,0))
            doc.save(pdf)
        for crop,size in [([20,10,181,139],(1000,801)),([20,10,181,139],(101,81)),
                          ([0,0,181,139],(1000,768))]:
            with self.subTest(crop=crop,size=size):
                scale,offset=views.crop_transform(crop,size)
                image,meta=render_pdf_crop(pdf,source_size,crop,size,scale,offset)
                a=np.asarray(image,float)[:,:,0]
                local=lambda x,y:((x-crop[0])*scale-.5+offset[0],(y-crop[1])*scale-.5+offset[1])
                x,y=local(40,40);px=1-a[round(y)]/255;py=1-a[:,round(x)]/255
                self.assertEqual(image.size,size)
                self.assertAlmostEqual(float(px.sum()/scale),4,delta=.4)
                self.assertAlmostEqual(float(py.sum()/scale),4,delta=.4)
                self.assertAlmostEqual(float(np.dot(np.arange(size[0]),px)/px.sum()),local(72,40)[0],delta=.6)
                self.assertAlmostEqual(float(np.dot(np.arange(size[1]),py)/py.sum()),local(40,92)[1],delta=.6)
                self.assertEqual(meta['sha256'],sha(pdf))

    def test_pdf_crop_keeps_thin_vector_stroke_instead_of_enlarging_raster(self):
        pdf=self.folder/'thin.pdf';source_size=(200,160)
        with fitz.open() as doc:
            page=doc.new_page(width=200,height=160)
            page.draw_line(fitz.Point(80.5,0),fitz.Point(80.5,160),color=(0,0,0),width=.25)
            doc.save(pdf)
            pix=page.get_pixmap(alpha=False)
            low=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        crop=[20,20,180,140];size=(1000,750);scale,offset=views.crop_transform(crop,size)
        enlarged=views.resample_crop(low,crop,size,scale,offset)
        direct,_=render_pdf_crop(pdf,source_size,crop,size,scale,offset)
        profile=1-np.asarray(direct,float)[375,:,0]/255
        blurred=1-np.asarray(enlarged,float)[375,:,0]/255
        self.assertGreater(float(profile.max()),float(blurred.max())+.3)
        self.assertAlmostEqual(float(profile.sum()/scale),.25,delta=.08)

    def test_pdf_crop_handles_blank_page_and_rejects_non_pdf_or_multiple_slides(self):
        pdf=self.folder/'blank.pdf'
        with fitz.open() as doc:
            doc.new_page(width=240,height=180);doc.save(pdf)
        image,_=render_pdf_crop(pdf,(240,180),[0,0,80,60],(320,240),4,(0,0))
        self.assertTrue(np.all(np.asarray(image)==255))
        with self.assertRaisesRegex(ValueError,'native PPT PDF'):
            render_pdf_crop(self.source,(240,180),[0,0,80,60],(320,240),4,(0,0))
        with fitz.open(pdf) as doc:
            doc.new_page(width=240,height=180);doc.save(self.folder/'two.pdf')
        with self.assertRaisesRegex(ValueError,'single-slide'):
            render_pdf_crop(self.folder/'two.pdf',(240,180),[0,0,80,60],(320,240),4,(0,0))

    def test_whole_refine_receives_previous_styles_in_current_crop_units(self):
        entries=[rectangle('A',str(i),50+i*130,85,25,55) for i in range(2)]
        scene=scene_of(entries,source_hash=sha(self.source));refined=[]
        def respond(tag,p):
            mode=tag.split('_')[0]
            self.assertEqual(p['units']['length_unit'],'pixels of the main CLEAN crop')
            self.assertNotIn('source_to_display_scale_xy',p['units'])
            if mode=='critic':
                return {'checks':[{'id':t['id'],'verdict':'revise' if t['id']=='P1' and not refined else 'accept',
                                  'source_evidence':'source has a fine black edge','rendered_difference':'P1 is too thick',
                                  'suggestion':'Make P1 thinner to match its source edge'} for t in p['targets']]}
            if mode=='refine':
                refined.extend(t['id'] for t in p['targets'])
                for t in p['targets']:
                    self.assertAlmostEqual(t['previous_style']['width_px'],13.7)
                    np.testing.assert_allclose(t['previous_style']['dash_pattern_px'],[21,8])
            else:
                for t in p['targets']:
                    self.assertNotIn('candidates',t);self.assertNotIn('rule',t);self.assertNotIn('reasons',t)
                    self.assertNotIn('previous_style',t)
            return {'styles':[{'id':t['id'],'fill':None,'stroke':'#000000' if mode=='refine' else p['color_candidates'][-1],
                              'width_px':13.7,'dash':'dash','dash_pattern_px':[21,8]} for t in p['targets']]}
        with patch('shape_integration.styling.calibration.calibrate',side_effect=lambda im,p,s,e:(dict(s),{'apply':False})), ThreadPoolExecutor(max_workers=4) as pool:
            run_case(scene,self.source,self.folder/'units',FakeRuntime(respond),self.opt,pool,threading.BoundedSemaphore(1))
        initial=read(self.folder/'units/objects/A/revision_00/select_000/input/frame.json')['source_to_crop_scale']
        final=read(self.folder/'units/objects/A/revision_01/refine_000/input/frame.json')['source_to_crop_scale']
        self.assertEqual(initial,final)
        self.assertEqual(refined,['P0','P1'])
        styles=read(self.folder/'units/styles.json')['parts']
        self.assertAlmostEqual(styles[entries[1][0]]['style']['width_px'],13.7/final)
        self.assertAlmostEqual(styles[entries[0][0]]['style']['width_px'],13.7/initial)
        np.testing.assert_allclose(styles[entries[1][0]]['style']['dash_pattern_px'],[21/final,8/final])
        self.assert_style_context_matches_snapshot(self.folder/'units')

    def test_calibration_switch_covers_selection_and_refine_without_skipping_units(self):
        self.assertFalse(options({})['pixel_calibration_enabled'])
        for invalid in ('false',0,None):
            with self.assertRaisesRegex(ValueError,'pixel_calibration_enabled'):
                options({'styling':{'pixel_calibration_enabled':invalid}})
        evidence={'style':{'fill':None,'stroke':'#123456','width_px':1,'dash':'solid'},
                  'candidates':{'fills':[None],'strokes':['#123456']},'needs_model':True,'reasons':[]}
        def calibration_result(image,part,style,excluded):
            return {**style,'width_px':style['width_px']+100,'dash_pattern_px':[100,50]}, {'changed':True,'apply':True}
        for enabled in (False,True):
            with self.subTest(enabled=enabled):
                counts={'critic':0}
                def respond(tag,p):
                    mode=tag.split('_')[0]
                    if mode=='critic':
                        counts['critic']+=1
                        return {'checks':[{'id':t['id'],'verdict':'revise' if counts['critic']==1 else 'accept',
                                           'source_evidence':'visible blue dashed edge','rendered_difference':'edge too thin',
                                           'suggestion':'match the visible line weight'} for t in p['targets']]}
                    return {'styles':[{'id':t['id'],'fill':None,'stroke':'#CC5533',
                                       'width_px':10 if mode=='select' else 12,'dash':'dash',
                                       'dash_pattern_px':[30,15]} for t in p['targets']]}
                folder=self.folder/('calibration_on' if enabled else 'calibration_off')
                opt={**self.opt,'pixel_calibration_enabled':enabled}
                with patch('shape_integration.styling.rules.estimate',return_value=evidence), \
                     patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
                     patch('shape_integration.styling.calibration.calibrate',side_effect=calibration_result) as calibrator, \
                     ThreadPoolExecutor(max_workers=4) as pool:
                    summary=run_case(self.scene,self.source,folder,FakeRuntime(respond),opt,pool,threading.BoundedSemaphore(1))
                self.assertEqual(calibrator.call_count,2 if enabled else 0)
                self.assertEqual(summary['pixel_calibration_enabled'],enabled)
                for revision,mode,width in [('revision_00','select',10),('revision_01','refine',12)]:
                    scale=read(folder/'objects/A'/revision/(mode+'_000')/'input/frame.json')['source_to_crop_scale']
                    style=read(folder/'objects/A'/revision/'styles.json')[self.k]
                    self.assertAlmostEqual(style['width_px'],width/scale+(100 if enabled else 0))
                    np.testing.assert_allclose(style['dash_pattern_px'],[100,50] if enabled else [30/scale,15/scale])
                    audit=read(folder/'objects/A'/revision/'calibration.json')[self.k]
                    self.assertEqual(audit['enabled'],enabled)
                    if not enabled:self.assertEqual(audit['reason'],'disabled_by_config')
                history=read(folder/'styles.json')['parts'][self.k]['history']
                self.assertEqual(sum(h['stage']=='pixel_calibration' for h in history),2 if enabled else 0)
                job=folder/'objects/A/revision_01/refine_000'
                context=read(job/'job.json')['prompt']['targets'][0]['previous_style']
                scale=read(job/'input/frame.json')['source_to_crop_scale']
                rendered_style=read(folder/read(job/'job.json')['render_snapshot']/'styles.json')[self.k]
                self.assertAlmostEqual(context['width_px'],rendered_style['width_px']*scale)
                np.testing.assert_allclose(context['dash_pattern_px'],np.array(rendered_style['dash_pattern_px'])*scale)
                self.assertEqual(context['fill'],rendered_style['fill'])
                self.assertEqual(context['stroke'],rendered_style['stroke'])
                self.assertEqual(context['dash'],rendered_style['dash'])
                self.assert_style_context_matches_snapshot(folder)

    def test_palette_counts_real_pixels_top_fifteen_without_overlap_double_votes(self):
        pixels=[]
        for i in range(20):pixels += [(i*11,20,30)]*(20-i)
        im=Image.fromarray(np.array(pixels,dtype=np.uint8).reshape(10,21,3))
        a=rectangle('A','one',10,4.5,20,9);b=rectangle('B','two',10,4.5,20,9)
        result=rules.ranked_source_colors(im,dict([a,b]),lambda k:GeometryCollection())
        self.assertEqual(result['pixel_count'],210)
        self.assertEqual([x['pixel_count'] for x in result['colors']],list(range(20,5,-1)))
        self.assertEqual([x['color'] for x in result['colors']],['#%02X141E'%(i*11) for i in range(15)])

    def test_repeated_component_group_shares_one_style_compound_and_unique_stay_ordinary(self):
        entries=[rectangle('B'+str(i),'face',35+i*60,70,25,40) for i in range(3)]
        entries += [rectangle('A',str(i),215,50+i*65,20,30) for i in range(2)]
        draw=ImageDraw.Draw(self.image)
        for i,(_,part) in enumerate(entries):draw.rectangle(geometry.bounds(part),fill='#335577' if i<2 else '#CC5533')
        self.image.save(self.source)
        scene=scene_of(entries,source_hash=sha(self.source));refined=[]
        def respond(tag,p):
            if tag.startswith('group'):
                self.assertEqual(len(p['targets']),3)
                return {'groups':[{'members':['C0','C1'],'representative':'C1','evidence':'both blue faces have matching borders'}]}
            if tag.startswith('critic'):
                revise=bool(p['targets'][0].get('shared_with')) and not refined
                return {'checks':[{'id':t['id'],'verdict':'revise' if revise else 'accept',
                                  'source_evidence':'same blue border on both','rendered_difference':'both borders are thin',
                                  'suggestion':'match both border weights'} for t in p['targets']]}
            if p['targets'][0].get('shared_with'):
                if tag.startswith('refine'):refined.append(True)
                self.assertEqual(len(p['targets']),1)
                return {'styles':[{'id':'P0','fill':'#335577','stroke':'#335577','width_px':8 if refined else 4}]}
            return {'styles':[{'id':t['id'],'fill':None,'stroke':None} for t in p['targets']]}
        rt=FakeRuntime(respond)
        with patch('shape_integration.styling.controller.export',side_effect=self.fake_export), ThreadPoolExecutor(max_workers=4) as pool:
            run_case(scene,self.source,self.folder/'groups',rt,self.opt,pool,threading.BoundedSemaphore(1))
        final=read(self.folder/'groups/styles.json');records=final['parts']
        self.assertEqual(records[entries[0][0]]['style'],records[entries[1][0]]['style'])
        self.assertEqual(records[entries[0][0]]['style_group'],'S001')
        self.assertNotIn('style_group',records[entries[2][0]])
        self.assertTrue((self.folder/'groups/objects/A/revision_00/select_000/job.json').exists())
        self.assertTrue((self.folder/'groups/objects/B2/revision_00/critic_000/job.json').exists())
        self.assertEqual(set(final['style_groups'][0]['members']),{entries[0][0],entries[1][0]})
        shared_calls=[p for tag,p in rt.events if tag.startswith(('select','refine')) and p['targets'][0].get('shared_with')]
        self.assertEqual(len(shared_calls),2)
        self.assertNotIn('previous_style',shared_calls[0]['targets'][0])
        self.assertAlmostEqual(shared_calls[1]['targets'][0]['previous_style']['width_px'],4)
        self.assertEqual(shared_calls[1]['targets'][0]['previous_style']['fill'],'#335577')
        frame=read(self.folder/'groups/component_groups/S001/revision_01/refine_000/input/frame.json')
        self.assertAlmostEqual(records[entries[0][0]]['style']['width_px'],8/frame['source_to_crop_scale'])
        self.assert_style_context_matches_snapshot(self.folder/'groups')

    def test_grouping_protocol_rejects_duplicate_or_unknown_members(self):
        mapping={'C0':'a','C1':'b','C2':'c'}
        group={'members':['C0','C1'],'representative':'C0','evidence':'matching source appearances'}
        for groups in ([group,group],[{**group,'members':['C0','X']}],[{**group,'representative':'C2'}]):
            with self.assertRaises(ValueError):grouping.validate({'groups':groups},mapping)
        self.assertEqual(grouping.validate({'groups':[]},mapping),{'groups':[]})

    def test_grouping_failure_falls_back_and_never_retries_timeout(self):
        entries=[rectangle('A','one',45,85,30,40),rectangle('B','one',155,85,30,40)]
        scene=scene_of(entries,source_hash=sha(self.source));rt=FakeRuntime(lambda tag,p:RequestTimeout('test'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=grouping.discover(scene,self.source,dict(entries),set(dict(entries)),self.folder/'failed_groups',rt,self.opt,pool)
        self.assertEqual(len(rt.events),1);self.assertFalse(result['groups'])
        self.assertEqual(result['errors'][0]['fallback'],'ordinary_pipeline')

    def test_large_object_view_has_one_scale_and_no_auxiliary_edge_board(self):
        source=self.folder/'large.png';Image.new('RGB',(600,400),'white').save(source)
        key,part=rectangle('large','face',300,200,500,300)
        images,_,frame=views.prepare(source,{key:part},{'P0':key},self.folder/'large_view',self.opt)
        self.assertEqual(len(images),2)
        self.assertFalse((self.folder/'large_view/edges.png').exists())
        self.assertTrue(all(list(Image.open(p).size)==frame['display_size'] for _,p in images))


    def test_fast_unit_finishes_refine_and_publishes_while_slow_unit_is_selecting(self):
        entries=[rectangle('fast','face',40,80,30,40),rectangle('slow','face',195,80,30,40)]
        scene=scene_of(entries,source_hash=sha(self.source));folder=self.folder/'parallel'
        slow_started=threading.Event();fast_done=threading.Event();events=[];errors=[];critic_count={}
        evidence={'style':{'fill':None,'stroke':'#CC5533','width_px':1,'dash':'solid'},
                  'candidates':{'fills':[None],'strokes':['#CC5533']},'needs_model':True,'reasons':[]}
        def ask(rt,root,mode,p,images,expected,table,opt,**kwargs):
            ident=root.parents[2].name;events.append((ident,mode))
            if ident=='slow' and mode=='select':
                slow_started.set()
                if not fast_done.wait(8):errors.append('fast unit was blocked behind slow select')
            if mode=='critic':
                critic_count[ident]=critic_count.get(ident,0)+1
                if ident=='fast' and critic_count[ident]==2:
                    if not slow_started.is_set():errors.append('slow unit never started concurrently')
                    if not (folder/'styled.pptx').exists():errors.append('no live PPT during slow select')
                    if (folder/'summary.json').exists():errors.append('preview appeared only after completion')
                    actual=read(folder/'current/styles.json')[entries[0][0]]
                    if actual['fill']!='#335577':errors.append('live PPT did not contain refined styles')
                    fast_done.set()
                value={'verdict':'revise' if ident=='fast' and critic_count[ident]==1 else 'accept',
                       'source_evidence':'blue face','rendered_difference':'wrong fill','suggestion':'match blue fill'}
            else:
                value={'fill':'#335577' if mode=='refine' else None,'stroke':'#CC5533','width_px':2,'dash':'solid'}
            return {'valid':{a:value for a in expected},'errors':{}}
        with patch('shape_integration.styling.rules.estimate',return_value=evidence), \
             patch('shape_integration.styling.model.ask',side_effect=ask), \
             patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
             ThreadPoolExecutor(max_workers=2) as pool:
            summary=run_case(scene,self.source,folder,FakeRuntime(None),
                {**self.opt,'component_grouping_enabled':False},pool,threading.BoundedSemaphore(1))
        self.assertFalse(errors);self.assertTrue(fast_done.is_set());self.assertEqual(summary['critic_accepted'],2)
        self.assertEqual([mode for ident,mode in events if ident=='fast'],['select','critic','refine','critic'])
        self.assertEqual(critic_count['slow'],1)
        initial=read(folder/'objects/fast/revision_00/critic_000/job.json')
        refine=read(folder/'objects/fast/revision_01/refine_000/job.json')
        self.assertEqual(initial['render_snapshot'],refine['render_snapshot'])
        self.assertEqual(read(folder/'progress.json')['completed_units'],2)
        self.assertEqual(read(folder/'live.json')['sequence'],read(folder/'styles.json')['sequence'])
        self.assert_style_context_matches_snapshot(folder)

    def test_later_neighbour_change_rechecks_the_original_accepted_crop(self):
        entries=[rectangle('outer','face',100,85,140,100),rectangle('inner','face',100,85,30,30)]
        scene=scene_of(entries,source_hash=sha(self.source));folder=self.folder/'neighbour'
        outer_checked=threading.Event();events=[]
        evidence={'style':{'fill':None,'stroke':'#CC5533','width_px':1,'dash':'solid'},
                  'candidates':{'fills':[None],'strokes':['#CC5533']},'needs_model':True,'reasons':[]}
        def ask(rt,root,mode,p,images,expected,table,opt,**kwargs):
            ident=root.parents[2].name;revision=root.parents[1].name;events.append((ident,mode,revision))
            if ident=='inner' and mode=='select':self.assertTrue(outer_checked.wait(8))
            if mode=='critic':
                if ident=='outer':outer_checked.set()
                value={'verdict':'accept'}
            else:value={'fill':'#335577' if ident=='inner' else None,'stroke':'#CC5533','width_px':2,'dash':'solid'}
            return {'valid':{a:value for a in expected},'errors':{}}
        with patch('shape_integration.styling.rules.estimate',return_value=evidence), \
             patch('shape_integration.styling.model.ask',side_effect=ask), \
             patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
             ThreadPoolExecutor(max_workers=2) as pool:
            summary=run_case(scene,self.source,folder,FakeRuntime(None),
                {**self.opt,'component_grouping_enabled':False},pool,threading.BoundedSemaphore(1))
        self.assertIn(('outer','critic','context_recheck'),events)
        self.assertFalse(any(mode=='refine' for _,mode,_ in events))
        final=read(folder/'objects/outer/context_recheck/critic_000/job.json')['render_snapshot']
        self.assertEqual(final,summary['selected_revision'])
        self.assertEqual(read(folder/final/'styles.json')[entries[1][0]]['fill'],'#335577')
        self.assert_style_context_matches_snapshot(folder)

    def test_failed_render_keeps_last_published_ppt_and_saves_new_styles(self):
        folder=self.folder/'render_failure';calls=[]
        evidence={'style':{'fill':None,'stroke':'#CC5533','width_px':1,'dash':'solid'},
                  'candidates':{'fills':[None],'strokes':['#CC5533']},'needs_model':True,'reasons':[]}
        def ask(rt,root,mode,p,images,expected,table,opt,**kwargs):
            if mode=='critic':value={'verdict':'revise','source_evidence':'blue fill',
                                    'rendered_difference':'missing blue','suggestion':'use blue'}
            else:value={'fill':'#335577' if mode=='refine' else None,'stroke':None,'width_px':0,'dash':'solid'}
            return {'valid':{a:value for a in expected},'errors':{}}
        def fail_export(scene,styles,rel,root,*args):
            if styles[self.k]['fill']=='#335577':
                calls.append(sha(folder/'styled.pptx'));raise RuntimeError('native render unavailable')
            return self.fake_export(scene,styles,rel,root,*args)
        with patch('shape_integration.styling.rules.estimate',return_value=evidence), \
             patch('shape_integration.styling.model.ask',side_effect=ask), \
             patch('shape_integration.styling.controller.export',side_effect=fail_export), \
             ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaisesRegex(RuntimeError,'previous live PPT retained'):
                run_case(self.scene,self.source,folder,FakeRuntime(None),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(len(calls),1);self.assertEqual(sha(folder/'styled.pptx'),calls[0])
        self.assertEqual(read(folder/'styles.json')['parts'][self.k]['style']['fill'],'#335577')
        self.assertIsNone(read(folder/'current/styles.json')[self.k]['fill'])
        self.assertFalse((folder/'summary.json').exists())
        # Resume the checkpoint without redoing any completed model stage.
        with patch('shape_integration.styling.model.ask',side_effect=AssertionError('completed model call replayed')), \
             patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
             ThreadPoolExecutor(max_workers=1) as pool:
            # A failed-render unit still needs its final visual check.
            with patch('shape_integration.styling.model.ask',return_value={'valid':{'P0':{'verdict':'accept'}},'errors':{}}) as reviewer:
                result=run_case(self.scene,self.source,folder,FakeRuntime(None),self.opt,pool,threading.BoundedSemaphore(1))
        self.assertEqual(reviewer.call_count,1);self.assertEqual(result['critic_accepted'],1)
        self.assertEqual(read(folder/'current/styles.json')[self.k]['fill'],'#335577')

    def test_progress_records_active_critic_before_response_arrives(self):
        folder=self.folder/'active_critic';visible=threading.Event();failures=[]
        def ask(rt,root,mode,p,images,expected,table,opt,**kwargs):
            if mode=='critic':
                if not visible.wait(3):failures.append('critic phase not saved while request active')
                value={'verdict':'accept'}
            else:value={'fill':None,'stroke':None,'width_px':0,'dash':'solid'}
            return {'valid':{a:value for a in expected},'errors':{}}
        def progress(stage,done,total):
            if stage=='components' and read(folder/'progress.json')['units']['objects/A']['stage']=='critic':
                saved=read(folder/'scheduler.json')
                if saved['pending'][0]['mode']=='critic':visible.set()
        with patch('shape_integration.styling.model.ask',side_effect=ask), \
             patch('shape_integration.styling.controller.export',side_effect=self.fake_export), \
             ThreadPoolExecutor(max_workers=1) as pool:
            run_case(self.scene,self.source,folder,FakeRuntime(None),self.opt,pool,
                     threading.BoundedSemaphore(1),progress=progress)
        self.assertFalse(failures);self.assertTrue(visible.is_set())

    def test_live_publication_rejects_incomplete_snapshot_before_switching(self):
        from shape_integration.styling.live import publish
        folder=self.folder/'publish';old=folder/'snapshots/old';new=folder/'snapshots/new'
        for target in (old,new):
            self.fake_export(self.scene,{self.k:{'version':target.name}},None,target,None,None,None)
            save(target/'styles.json',{'version':target.name});save(target/'calibration.json',{})
        publish(folder,old,0);before=sha(folder/'styled.pptx')
        (new/'native.pdf').write_text('truncated')
        with self.assertRaisesRegex(ValueError,'Incomplete or changed'):
            publish(folder,new,1)
        self.assertEqual(sha(folder/'styled.pptx'),before)
        self.assertEqual(read(folder/'live.json')['sequence'],0)
        self.assertEqual((folder/'current').resolve(),old.resolve())


if __name__=='__main__':unittest.main()
