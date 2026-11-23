"""Cylinder protocol, fit, affine transport, part edits and native rendering."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.fitting.adapter import fit_object,transform_fit
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.scene import native_parameters,write_pptx
from shape_integration.fitting.paths import command_strokes
from shape_integration.fitting.contracts import normalize_observation
from shape_integration.fitting.revision import merge_revision


def fixture(top_height=30,depth=50):
    t=np.linspace(-np.pi/2,3*np.pi/2,8,endpoint=False)
    xy=np.c_[240+180*np.cos(t),80+top_height/2*np.sin(t)].tolist()+[[420,80+depth],[240,80+depth+top_height/2],[60,80+depth]]
    raw={'status':'ok','points':[{'id':f'p{i}','xy':p} for i,p in enumerate(xy)],
         'parts':[{'id':'body','kind':'cylinder','point_ids':[f'p{i}' for i in range(11)]},
                  {'id':'top','kind':'ellipse','point_ids':[f'p{i}' for i in range(8)]}]}
    frame=FitFrame((500,200),(500,200),(500,200),(0,0,500,200))
    raw=normalize_observation(raw,'T',frame.input_size)
    return raw,frame


class CylinderTests(unittest.TestCase):
    def test_partial_recovery_keeps_valid_cylinder_and_cap_together(self):
        raw,frame=fixture();before=copy.deepcopy(raw)
        raw['parts'].append({'id':'bad','kind':'ellipse','point_ids':['missing']*8})
        audit={}
        actual=normalize_observation(raw,'T',frame.input_size,recover_parts=True,audit=audit)
        self.assertEqual(actual['parts'],before['parts'])
        self.assertEqual(actual['points'],before['points'])
        self.assertEqual(actual['status'],'partial')
        self.assertEqual([p['id'] for p in audit['rejected_parts']],['bad'])

    def test_partial_recovery_does_not_reorder_or_invent_cylinder_caps(self):
        raw,frame=fixture()
        raw['parts'].reverse()
        audit={}
        actual=normalize_observation(raw,'T',frame.input_size,recover_parts=True,audit=audit)
        self.assertEqual([p['id'] for p in actual['parts']],['top'])
        self.assertEqual([p['id'] for p in audit['rejected_parts']],['body'])
        raw,frame=fixture();raw['parts'].append({'id':'uncapped','kind':'cylinder',
                                                'point_ids':list(reversed(raw['parts'][0]['point_ids']))})
        audit={}
        actual=normalize_observation(raw,'T',frame.input_size,recover_parts=True,audit=audit)
        self.assertEqual([p['id'] for p in actual['parts']],['body','top'])
        self.assertEqual([p['id'] for p in audit['rejected_parts']],['uncapped'])

    def test_protocol_fit_transport_and_native_preset(self):
        raw,frame=fixture();local=fit_object(raw,frame,object_id='T',namespace='test')
        self.assertEqual(local['fit_status'],'ok')
        original=transform_fit(local);part=original['parts'][0]
        native=native_parameters(part,frame.metadata());part['native_parameters_original']=native
        self.assertEqual(native['preset'],'can')
        self.assertAlmostEqual(native['width'],360,places=4)
        self.assertAlmostEqual(native['height'],80,places=4)
        self.assertAlmostEqual(native['adjustments'][0],30/80,places=4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'can.pptx';scene={'source_size':[500,200],
                'objects':{'T':{'id':'T','state':'accepted_by_model','part_keys':['p']}},'parts':{'p':part}}
            self.assertEqual(write_pptx(scene,path)['native'],1)
            geom=Presentation(path).slides[0].shapes[0]._element.spPr.prstGeom
            self.assertEqual(geom.get('prst'),'can')
            self.assertAlmostEqual(int(geom.find('{http://schemas.openxmlformats.org/drawingml/2006/main}avLst')[0].get('fmla').split()[1])/100000,30/80,places=4)

    def test_revision_preserves_unrelated_geometry_and_accepts_cylinder(self):
        raw,frame=fixture();context={'base_observation':copy.deepcopy(raw),'editable_ids':['body','top'],
            'remove_ids':[],'preserved_ids':[],'allow_new_parts':False,'source_revision':'fixture'}
        merged=merge_revision(copy.deepcopy(raw),context,'T',frame.input_size,audit={})
        self.assertEqual(merged,raw)
        # A tilted native cylinder transported through unequal scales must
        # retain the affine custom path, not a distorted Office preset.
        local=fit_object(raw,frame,object_id='T',namespace='x');part=transform_fit(local)['parts'][0]
        part['parameters_local']['angle_radians']=.2
        anisotropic=FitFrame((500,200),(500,200),(1000,200),(0,0,500,200))
        self.assertIsNone(native_parameters(part,anisotropic.metadata()))

    def test_can_handle_limit_and_missing_top_face_never_distort_a_flat_disc(self):
        from shape_integration.fitting.contracts import validate_observation,ObservationError
        raw,frame=fixture(50,30)
        part=transform_fit(fit_object(raw,frame,object_id='T',namespace='x'))['parts'][0]
        self.assertIsNone(native_parameters(part,frame.metadata()))
        raw['parts']=raw['parts'][:1]
        with self.assertRaisesRegex(ObservationError,'top-face ellipse'):
            validate_observation(raw,'T',frame.input_size)
        raw['status']='partial';raw['reason']='Top face omitted'
        part=transform_fit(fit_object(raw,frame,object_id='T',namespace='x'))['parts'][0]
        self.assertIsNone(native_parameters(part,frame.metadata()))

    def test_transparent_or_reordered_top_face_cannot_expose_automatic_lightening(self):
        from shape_integration.styling.export import source_safe_native_scene
        raw,frame=fixture();parts=transform_fit(fit_object(raw,frame,object_id='T',namespace='x'))['parts']
        for p in parts:p['native_parameters_original']=native_parameters(p,frame.metadata())
        scene={'parts':{'body':parts[0],'top':parts[1]}};before=copy.deepcopy(scene)
        styles={'body':{'fill':'#2266CC'},'top':{'fill':'#2266CC'}};relations={'flat_order':['body','top']}
        self.assertFalse(source_safe_native_scene(scene,styles,relations)[1])
        for top in ({'fill':None},{'fill':'#2266CC','fill_opacity':.5}):
            actual,audit=source_safe_native_scene(scene,{**styles,'top':top},relations)
            self.assertIsNone(actual['parts']['body']['native_parameters_original']);self.assertEqual(len(audit),1)
            self.assertEqual(actual['parts']['body']['commands'],parts[0]['commands'])
        self.assertEqual(len(source_safe_native_scene(scene,styles,{'flat_order':['top','body']})[1]),1)
        self.assertEqual(scene,before)

    def test_style_sampling_does_not_join_seam_to_silhouette_and_occlusion_uses_body(self):
        from shape_integration.styling.rules import contour_samples
        from shape_integration.styling.geometry import footprint
        from full_recovery.handoff import source_occlusion
        raw,frame=fixture();part=transform_fit(fit_object(raw,frame,object_id='T',namespace='x'))['parts'][0]
        pts,_,_=contour_samples(part)
        self.assertFalse(np.any((abs(pts[:,1]-80)<.5)&(pts[:,0]>100)&(pts[:,0]<380)))
        geometry,error=footprint(part);self.assertIsNone(error);self.assertGreater(geometry.area,0)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.png';im=Image.new('RGB',(500,200),'white')
            outline=list(command_strokes(part['commands'],128))[-1]
            ImageDraw.Draw(im).polygon([tuple(p) for p in outline],fill='#286ED2');im.save(source)
            arrow={'id':'a','start':[10,110],'end':[490,110],'style':{'color':'#000000'}}
            plan,_=source_occlusion({'parts':{'body':part}},{'body':{'fill':'#286ED2'}},[arrow],source,set(),part_placements=True)
            self.assertEqual(plan['a']['before'],['T/body'])

    @unittest.skipUnless(__import__('shutil').which('libreoffice'),'Native renderer required')
    def test_native_can_render_agrees_with_fitted_silhouette(self):
        from shape_integration.styling.export import render_native
        raw,frame=fixture();parts=transform_fit(fit_object(raw,frame,object_id='T',namespace='x'))['parts']
        for part in parts:part['native_parameters_original']=native_parameters(part,frame.metadata())
        part=parts[0]
        scene={'source_size':[500,200],'objects':{'T':{'id':'T','state':'accepted_by_model','part_keys':['p','cap']}},'parts':{'p':part,'cap':parts[1]}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'can.pptx'
            def paint(shape,part,scale):
                shape.fill.solid();shape.fill.fore_color.rgb=RGBColor(40,110,210);shape.line.fill.background()
            write_pptx(scene,path,style_writer=paint)
            actual=np.asarray(Image.open(render_native(path,Path(tmp)/'render',[500,200])).convert('RGB'))
            target=Image.new('L',(500,200));points=list(command_strokes(part['commands'],128))[-1]
            ImageDraw.Draw(target).polygon([tuple(p) for p in points],fill=255)
            a=np.asarray(target)>0;b=(actual[:,:,0]<100)&(actual[:,:,2]>150)
            self.assertGreater(np.sum(a&b)/np.sum(a|b),.97)


if __name__=='__main__':unittest.main()
