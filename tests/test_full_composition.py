"""Native geometry, Office Math and linked previews must survive stage merging."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile,ZIP_DEFLATED

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR,MSO_SHAPE
from pptx.util import Inches

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from full_recovery.compose import merge,read_package,resolve,P,A,R,PR,MC,SLIDE,RELS
from full_recovery.resume import stage_completed

M='http://schemas.openxmlformats.org/officeDocument/2006/math'
A14='http://schemas.microsoft.com/office/drawing/2010/main'


def presentation(width=12,height=6):
    result=Presentation();result.slide_width=Inches(width);result.slide_height=Inches(height)
    result.slides.add_slide(result.slide_layouts[6]);return result


class CompositionTests(unittest.TestCase):
    def test_invalid_fitted_image_bbox_falls_back_to_valid_source_bbox(self):
        from png2pptx.images.detect import source_box
        from shape_integration.composition import crop_box
        asset={'id':'I0001','bbox_2d':[-0.02,20,40,60],'bbox':[0,20,40,60]}
        inventory={'input_size':[100,100]}
        self.assertEqual(crop_box(asset,inventory,[200,200]),
                         source_box(asset['bbox'],inventory['input_size'],[200,200]))

    def test_pattern_fill_preserves_native_geometry_and_stroke(self):
        from full_recovery.texture_fallback import apply_patterns
        p=presentation();s=p.slides[0].shapes.add_shape(MSO_SHAPE.OCTAGON,100,200,300,400)
        s.name='owner/surface [accepted_by_model]';p.save(self.base)
        before=etree.fromstring(read_package(self.base)[SLIDE])
        transform=etree.tostring(before.find('.//{'+P+'}sp/{'+P+'}spPr/{'+A+'}xfrm'))
        apply_patterns(self.base,self.output,[{'shape_name':'owner/surface','pattern':'smCheck',
                                              'foreground':'#777777','background':'#FFFFFF'}])
        after=etree.fromstring(read_package(self.output)[SLIDE])
        self.assertEqual(etree.tostring(after.find('.//{'+P+'}sp/{'+P+'}spPr/{'+A+'}xfrm')),transform)
        self.assertEqual(after.find('.//{'+A+'}pattFill').get('prst'),'smCheck')
        self.assertEqual(len(after.findall('.//{'+P+'}sp')),1)
        with ZipFile(self.output) as archive:self.assertFalse(any(n.startswith('ppt/media/') for n in archive.namelist()))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.base=self.root/'shape.pptx';self.arrow=self.root/'arrow.pptx';self.text=self.root/'text.pptx';self.output=self.root/'full.pptx'
        p=presentation();p.slides[0].shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(1),Inches(1),Inches(3),Inches(2));p.save(self.base)

    def test_resume_keeps_image_registry_with_own_completed_shape(self):
        import json
        from full_recovery.compose import sha
        old=self.root/'old';new=self.root/'new';image=self.root/'original.png'
        Image.new('RGB',(80,60),'white').save(image)
        item={'id':'own','original':str(image),'source_sha256':sha(image),'source_size':[80,60]}
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        for root in (old,new):save(root/'source_manifest.json',{'images':[item]})
        save(old/'status.json',{'state':'failed','error':'143'})
        ppt=old/'shape/delivery/own/shapes.pptx';ppt.parent.mkdir(parents=True);ppt.write_bytes(self.base.read_bytes())
        save(old/'shape/geometry/cases/own/complete.json',{'pptx':str(ppt),'pptx_sha256':sha(ppt)})
        registry=old/'shape/localization/cases/own/dedup/image_boxes_original.json'
        save(registry,{'assets':[{'id':'I1','bbox_2d':[20,20,40,40]}]})
        # A completed PPT from the former whole-compound merge must not bypass
        # the new part-level placement when its upstream stages are resumed.
        whole=old/'cases/own';whole.mkdir(parents=True)
        (whole/'combined.pptx').write_bytes(self.base.read_bytes())
        save(whole/'completed.json',{'pptx_sha256':sha(whole/'combined.pptx')})
        result=stage_completed(old,new)
        self.assertIn('shape_localization',{row['stage'] for row in result['artifacts']})
        self.assertEqual((new/registry.relative_to(old)).read_bytes(),registry.read_bytes())
        self.assertFalse((new/'cases/own').exists())
        repaired=self.root/'repaired';save(repaired/'source_manifest.json',{'images':[item]})
        self.assertEqual(stage_completed(old,repaired,repair_shapes=['own'])['artifacts'],[])
        changed=self.root/'changed';save(changed/'source_manifest.json',{'images':[{**item,'source_sha256':'different'}]})
        with self.assertRaisesRegex(ValueError,'provenance changed'):stage_completed(old,changed)

    def test_formula_structure_colors_preview_and_native_arrow_survive(self):
        p=presentation();arrow=p.slides[0].shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(1),Inches(1),Inches(4),Inches(3))
        ln=arrow._element.spPr.get_or_add_ln();etree.SubElement(ln,'{'+A+'}tailEnd',type='triangle',w='med',len='med');p.save(self.arrow)
        p=presentation();slide=p.slides[0];math=slide.shapes.add_textbox(Inches(1),Inches(1),Inches(2),Inches(1))
        container=etree.SubElement(math._element.txBody.p_lst[0],'{'+A14+'}m')
        equation=etree.SubElement(container,'{'+M+'}oMath');fraction=etree.SubElement(equation,'{'+M+'}f')
        for tag,text,color in [('num','x','2266CC'),('den','y','FF6600')]:
            run=etree.SubElement(etree.SubElement(fraction,'{'+M+'}'+tag),'{'+M+'}r')
            prop=etree.SubElement(run,'{'+A+'}rPr');etree.SubElement(etree.SubElement(prop,'{'+A+'}solidFill'),'{'+A+'}srgbClr',val=color)
            run.append(etree.fromstring(('<m:t xmlns:m="'+M+'">'+text+'</m:t>').encode(),etree.XMLParser()))
        image=BytesIO();Image.new('RGBA',(20,10),(255,100,0,255)).save(image,format='PNG');blob=image.getvalue();image.seek(0)
        preview=slide.shapes.add_picture(image,Inches(1),Inches(1),Inches(2),Inches(1))
        native=math._element;fallback=preview._element;tree=native.getparent();position=tree.index(native)
        tree.remove(native);tree.remove(fallback)
        alternate=etree.Element('{'+MC+'}AlternateContent',nsmap={'mc':MC,'a14':A14})
        choice=etree.SubElement(alternate,'{'+MC+'}Choice',Requires='a14');choice.append(native)
        etree.SubElement(alternate,'{'+MC+'}Fallback').append(fallback);tree.insert(position,alternate);p.save(self.text)
        before={path:path.read_bytes() for path in (self.base,self.arrow,self.text)}
        audit=merge(self.base,[('arrows',self.arrow),('text',self.text)],self.output)
        package=read_package(self.output);root=etree.fromstring(package[SLIDE])
        self.assertEqual(root.find('.//{'+A+'}tailEnd').get('type'),'triangle')
        self.assertEqual([n.text for n in root.iter('{'+M+'}t')],['x','y'])
        self.assertEqual([n.get('val') for n in root.iter('{'+A+'}srgbClr')][-2:],['2266CC','FF6600'])
        choice=root.find('.//{'+MC+'}Choice');self.assertEqual(choice.nsmap['a14'],A14)
        embed=root.find('.//{'+MC+'}Fallback//{'+A+'}blip').get('{'+R+'}embed')
        relationships=etree.fromstring(package[RELS]);target=next(n.get('Target') for n in relationships if n.get('Id')==embed)
        self.assertEqual(package[resolve(SLIDE,target)],blob)
        ids=[n.get('id') for n in root.iter('{'+P+'}cNvPr')];self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(Presentation(self.output).slides),1)
        self.assertTrue(audit['stage_sources_unchanged']);self.assertEqual(audit['layers'][-1]['native_math'],1)
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)

    def test_canvas_mapping_does_not_rebuild_paths_or_line_style(self):
        p=presentation(10,5);line=p.slides[0].shapes.add_connector(MSO_CONNECTOR.STRAIGHT,100,200,300,400)
        line.line.width=12700;p.save(self.arrow)
        original=etree.fromstring(read_package(self.arrow)[SLIDE]).find('.//{'+P+'}cxnSp')
        merge(self.base,[('arrows',self.arrow)],self.output)
        root=etree.fromstring(read_package(self.output)[SLIDE]);group=root.find('.//{'+P+'}grpSp')
        self.assertEqual(group.find('{'+P+'}grpSpPr/{'+A+'}xfrm/{'+A+'}chExt').get('cx'),str(Inches(12)))
        imported=group.find('{'+P+'}cxnSp')
        # Canonical group units materialize the existing 12/10 canvas scale;
        # world position, complete geometry and absolute stroke must agree.
        before=original.find('{'+P+'}spPr');after=imported.find('{'+P+'}spPr')
        for tag,attrs in [('off',('x','y')),('ext',('cx','cy'))]:
            a=before.find('{'+A+'}xfrm/{'+A+'}'+tag)
            b=after.find('{'+A+'}xfrm/{'+A+'}'+tag)
            for attr in attrs:self.assertEqual(int(b.get(attr)),round(int(a.get(attr))*1.2))
        self.assertEqual(int(after.find('{'+A+'}ln').get('w')),round(int(before.find('{'+A+'}ln').get('w'))*1.2))
        self.assertEqual(etree.tostring(before.find('{'+A+'}prstGeom')),etree.tostring(after.find('{'+A+'}prstGeom')))

    def test_external_links_are_preserved(self):
        p=presentation();shape=p.slides[0].shapes.add_textbox(0,0,10000,10000)
        run=shape.text_frame.paragraphs[0].add_run();run.text='link';run.hyperlink.address='https://example.invalid/reference';p.save(self.text)
        merge(self.base,[('text',self.text)],self.output)
        rels=etree.fromstring(read_package(self.output)[RELS])
        self.assertTrue(any(r.get('Target')=='https://example.invalid/reference' and r.get('TargetMode')=='External' for r in rels))

    def test_arrow_between_nested_group_parts_keeps_transforms_and_native_shapes(self):
        p=presentation();group=p.slides[0].shapes.add_group_shape();group.name='network'
        panel=group.shapes.add_shape(MSO_SHAPE.RECTANGLE,100,100,9000,6000);panel.name='network/panel [ok]'
        nested=group.shapes.add_group_shape();nested.name='nodes'
        node=nested.shapes.add_shape(MSO_SHAPE.OVAL,2000,1000,2000,2000);node.name='network/node [ok]'
        group.rotation=17;p.save(self.base)
        original={s.name:etree.tostring(s._element.spPr) for s in (panel,node)}
        p=presentation();line=p.slides[0].shapes.add_connector(MSO_CONNECTOR.STRAIGHT,500,2000,8000,2000)
        line.name='route';p.save(self.arrow)
        audit=merge(self.base,[('arrows',self.arrow)],self.output,
            between_objects={'arrows':{'route':{'before':['network/node'],'after':['network/panel']}}})
        root=etree.fromstring(read_package(self.output)[SLIDE]);tree=root.find('{'+P+'}cSld/{'+P+'}spTree')
        contents=[[n.get('name') for n in child.iter('{'+P+'}cNvPr')] for child in tree]
        at=lambda name:next(i for i,names in enumerate(contents) if name in names)
        self.assertLess(at(panel.name),at('route'));self.assertLess(at('route'),at(node.name))
        for name,props in original.items():
            native=next(n.getparent().getparent() for n in root.iter('{'+P+'}cNvPr') if n.get('name')==name)
            self.assertEqual(etree.tostring(native.find('{'+P+'}spPr')),props)
            outer=native
            while outer.getparent() is not tree:outer=outer.getparent()
            self.assertEqual(outer.find('{'+P+'}grpSpPr/{'+A+'}xfrm').get('rot'),str(17*60000))
        ids=[n.get('id') for n in root.iter('{'+P+'}cNvPr')];self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(audit['layers'][0]['source_occlusion_placements']),1)
        self.assertTrue(audit['stage_sources_unchanged'])

    def test_conflicting_part_visibility_does_not_hide_visible_arrow(self):
        p=presentation();group=p.slides[0].shapes.add_group_shape();group.name='G'
        for name in ('node','panel'):
            s=group.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,1000,1000);s.name='G/'+name
        p.save(self.base)
        p=presentation();line=p.slides[0].shapes.add_connector(MSO_CONNECTOR.STRAIGHT,0,500,1500,500)
        line.name='route';p.save(self.arrow)
        audit=merge(self.base,[('arrows',self.arrow)],self.output,
            between_objects={'arrows':{'route':{'before':['G/node'],'after':['G/panel']}}})
        self.assertFalse(audit['layers'][0]['source_occlusion_placements'])
        self.assertEqual(audit['layers'][0]['source_occlusion_unresolved'][0]['reason'],'no_slot_preserving_source_visible_parts')

    def test_aspect_ratio_or_missing_relationship_fails_without_publishing(self):
        p=presentation(10,10);p.save(self.arrow)
        with self.assertRaisesRegex(ValueError,'aspect ratios'):merge(self.base,[('arrows',self.arrow)],self.output)
        self.assertFalse(self.output.exists())
        p=presentation();p.slides[0].shapes.add_textbox(0,0,1000,1000).text='x';p.save(self.text)
        package=read_package(self.text);root=etree.fromstring(package[SLIDE]);root.find('.//{'+P+'}sp').set('{'+R+'}id','missing')
        package[SLIDE]=etree.tostring(root)
        with ZipFile(self.text,'w',ZIP_DEFLATED) as z:
            for name,data in package.items():z.writestr(name,data)
        with self.assertRaisesRegex(ValueError,'Missing slide relationship'):merge(self.base,[('text',self.text)],self.output)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
