"""Source cross silhouettes recover independently of text/diagram taxonomy."""
from PIL import Image, ImageDraw
from pptx import Presentation
from lxml import etree

from full_recovery.source_marks import recover, append
from full_recovery.text_coverage import candidates


def fixture():
    source = Image.new('RGB', (200,160),'white'); d = ImageDraw.Draw(source)
    d.line((80,65,100,85),fill='black',width=5);d.line((80,85,100,65),fill='black',width=5)
    native = Image.new('RGB',source.size,'white')
    known = [{'id':'a','bbox_2d':[10,10,30,35]}]
    return source,native,known


def test_isolated_cross_is_native_editable_geometry_and_preserves_existing_text(tmp_path):
    source,native,known = fixture()
    marks = recover(source,native,candidates(source,known,native))
    assert len(marks)==1 and marks[0]['evidence']['four_concavities']
    assert 77<=marks[0]['bbox_2d'][0]<=80 and 100<=marks[0]['bbox_2d'][2]<=103
    path = tmp_path/'text.pptx';prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    text=slide.shapes.add_textbox(100,200,300,400);text.name='TXT old';text.text='unchanged'
    before=etree.tostring(text._element);prs.save(path)
    append(path,marks,source.size)
    shapes=Presentation(path).slides[0].shapes
    assert etree.tostring(shapes[0]._element)==before
    assert shapes[1].name.startswith('SOURCE_MARK_')
    assert shapes[1]._element.xpath('.//a:custGeom') and not shapes[1]._element.xpath('.//a:blip')


def test_shifted_existing_cross_is_not_duplicated():
    source,native,known=fixture();native.paste(source,(4,0))
    assert not recover(source,native,candidates(source,known,native))


def test_arrows_and_incomplete_junctions_do_not_become_source_marks():
    source,native,known=fixture();source=Image.new('RGB',source.size,'white');d=ImageDraw.Draw(source)
    d.polygon([(30,40),(65,40),(65,30),(90,50),(65,70),(65,60),(30,60)],fill='black')
    d.line([(130,70),(150,70),(150,90)],fill='black',width=4)
    assert not recover(source,native,candidates(source,known,native))
