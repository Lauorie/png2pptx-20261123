from zipfile import ZipFile
from lxml import etree
from pptx import Presentation
from full_recovery.source_ellipsis import replace
from full_recovery.text_product import select_native_text,NS


def mark(ident,dx=0):
    return {'text_id':ident,'text':'...','color':'#000000',
            'dots':[{'center':[x+dx,20],'diameter':2.} for x in (20,30,40)]}


def identities(path):
    with ZipFile(path) as archive:slide=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    return slide.xpath('//*[local-name()="cNvPr"]/@name')


def test_reapply_source_triplet_keeps_one_owner_and_preserves_other_occurrences(tmp_path):
    path=tmp_path/'text.pptx';prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    for ident,text in [('p','...'),('q','...'),('word','word')]:
        shape=slide.shapes.add_textbox(1,2,3,4);shape.name='TXT '+ident;shape.text=text
    prs.save(path)
    replace(path,[mark('p'),mark('q',50)],(120,60))
    replace(path,[mark('p')],(120,60))
    names=identities(path)
    assert len([n for n in names if n.startswith('SOURCE_ELLIPSIS p/')])==3
    assert len([n for n in names if n.startswith('SOURCE_ELLIPSIS q/')])==3
    assert names.count('TXT word')==1


def test_native_source_dots_can_reenter_text_selection_with_same_source_id(tmp_path):
    path=tmp_path/'source.pptx';output=tmp_path/'selected.pptx'
    prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    for ident in ['p','q']:
        shape=slide.shapes.add_textbox(1,2,3,4);shape.name='TXT '+ident;shape.text='...'
    prs.save(path);replace(path,[mark('p'),mark('q',50)],(120,60));original=path.read_bytes()
    select_native_text(path,output,[{'id':'p'}])
    assert path.read_bytes()==original
    assert sorted(n for n in identities(output) if n.startswith('SOURCE_ELLIPSIS '))==[
        'SOURCE_ELLIPSIS p/0','SOURCE_ELLIPSIS p/1','SOURCE_ELLIPSIS p/2']
    replace(output,[mark('p')],(120,60))
    assert len([n for n in identities(output) if n.startswith('SOURCE_ELLIPSIS ')])==3


def test_two_inventory_ids_for_one_source_triplet_share_one_physical_owner(tmp_path):
    from PIL import Image,ImageDraw
    from full_recovery.source_ellipsis import observe
    source=Image.new('RGB',(120,60),'white');draw=ImageDraw.Draw(source)
    for x in (20,30,40):draw.rectangle((x,20,x+1,21),fill='black')
    rows=[{'id':ident,'text':text,'bbox_2d':[15,15,45,25],'color':'#000000'}
          for ident,text in [('p','...'),('alias','…')]]
    marks=observe(source,rows)
    assert len(marks)==1 and marks[0]['duplicate_text_ids']==['alias']
    prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    for row in rows:
        shape=slide.shapes.add_textbox(1,2,3,4);shape.name='TXT '+row['id'];shape.text=row['text']
    path=tmp_path/'text.pptx';prs.save(path);replace(path,marks,source.size)
    names=identities(path)
    assert not any(n.startswith('TXT ') for n in names)
    assert len([n for n in names if n.startswith('SOURCE_ELLIPSIS ')])==3
    # An independent second source triplet remains a separate occurrence.
    for x in (70,80,90):draw.rectangle((x,20,x+1,21),fill='black')
    rows.append({'id':'other','text':'...','bbox_2d':[65,15,95,25],'color':'#000000'})
    marks=observe(source,rows)
    assert len(marks)==2 and marks[1]['text_id']=='other'


def test_measured_dots_do_not_inherit_presentation_shadow_or_bevel(tmp_path):
    path=tmp_path/'dots.pptx';prs=Presentation();prs.slides.add_slide(prs.slide_layouts[6]);prs.save(path)
    replace(path,[mark('punctuation')],(120,60))
    with ZipFile(path) as archive:
        slide=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    shapes=slide.xpath('//*[local-name()="sp"]')
    assert len(shapes)==3
    for shape in shapes:
        assert not shape.xpath('./*[local-name()="style"]')
        effects=shape.xpath('./*[local-name()="spPr"]/*[local-name()="effectLst"]')
        assert len(effects)==1 and not len(effects[0])
