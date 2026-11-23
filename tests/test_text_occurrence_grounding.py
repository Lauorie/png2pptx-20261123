from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from full_recovery.text_occurrences import ground, reconcile_recalls

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def fixture(tmp_path, positions=None, size=20, text='Token', prefix=False):
    positions = positions or [(25,25),(150,25),(25,75),(150,75),(25,125),(150,125)]
    font = ImageFont.truetype(FONT,size);glyph=Image.new('RGBA',(150,60))
    ImageDraw.Draw(glyph).text((3,3),text,font=font,fill='#111111')
    glyph=glyph.crop(glyph.getbbox());template=tmp_path/f'{size}_{text}.png';glyph.save(template)
    source=Image.new('RGB',(450,260),'white');rows=[]
    for i,(x,y) in enumerate(positions):
        source.paste(glyph,(x,y),glyph)
        # The final row contains a progressively biased predicted location.
        dy=0 if i<2 else 5*(i//2)
        rows.append({'id':str(i),'text':text,'kind':'text','font':'Arial','bold':False,
            'italic':False,'color':'#111111','rotation_deg':0,'font_size_px':size,
            'bbox_2d':[x,y+dy,x+glyph.width-1,y+dy+glyph.height-1],
            'runs':[{'text':text,'script':'normal'}],
            'layout':{'measurement':{'ink_template':str(template)}}})
    if prefix:
        ImageDraw.Draw(source).text((275,75),text+'Suffix',font=font,fill='#111111')
    return source,rows,positions,glyph.size


def test_repeated_source_positions_correct_drift_without_losing_instances(tmp_path):
    source,rows,positions,size=fixture(tmp_path);before=deepcopy(rows)
    output,changed,audit=ground(rows,source)
    assert rows==before and len(output)==len(rows)==6 and changed
    assert audit[0]['source_count']==6
    for row,(x,y) in zip(output,positions):
        assert np.allclose(row['bbox_2d'],[x,y,x+size[0]-1,y+size[1]-1],atol=1)
        assert row['text']==before[int(row['id'])]['text']


def test_recall_alias_is_retired_only_at_verified_same_source_ink(tmp_path):
    source,rows,positions,size=fixture(tmp_path);grounded,_,_=ground(rows,source)
    x,y=positions[-1]
    alias={**deepcopy(rows[-1]),'id':'recall','bbox_2d':[x,y,x+size[0]-1,y+size[1]-1],
        'source_coverage':{'complete_label_reference':{'text':'Token'}}}
    output,audit=reconcile_recalls([*grounded,alias],source)
    assert len(output)==6 and audit[0]['owner_id']=='5'
    # Text equality without grounded source identity must not delete a label.
    original,audit=reconcile_recalls([*rows,alias],source)
    assert len(original)==7 and audit==[]


def test_distinct_real_recall_and_longer_word_prefix_are_preserved(tmp_path):
    source,rows,positions,size=fixture(tmp_path,prefix=True)
    output,_,audit=ground(rows,source)
    assert audit[0]['source_count']==6 and audit[0]['state']=='grounded'
    alias={**deepcopy(rows[0]),'id':'separate','bbox_2d':[275,200,275+size[0]-1,200+size[1]-1],
           'source_coverage':{'complete_label_reference':{'text':'Token'}}}
    glyph=Image.open(rows[0]['layout']['measurement']['ink_template']);source.paste(glyph,(275,200),glyph)
    kept,audit=reconcile_recalls([*output,alias],source)
    assert len(kept)==7 and audit==[]


def test_source_count_ambiguity_and_blank_source_abstain(tmp_path):
    source,rows,positions,size=fixture(tmp_path)
    output,_,audit=ground(rows[:-1],source)
    assert len(output)==5 and not any(a['state']=='grounded'for a in audit)
    output,changed,_=ground(rows,Image.new('RGB',source.size,'white'))
    assert output==rows and changed==[]


def test_distinct_typographic_levels_do_not_exchange_positions(tmp_path):
    source,a,pa,sa=fixture(tmp_path,size=12,positions=[(20,20),(20,55),(20,90)])
    other,b,pb,sb=fixture(tmp_path,size=30,positions=[(240,20),(240,80),(240,140)])
    source.paste(other.crop((220,0,450,200)),(220,0))
    for row in b:row['id']='large'+row['id']
    output,_,audit=ground([*a,*b],source)
    assert len(output)==6 and sum(a['state']=='grounded'for a in audit)==2
    assert all(r['bbox_2d'][0]<100 for r in output[:3])
    assert all(r['bbox_2d'][0]>200 for r in output[3:])


def test_rotation_formula_and_scripted_text_are_outside_this_repair(tmp_path):
    source,rows,_,_=fixture(tmp_path);before=deepcopy(rows)
    rows[0]['rotation_deg']=90;rows[1]['kind']='formula';rows[2]['runs'][0]['script']='sub'
    output,_,_=ground(rows[:3],source)
    assert output==rows[:3]


def test_estimated_shade_variation_does_not_split_source_identity(tmp_path):
    source,rows,positions,size=fixture(tmp_path)
    for i,row in enumerate(rows):
        row['color']='#111111'if i<4 else '#333333'
    output,changed,audit=ground(rows,source)
    assert audit[0]['source_count']==6 and len(audit)==1 and changed
    for row,old,(x,y)in zip(output,rows,positions):
        assert row['color']==old['color']
        assert np.allclose(row['bbox_2d'],[x,y,x+size[0]-1,y+size[1]-1],atol=1)


def test_occurrence_position_retains_native_size_and_style():
    from full_recovery.text_occurrences import refit_position
    row={'id':'label','bbox_2d':[20.,30.,60.,42.],'font_size_px':17.,
         'color':'#335577','font':'Arial','rotation_deg':0.,
         'layout':{'measurement':{'width':50.,'height':16.,'reference_size_px':24.}}}
    def fit(r,m,f):
        scale=min((r['bbox_2d'][2]-r['bbox_2d'][0]+1)/m['width'],
                  (r['bbox_2d'][3]-r['bbox_2d'][1]+1)/m['height'])*f
        return {**r,'font_size_px':m['reference_size_px']*scale,'scale':scale}
    output=refit_position(row,fit)
    assert abs(output['font_size_px']-17.)<1e-8
    assert all(output[k]==row[k]for k in ('bbox_2d','font','color','rotation_deg'))


def test_native_repair_removes_only_named_plain_text_and_preserves_parts(tmp_path):
    from pptx import Presentation
    from zipfile import ZipFile
    from lxml import etree
    from full_recovery.text_product import remove_plain_text, NS
    source=tmp_path/'input.pptx';output=tmp_path/'output.pptx'
    deck=Presentation();slide=deck.slides.add_slide(deck.slide_layouts[6])
    for name in ('TXT move','TXT retain','EQ equation','source mark'):
        shape=slide.shapes.add_textbox(10,20,30,40);shape.name=name;shape.text=name
    deck.save(source)
    original=source.read_bytes()
    remove_plain_text(source,output,{'move'})
    with ZipFile(source)as a,ZipFile(output)as b:
        assert a.namelist()==b.namelist()
        for name in a.namelist():
            if name!='ppt/slides/slide1.xml':assert a.read(name)==b.read(name)
        before=etree.fromstring(a.read('ppt/slides/slide1.xml'))
        after=etree.fromstring(b.read('ppt/slides/slide1.xml'))
        assert [etree.tostring(n)for n in before.findall('.//p:sp',NS)[1:]]==[
            etree.tostring(n)for n in after.findall('.//p:sp',NS)]
    assert source.read_bytes()==original
