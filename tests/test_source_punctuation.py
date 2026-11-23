from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from full_recovery.source_punctuation import observe


def fixture():
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    for y in (35,45,55):draw.line((50,y-2,50,y+2),fill='black',width=1)
    rows=[{'id':'target','text':'⋮','color':'#000000','bbox_2d':[47,30,53,60]}]
    scene={'parts':{}}
    styles={'parts':{}}
    arrows={'arrows':[{'id':'route','start':[50,25],'end':[50,65],
                      'style':{'color':'#000000','width_px':1,'dash':'dash'}}]}
    return image,rows,scene,styles,arrows


def test_original_dashes_already_rendered_by_named_native_route_retire_only_duplicate_text():
    source,rows,scene,styles,arrows=fixture()
    before=deepcopy((rows,scene,styles,arrows))
    selected,receipts=observe(source,source,rows,scene,styles,arrows,{})
    assert not selected and receipts[0]['action']=='retire_duplicate_text'
    assert receipts[0]['replacement_refs']==['arrow:route']
    assert len(receipts[0]['components'])==3
    assert (rows,scene,styles,arrows)==before


def test_continuous_line_through_source_dashes_cannot_witness_glyph_deletion():
    source,rows,scene,styles,arrows=fixture()
    native=Image.new('RGB',source.size,'white')
    ImageDraw.Draw(native).line((50,20,50,70),fill='black')
    selected,receipts=observe(source,native,rows,scene,styles,arrows,{})
    assert selected==rows and receipts[0]['action']=='preserve'


@pytest.mark.parametrize('mode',['displaced','wrong_pigment','missing_part','wrong_extent'])
def test_source_component_position_pigment_count_and_extent_are_required(mode):
    source,rows,scene,styles,arrows=fixture()
    native=Image.new('RGB',source.size,'white');draw=ImageDraw.Draw(native)
    for y in ((35,45) if mode=='missing_part' else (35,45,55)):
        x=53 if mode=='displaced' else 50
        draw.line((x,y-2,x,y+2),fill='red' if mode=='wrong_pigment' else 'black',
                  width=5 if mode=='wrong_extent' else 1)
    selected,receipts=observe(source,native,rows,scene,styles,arrows,{})
    assert selected==rows and receipts[0]['action']=='preserve'


@pytest.mark.parametrize('plan',[{'drop_arrow_names':['route']},{}])
def test_matching_preview_without_surviving_named_replacement_preserves_text(plan):
    source,rows,scene,styles,arrows=fixture()
    if not plan:arrows={'arrows':[]}
    selected,receipts=observe(source,source,rows,scene,styles,arrows,plan)
    assert selected==rows
    assert receipts[0]['reason']=='no_surviving_native_reference_for_every_component'


def test_filled_shape_fragments_can_supply_the_same_physical_replacement():
    source,rows,scene,styles,arrows=fixture()
    for i,y in enumerate((35,45,55)):
        key=f'part{i}'
        scene['parts'][key]={'id':key,'object_id':'owner','status':'ok',
            'commands':[['M',49,y-2],['L',51,y-2],['L',51,y+2],['L',49,y+2],['Z']]}
        styles['parts'][key]={'style':{'fill':'#000000','stroke':None}}
    selected,receipts=observe(source,source,rows,scene,styles,{'arrows':[]},{})
    assert not selected
    assert receipts[0]['replacement_refs']==['shape:part0','shape:part1','shape:part2']
    selected,_=observe(source,source,rows,scene,styles,{'arrows':[]},{'drop_shape_keys':['part1']})
    assert selected==rows


def test_inline_punctuation_and_other_labels_do_not_enter_native_retirement():
    source,rows,scene,styles,arrows=fixture()
    rows[0]['text']='A ⋮ B'
    selected,receipts=observe(source,source,rows,scene,styles,arrows,{})
    assert selected==rows and not receipts


@pytest.mark.parametrize('hypothesis',['L','7','+'])
def test_single_glyph_uses_actual_ink_and_native_reference_instead_of_its_spelling(hypothesis):
    source=Image.new('RGB',(100,100),'white')
    vertices=[(45,30),(49,30),(49,55),(60,55),(60,59),(45,59)]
    ImageDraw.Draw(source).polygon(vertices,fill='black')
    rows=[{'id':'label','text':hypothesis,'color':'#000000','bbox_2d':[44,29,61,60]}]
    scene={'parts':{'glyph':{'id':'glyph','object_id':'badge','status':'ok',
        'commands':[['M',*vertices[0]],*[['L',*p] for p in vertices[1:]],['Z']]}}}
    styles={'parts':{'glyph':{'style':{'fill':'#000000','stroke':None}}}}
    selected,receipts=observe(source,source,rows,scene,styles,{'arrows':[]},{})
    assert not selected and receipts[0]['replacement_refs']==['shape:glyph']


def test_source_covered_native_occurrence_can_dominate_a_misplaced_glyph():
    source=Image.new('RGB',(100,100),'white');ImageDraw.Draw(source).rectangle((48,30,50,60),fill='black')
    native=Image.new('RGB',source.size,'white');ImageDraw.Draw(native).rectangle((47,30,51,60),fill='black')
    glyph=Image.new('RGB',source.size,'white');ImageDraw.Draw(glyph).rectangle((53,30,55,60),fill='black')
    rows=[{'id':'target','text':'I','color':'#000000','bbox_2d':[43,28,58,62]}]
    scene={'parts':{'stem':{'id':'stem','object_id':'owner','status':'ok',
        'commands':[['M',47,30],['L',51,30],['L',51,60],['L',47,60],['Z']]}}}
    styles={'parts':{'stem':{'style':{'fill':'#000000','stroke':None}}}}
    selected,receipts=observe(source,native,rows,scene,styles,{'arrows':[]},{},glyph_native=glyph)
    assert not selected and receipts[0]['rendered_source_arbitration']['native_dominates']
    # The same heavier native stem cannot defeat an already correct glyph.
    selected,receipts=observe(source,native,rows,scene,styles,{'arrows':[]},{},glyph_native=source)
    assert selected==rows and not receipts[0]['rendered_source_arbitration']['native_dominates']


def test_padded_neighbor_is_context_and_named_complete_rings_replace_repeated_glyph():
    source = Image.new('RGB', (160,80), 'white')
    draw = ImageDraw.Draw(source)
    scene = {'parts': {}}
    styles = {'parts': {}}
    for index, x in enumerate((45, 70, 95)):
        draw.ellipse((x-5, 35, x+5, 45), outline='#777777', width=2)
        key = 'node' + str(index)
        scene['parts'][key] = {'id':key, 'object_id':'nodes', 'status':'ok',
            'kind':'ellipse', 'bbox_2d':[x-5,35,x+5,45],
            'commands':[['M',x-5,40], ['C',x-5,33,x+5,33,x+5,40],
                        ['C',x+5,47,x-5,47,x-5,40], ['Z']]}
        styles['parts'][key] = {'style':{'fill':None, 'stroke':'#777777', 'width_px':2}}
    draw.line((113,34,113,37), fill='#777777', width=2)
    native = source.copy()
    ImageDraw.Draw(native).rectangle((110,30,118,42), fill='white')
    rows = [{'id':'proposed','text':'○ ○ ○','color':'#777777','bbox_2d':[39,34,101,46]}]
    selected, receipts = observe(source,native,rows,scene,styles,{'arrows':[]},{})
    assert not selected
    assert receipts[0]['source_component_count'] == 3
    assert receipts[0]['replacement_refs'] == ['shape:node0','shape:node1','shape:node2']


def test_empty_holes_and_panel_fill_are_not_foreground_pigment_hypotheses():
    source = Image.new('RGB', (140,70), '#FFF8E5')
    draw = ImageDraw.Draw(source)
    scene = {'parts': {}}
    styles = {'parts': {}}
    for index,x in enumerate((45,70,95)):
        draw.ellipse((x-6,29,x+6,41), outline='#777777', width=2)
        key='node'+str(index)
        scene['parts'][key]={'id':key,'object_id':'nodes','status':'ok',
            'commands':[['M',x-6,35],['C',x-6,27,x+6,27,x+6,35],
                        ['C',x+6,43,x-6,43,x-6,35],['Z']],
            'bbox_2d':[x-6,29,x+6,41]}
        styles['parts'][key]={'style':{'fill':'#FFF8E5','stroke':None,'width_px':0}}
    row={'id':'mark','text':'…','color':'#000000','bbox_2d':[38,28,102,42]}
    selected,receipts=observe(source,source,[row],scene,styles,{'arrows':[]},{})
    assert selected == [row] and receipts[0]['action']=='preserve'
