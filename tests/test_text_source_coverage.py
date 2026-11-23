"""Missing symbols need original ink, semantic confirmation and unique identity."""
import copy
import pytest
from PIL import Image, ImageDraw

from full_recovery.text_coverage import candidates, validate, observe, verify_glyphs
from shape_integration.common import save


def fixture():
    native = Image.new('RGB', (200, 160), 'white')
    ImageDraw.Draw(native).rectangle((5, 5, 194, 154), outline='black', width=2)
    source = native.copy(); draw = ImageDraw.Draw(source)
    draw.line((80, 65, 100, 85), fill='black', width=3)
    draw.line((80, 85, 100, 65), fill='black', width=3)
    rows = [{'id': 'known', 'text': 'A', 'font': 'Arial', 'bbox_2d': [24, 20, 40, 38]}]
    return source, native, rows


def test_enclosing_shape_does_not_claim_an_absent_symbol_in_its_interior():
    source, native, rows = fixture()
    result = candidates(source, rows, native)
    assert len(result) == 1
    assert result[0]['bbox_2d'] == [79, 64, 101, 86]
    assert result[0]['native_coverage'] == 0


def test_source_symbol_already_in_native_or_text_is_not_added_again():
    source, _, rows = fixture()
    assert not candidates(source, rows, source)
    rows.append({'id': 'operator', 'text': '×', 'bbox_2d': [78, 63, 102, 87]})
    assert not candidates(source, rows)


def test_source_components_only_nominate_and_visual_review_can_reject_a_crossing(tmp_path):
    source, native, rows = fixture()
    class Diagram:
        def call(self, folder, tag, system, prompt, images, validator, budget):
            return validator({'missing_text': [], 'other': [{'candidate_id': 'C1',
                'classification': 'diagram', 'reason': 'crossed diagram strokes'}]})
    added, audit = observe(source, native, rows, tmp_path, Diagram(), save)
    assert not added and audit['model_calls'] == 1
    assert not audit['arrow_geometry_changed']


def test_new_typographic_geometry_comes_from_original_ink_not_model_coordinates(tmp_path):
    source, native, rows = fixture()
    class Text:
        def call(self, folder, tag, system, prompt, images, validator, budget):
            return validator({'missing_text': [{'candidate_ids': ['C1'], 'text': '×', 'font': 'Arial',
                'bold': True, 'italic': False, 'reason': 'multiplication operator between source operands',
                'bbox_2d': [999, 999, 1000, 1000]}], 'other': []})
    added, audit = observe(source, native, rows, tmp_path, Text(), save)
    assert len(added) == 1 and added[0]['text'] == '×'
    assert added[0]['bbox_2d'] == [79, 64, 101, 86]
    assert added[0]['id'] not in {r['id'] for r in rows}
    assert audit['unreviewed_candidates'] == 0


def test_every_nominated_ink_component_requires_one_disposition():
    aliases = {'C1': {}, 'C2': {}}
    value = {'missing_text': [], 'other': [{'candidate_id': 'C1', 'classification': 'uncertain', 'reason': 'ambiguous'}]}
    with pytest.raises(ValueError, match='C2'):
        validate(value, aliases, ['Arial'])
    duplicate = copy.deepcopy(value); duplicate['other'].append(duplicate['other'][0])
    with pytest.raises(ValueError):
        validate(duplicate, aliases, ['Arial'])
    partial = validate(value, aliases, ['Arial'], allow_omissions=True)
    assert partial['other'][1]['candidate_id'] == 'C2'
    assert partial['other'][1]['classification'] == 'uncertain'
    assert not partial['missing_text']


def test_neighbouring_words_cannot_be_invented_on_a_diagram_component():
    result=validate({'missing_text':[{'candidate_ids':['C1'],'text':'Core','font':'Arial',
        'bold':False,'italic':False,'reason':'neighbouring word'}],'other':[]},{'C1':{}},['Arial'])
    assert not result['missing_text'] and result['other'][0]['classification']=='uncertain'


def test_single_character_label_requires_complete_unit_evidence():
    proposal={'candidate_ids':['C1'],'text':'R','font':'Arial','bold':False,
              'italic':False,'reason':'Independent source legend label'}
    assert not validate({'missing_text':[dict(proposal)],'other':[]},{'C1':{}},['Arial'])['missing_text']
    proposal['unit']='standalone_label'
    assert validate({'missing_text':[dict(proposal)],'other':[]},{'C1':{}},['Arial'])['missing_text']
    proposal['text']='Road'
    assert not validate({'missing_text':[proposal],'other':[]},{'C1':{}},['Arial'])['missing_text']


def test_added_glyph_requires_matching_original_ink():
    source,native,_=fixture()
    row={'id':'new','text':'×','bbox_2d':[79,64,101,86],'color':'#000000'}
    kept,audit=verify_glyphs(source,source,[row])
    assert kept and audit[0]['source_ink_iou']==1
    kept,audit=verify_glyphs(source,native,[row])
    assert not kept and not audit[0]['accepted']


def test_thin_glyph_raster_phase_does_not_retire_source_text():
    source=Image.new('RGB',(40,40),'white');probe=source.copy()
    ImageDraw.Draw(source).line((20,8,20,30),fill='black',width=1)
    ImageDraw.Draw(probe).line((21,8,21,30),fill='black',width=1)
    row={'id':'label','text':'I','bbox_2d':[19,8,22,30],'color':'#000000'}
    kept,audit=verify_glyphs(source,probe,[row])
    assert kept and audit[0]['source_ink_iou']==0
    assert audit[0]['one_pixel_precision']==1
    shifted=Image.new('RGB',(40,40),'white')
    ImageDraw.Draw(shifted).line((25,8,25,30),fill='black',width=1)
    assert not verify_glyphs(source,shifted,[row])[0]


def test_failed_optional_review_retains_original_text_and_adds_nothing(tmp_path):
    source,native,rows=fixture()
    class Unavailable:
        def call(self,*args,**kwargs):
            raise RuntimeError('symbol service unavailable')
    added,audit=observe(source,native,rows,tmp_path,Unavailable(),save)
    assert not added and audit['uncertain_source_marks']==1
    assert audit['receipts'][0]['review_error']['type']=='RuntimeError'
