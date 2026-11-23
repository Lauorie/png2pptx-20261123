from copy import deepcopy

import pytest
from PIL import Image, ImageDraw, ImageFont

from shape_integration.texts.render_feedback import evidence,gate,replace,validate_review,proposals,prompt,new_collisions


def row(ident='T1', box=None):
    return {'id':ident,'text':'Label','bbox_2d':box or [10,10,90,40],
            'font':'Arial','bold':False,'italic':False,'underline':False,'color':'#000000',
            'kind':'text','latex':'','runs':[{'text':'Label','script':'normal'}],
            'rotation_deg':0,'legibility':'clear'}


FRAME={'size':[200,100],'source_size':[200,100],'box':[0,0,200,100],'scale':1,'offset':[0,0]}


def test_keep_cannot_edit_and_extra_ids_are_rejected():
    source=Image.new('RGB',(200,100),'white')
    raw={'items':[{'id':'T1','decision':'keep','reason':'matches','issue_types':[],'units':[]}]}
    assert validate_review(raw,[row()],FRAME,source)['revisions']==[]
    raw['items'][0]['units']=[row()]
    with pytest.raises(ValueError,match='must not change'):validate_review(raw,[row()],FRAME,source)
    raw['items'][0]['id']='T2'
    with pytest.raises(ValueError,match='Every target'):validate_review(raw,[row()],FRAME,source)


def test_layout_edit_cannot_drop_a_line_even_if_pixels_would_improve():
    original=row();original['text']='Label 2019.'
    source=Image.new('RGB',(200,100),'white')
    unit={**row(),'direction_status':'confirmed','reason':'first baseline','visual_line_count':1}
    raw={'items':[{'id':'T1','decision':'revise','reason':'two baselines','issue_types':['wrapping'],'units':[unit]}]}
    with pytest.raises(ValueError,match='preserve ALL original characters'):
        validate_review(raw,[original],FRAME,source)
    second={**unit,'text':'2019.','runs':[{'text':'2019.','script':'normal'}],'bbox_2d':[10,55,60,75]}
    raw['items'][0]['units'].append(second)
    result=validate_review(raw,[original],FRAME,source)
    assert [p['text'] for p in result['revisions'][0]['parts']]==['Label','2019.']


def test_proposal_does_not_carry_old_measurement_and_does_not_mutate_neighbors():
    original=row();original['layout']={'old':True};neighbor=row('T2',[100,10,180,40])
    before=deepcopy([original,neighbor])
    revised={**row(),'bbox_2d':[12,12,92,42],'status':'located'}
    updates=proposals({'revisions':[revised]},[original],FRAME)
    assert 'layout' not in updates['T1'][0]
    updated=replace([original,neighbor],updates)
    assert updated[1]==neighbor and [original,neighbor]==before
    assert updated[0]['bbox_2d']==[12,12,92,42]
    revised['bbox_2d']=[130,10,190,40]
    with pytest.raises(ValueError,match='different occurrence'):proposals({'revisions':[revised]},[original],FRAME)


def glyph_image(offset=0,color='#000000'):
    im=Image.new('RGB',(200,100),'white')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    ImageDraw.Draw(im).text((12+offset,12),'Label',font=font,fill=color)
    return im


def test_fixed_support_rewards_alignment_and_rejects_regression():
    source=glyph_image();shifted=glyph_image(4)
    good=evidence(source,shifted,source,[row()])
    assert gate(good)[0]
    bad=evidence(source,source,shifted,[row()])
    assert not gate(bad)[0]
    assert not gate(evidence(source,source,source,[row()]))[0]
    assert not gate(evidence(source,shifted,Image.new('RGB',source.size,'white'),[row()]))[0]


def test_color_change_is_measured_without_changing_foreground_support():
    source=glyph_image(color='#1255CC');wrong=glyph_image(color='#CC5512')
    metric=evidence(source,wrong,source,[row()])
    assert metric['after']['color_agreement_proxy']>metric['before']['color_agreement_proxy']
    assert gate(metric)[0]


def test_wide_old_new_bbox_union_does_not_report_visible_text_as_empty():
    from shape_integration.texts.bbox_color_review import nonregression_gate
    source=Image.new('RGB',(400,200),'white')
    ImageDraw.Draw(source).rectangle((10,10,29,14),fill='black')
    old=row(box=[300,150,390,190]);new=row(box=[10,10,30,15])
    blank=Image.new('RGB',source.size,'white')
    metric=evidence(source,blank,source,[old],[new])
    assert metric['after']['render_mass']>0
    assert metric['after']['score']==pytest.approx(1.)
    assert nonregression_gate(metric)[0]


@pytest.mark.parametrize('score,passed',[(.3658,False),(.75,False),(.79,False),(.80,False),(.800001,True),(.90,True),(float('nan'),False),(float('inf'),False)])
def test_cpu_absolute_threshold_is_strict_even_with_large_improvement(score,passed):
    metric={'reliable':True,'before':{'score':.2,'mask_dice_proxy':.2},
            'after':{'score':score,'mask_dice_proxy':.9,'render_mass':100}}
    assert gate(metric)[0] is passed


def test_retained_low_score_baseline_is_not_marked_qualified():
    from shape_integration.texts.render_feedback import audit_cpu_quality
    result=audit_cpu_quality(glyph_image(),glyph_image(15),[row()])
    assert result['cpu_failed_ids']==['T1']
    assert not result['checks'][0]['cpu_passed']
    assert not result['model_approval_implied']


def test_new_collision_rejected_but_preexisting_overlap_not_invented():
    before=[row(),row('T2',[100,10,180,40])]
    assert not new_collisions(before,before)
    assert new_collisions(before,[row(),row('T2')])


def test_prompt_distinguishes_text_layer_from_missing_shapes():
    value=prompt([200,100],[row()],[])
    assert 'Missing graphics/backgrounds are expected' in value['evidence']['RENDER']
    assert 'PowerPoint verification' in value['evidence']['formula_preview']
    assert value['response_format']['items'][0]['decision']=='keep/revise/uncertain'


def test_real_pdf_crop_retains_same_pixel_grid_and_provenance(tmp_path):
    import fitz
    import json
    from shape_integration.texts.render_feedback import render_evidence
    pdf=tmp_path/'native.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=200,height=100)
        page.insert_text((20,30),'Label',fontsize=16)
        doc.save(pdf)
    frame={**FRAME,'box':[0,0,100,50],'size':[400,200],'scale':4,'offset':[0,0]}
    render_evidence(pdf,Image.new('RGB',(200,100),'white'),frame,tmp_path)
    with Image.open(tmp_path/'render.png') as im:
        assert im.size==(400,200)
        assert min(im.convert('L').getdata())<100
    metadata=json.loads((tmp_path/'render_provenance.json').read_text())
    assert metadata['path']==str(pdf.resolve()) and metadata['page_index']==0


def test_dual_gate_requires_every_cpu_and_actual_render_model_verdict():
    from shape_integration.texts.render_feedback import VERIFY_CRITERIA,dual_gate,validate_verification
    passed={'id':'T1','decision':'pass','checks':dict.fromkeys(VERIFY_CRITERIA,'pass'),
            'improvement':'pass','reason':'actual glyphs match source without losing suffix'}
    cpu=[{'id':'T1','passed':True}]
    assert dual_gate(cpu,[passed],['T1'])
    assert not dual_gate([{'id':'T1','passed':False}],[passed],['T1'])
    assert not dual_gate(cpu,[],['T1'])
    assert not dual_gate(cpu,[passed,passed],['T1'])
    for decision in ('fail','uncertain'):
        rejected={**passed,'decision':decision}
        assert not dual_gate(cpu,[rejected],['T1'])
    for field in VERIFY_CRITERIA:
        for status in ('fail','uncertain'):
            contradictory={**passed,'checks':{**passed['checks'],field:status}}
            assert not dual_gate(cpu,[contradictory],['T1'])
            with pytest.raises(ValueError,match='all criteria'):
                validate_verification({'items':[contradictory]},['T1'])
    assert not dual_gate(cpu,[{**passed,'improvement':'uncertain'}],['T1'])
    with pytest.raises(ValueError,match='read-only'):
        validate_verification({'items':[{**passed,'units':[row()]}]},['T1'])


def test_verification_uses_actual_after_render_and_excludes_cpu_score():
    from shape_integration.texts.render_feedback import verification_prompt
    value=verification_prompt([200,100],[{'id':'T1'}])
    assert 'actual rendered' in value['task']
    assert 'CPU scores' in ' '.join(value['rules'])
    assert 'uncertain' in value['response_format']['items'][0]['decision']
    assert 'units' not in value['response_format']['items'][0]


def test_split_may_recover_a_collapsed_line_only_with_enclosing_parent_evidence():
    original=row(box=[15,15,50,20])
    original.update(parent_id='P1',parent_geometry_prior_bbox=[10,10,90,85])
    parts=[row('T1.L01',[15,15,50,20]),row('T1.L02',[15,65,55,75])]
    checked={'revisions':[{'id':'T1','status':'split','parts':parts}]}
    assert len(proposals(checked,[original],FRAME)['T1'])==2
    original.pop('parent_geometry_prior_bbox')
    with pytest.raises(ValueError,match='different occurrence'):proposals(checked,[original],FRAME)
    original['parent_geometry_prior_bbox']=[100,10,190,85]
    with pytest.raises(ValueError,match='different occurrence'):proposals(checked,[original],FRAME)


def test_verifier_receives_actual_distinct_before_and_after_crops(tmp_path):
    import importlib.util
    from pathlib import Path
    from shape_integration.common import read
    from shape_integration.texts import options
    from shape_integration.texts.render_feedback import VERIFY_CRITERIA
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('feedback_runner_test',root/'review_rendered_text.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    source=tmp_path/'source.png';glyph_image().save(source)
    before=tmp_path/'before';after=tmp_path/'after';before.mkdir();after.mkdir()
    glyph_image(4).save(before/'native.png');glyph_image().save(after/'native.png')
    observed=[]
    class FakeRuntime:
        def call(self, folder, stage, system, prompt_value, images, validator, max_tokens):
            observed.append(prompt_value)
            assert stage=='verify' and len(images)==4
            assert Image.open(images[1][1]).size==Image.open(images[2][1]).size
            assert Image.open(images[1][1]).tobytes()!=Image.open(images[2][1]).tobytes()
            assert 'AFTER' in images[2][0]
            return validator({'items':[{'id':'T1','decision':'pass',
                'checks':dict.fromkeys(VERIFY_CRITERIA,'pass'),
                'improvement':'pass','reason':'source alignment improved'}]})
    result,errors=runner.verify_rendered(FakeRuntime(),tmp_path/'verification',source,
        before,after,[row()],{'T1':[row()]},[[row()]],options(read(root/'config.json')),1)
    assert not errors and result[0]['decision']=='pass' and len(observed)==1
