from copy import deepcopy
import json
import threading
from zipfile import ZipFile

from lxml import etree
import pytest

from shape_integration.texts import options,equations
from shape_integration.texts.numeric_routing import route
from shape_integration.texts.inventory_recovery import partition,recover
from shape_integration.texts.equation_recovery import measure_resilient
from shape_integration.texts.omml import latex_to_omml


def row(text='x',**kw):
    return dict(id='T1',kind='text',text=text,runs=[{'text':text,'script':'normal','color':'#000000'}],
                bbox_2d=[10,10,40,40],color='#000000',font='Times New Roman',bold=False,italic=False,
                rotation_deg=0,legibility='clear',**kw)


@pytest.mark.parametrize('text',['+','>','[',']','−','=','×'])
def test_isolated_literal_routes_but_not_structural_formula(text):
    r=row(text,latex=text);r['kind']='formula'
    assert route(r,'test')['kind']=='text'
    r['part_of_formula']=True
    assert route(r,'test')==r
    r.pop('part_of_formula');r['latex']=r'\sqrt{2}'
    assert route(r,'test')==r


def test_inventory_keeps_good_siblings_and_audits_regions():
    good=row();bad=row('bad');bad['bbox_2d']=[-10,0,0,0]
    raw={'texts':[good,bad],'regions':[{'bbox_2d':[0,0,2,2]}]};before=deepcopy(raw)
    value,rejected,warnings=partition(raw,[100,100],options({}))
    assert len(value['texts'])==1 and rejected[0]['index']==1
    assert len(warnings)==1 and value['regions']==[]
    assert raw==before


def test_inventory_retries_only_rejected_object(monkeypatch,tmp_path):
    from shape_integration.texts import inventory_recovery as ir
    good=row();bad=row('bad');bad['bbox_2d']=[-10,0,0,0]
    path=tmp_path/'content.txt';path.write_text('{}')
    monkeypatch.setattr(ir,'latest_complete',lambda root:({'texts':[good,bad]},path))
    calls=[]
    def ask(runtime,folder,prompt,images,validator,opt):
        calls.append(prompt)
        fixed=deepcopy(bad);fixed['bbox_2d']=[50,10,75,30]
        return validator({'texts':[fixed]})
    monkeypatch.setattr(ir.model,'ask',ask)
    result=recover(tmp_path,[100,100],options({}),None,'bad box')
    assert [r['text'] for r in result['texts']]==['x','bad']
    assert len(calls)==1 and calls[0]['rejected_candidate']==bad


def test_inventory_failed_retry_keeps_valid_objects_with_warning(monkeypatch,tmp_path):
    from shape_integration.texts import inventory_recovery as ir
    bad=row('bad');bad['bbox_2d']=[-10,0,0,0]
    path=tmp_path/'content.txt';path.write_text('{}')
    monkeypatch.setattr(ir,'latest_complete',lambda root:({'texts':[row(),bad]},path))
    def fail(*a,**k):raise RuntimeError('timeout')
    monkeypatch.setattr(ir.model,'ask',fail)
    value=recover(tmp_path,[100,100],options({}),None,'bad')
    assert len(value['texts'])==1
    assert value['recovery_warnings'][0]['status']=='unresolved_no_valid_bbox'


def test_formula_recovery_only_retries_failed_ids(monkeypatch,tmp_path):
    rows=[row('a'),row('b')];rows[1]['id']='T2';calls=[]
    def measure(items,folder,slots,timeout,**kw):
        calls.append(([r['id'] for r in items],kw))
        if not kw.get('literal_preview'):
            kw['failures']['T2']='bad glyph';return {'T1':{'method':'unchanged'}}
        return {'T2':{'method':'literal_preview'}}
    monkeypatch.setattr(equations,'measure',measure)
    values,audit=measure_resilient(rows,tmp_path,None)
    assert values['T1']=={'method':'unchanged'}
    assert calls[1][0]==['T2'] and len(calls)==2 and audit[0]['native_omml_unchanged']


def test_final_preview_fallback_keeps_fraction_in_export(monkeypatch,tmp_path):
    from shape_integration.texts.formula_state import adopt
    from shape_integration.texts.layout import fit
    from shape_integration.texts.export import write
    original=row('ab');original['kind']='formula'
    r=adopt(original,{'latex':r'\frac{a}{\sqrt{b}}'})
    before=deepcopy(r)
    def fail(*a,**kw):raise RuntimeError('renderer unavailable')
    monkeypatch.setattr(equations,'measure',fail)
    result,audit=measure_resilient([r],tmp_path/'measurement',None)
    assert result['T1']['preview_degraded'] and audit[0]['requires_review']
    assert r==before
    fitted=fit(r,result['T1']);write([fitted],[100,100],tmp_path/'text.pptx')
    with ZipFile(tmp_path/'text.pptx') as z:
        tree=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    assert tree.xpath('//*[local-name()="f"]') and tree.xpath('//*[local-name()="rad"]')


def test_literal_preview_does_not_change_native_formula_structure():
    r=row('ab',latex=r'\frac{a}{\sqrt{b}}');r['kind']='formula'
    native=etree.tostring(latex_to_omml(r['latex']))
    preview=equations.word_math(r,literal_preview=True)
    assert preview.xpath('.//*[local-name()="f"]') and preview.xpath('.//*[local-name()="rad"]')
    assert preview.xpath('.//*[local-name()="nor"]')
    assert etree.tostring(latex_to_omml(r['latex']))==native


def test_recovery_seed_is_read_only_and_rejects_other_image(tmp_path):
    from shape_integration.texts.run_recovery import seed
    from shape_integration.common import save,sha
    image=tmp_path/'image';image.write_bytes(b'pixels')
    old=tmp_path/'old';old.mkdir();out=tmp_path/'new';out.mkdir()
    save(old/'provenance.json',{'parameters':{'image_sha256':sha(image),'recognition_mode':'staged'}})
    save(old/'recognition.json',{'texts':[]})
    seed(old,out,image)
    assert (out/'recognition.json').read_bytes()==(old/'recognition.json').read_bytes()
    assert not (out/'provenance.json').exists()
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='image mismatch'):seed(old,out,image)
