from copy import deepcopy

import pytest

from shape_integration.texts import FONTS
from shape_integration.texts.occurrence_dedup import deduplicate, same_occurrence


def row(text, box, ident='T1'):
    return dict(id=ident,text=text,bbox_2d=box,kind='text',rotation_deg=0,
                color='#000000',runs=[{'text':text,'script':'normal'}])


def test_monospace_supported():
    assert 'Courier New' in FONTS


def test_flow_repair_requires_full_strong_local_word():
    from shape_integration.texts.registration import supported_flow_axis_repair
    r=row('Block',[10,10,30,120]);r['rotation_deg']=-90
    candidate=dict(score=.9,flow_coverage=.4,cross_axis_ratio=.8,bbox=[12,30,28,70])
    assert supported_flow_axis_repair(r,candidate)
    assert not supported_flow_axis_repair(r,{**candidate,'score':.7})
    assert not supported_flow_axis_repair(r,{**candidate,'bbox':[40,30,56,70]})
    assert not supported_flow_axis_repair({**r,'kind':'formula'},candidate)


def test_long_underscore_variant_reuses_physical_identity():
    a=row('calculate Salign',[10,10,210,35]);b=row('calculate S_align',[16,12,200,29],'T2')
    a['runs']=[{'text':'calculate S','script':'normal'},{'text':'align','script':'sub'}]
    before=deepcopy([a,b]);rows,audit=deduplicate([a,b])
    assert len(rows)==1 and rows[0]['id']=='T1'
    assert rows[0]['text']=='calculate S_align' and audit[0]['removed_id']=='T2'
    assert before==[a,b]


@pytest.mark.parametrize('text,box',[('calculate S_align',[300,10,490,35]),
                                    ('calculate S_along',[10,10,210,35]),
                                    ('calculate',[10,10,210,35])])
def test_distinct_occurrences_and_content_not_merged(text,box):
    assert not same_occurrence(row('calculate Salign',[10,10,210,35]),row(text,box))


def test_math_and_short_label_not_fuzzy_merged():
    assert not same_occurrence(row('A_1',[0,0,20,20]),row('A1',[0,0,20,20]))
    a=row('calculate Salign',[0,0,200,30]);a.update(kind='formula',latex='S_{align}')
    assert not same_occurrence(a,row('calculate S_align',[0,0,200,30]))

