import threading
import fitz
import pytest

from shape_integration.texts.local_content import validate, collisions, conflict_groups
from shape_integration.texts.equations import measure, NS, word_math
from shape_integration.texts.omml import latex_to_omml


def row(id='T1', **kw):
    return dict(id=id, text='Mask', bbox_2d=[10,10,30,80], font='Times New Roman',
                color='#000000', bold=False, italic=False, rotation_deg=90,
                legibility='clear', runs=[{'text':'Mask','script':'normal'}],
                kind='text', latex='', **kw)


def proposal(id='T1'):
    return dict(id=id, status='located', text='Attention', kind='text', latex='',
                runs=[{'text':'Attention','script':'normal'}], bbox_2d=[12,12,28,78],
                rotation_deg=90, direction_status='confirmed', reason='Visible Attention in left column')


def test_joint_review_can_correct_content_not_only_geometry():
    value=validate({'items':[proposal()]},[row()],[100,100])
    assert value['items'][0]['text']=='Attention'
    assert value['items'][0]['runs'][0]['text']=='Attention'


def test_two_ids_cannot_claim_same_occurrence():
    with pytest.raises(ValueError,match='overlapping glyphs'):
        validate({'items':[proposal(),proposal('T2')]},[row(),row('T2')],[100,100])


def test_latest_runs_override_redundant_text():
    p=proposal();p['runs']=[{'text':'Mask','script':'normal'}]
    result=validate({'items':[p]},[row()],[100,100])
    assert result['items'][0]['text']=='Mask'


def test_unicode_script_display_matches_structured_runs():
    p=proposal();p.update(text='z¹₂',kind='formula',latex=r'z^{1}_{2}',
        runs=[{'text':'z','script':'normal'},{'text':'1','script':'super'},{'text':'2','script':'sub'}])
    v=validate({'items':[p]},[row()],[100,100])['items'][0]
    assert set(v['text'])==set('z12') and v['latex']==r'z^{1}_{2}'


def test_collision_components_and_disjoint_neighbors():
    rows=[row(),row('T2'),row('T3')];rows[2]['bbox_2d']=[40,10,60,80]
    assert collisions(rows)==[['T1','T2']]
    assert [[r['id'] for r in g] for g in conflict_groups(rows,collisions(rows))]==[['T1','T2']]


def test_arg_operator_converted_not_literal_command():
    tree=latex_to_omml(r'\theta^{*}=\arg\min_{\theta}x')
    text=''.join(tree.xpath('.//m:t/text()',namespaces=NS))
    assert 'arg' in text and '\\' not in text
    # Native presentation math must not receive the preview-only quoting.
    assert tree.find('.//m:nor',NS) is None
    preview=word_math({'latex':r'=x','color':'#000000'})
    assert preview.find('.//m:nor',NS) is not None


def test_real_render_preserves_equals_star_and_arg(tmp_path):
    rows=[{'id':'leading','latex':r'=\frac{1}{m}','color':'#000000'},
          {'id':'argmin','latex':r'\theta^{*}=\arg\min_{\theta}x','color':'#000000'}]
    measure(rows,tmp_path,threading.BoundedSemaphore(1))
    with fitz.open(tmp_path/'black/black.pdf') as doc:
        text=[p.get_text() for p in doc]
    assert text[0].lstrip().startswith('=')
    assert '*' in text[1] and 'arg' in text[1]
    assert all('¿' not in t for t in text)
