from copy import deepcopy
from pathlib import Path
import runpy

import pytest

merge=runpy.run_path(str(Path(__file__).resolve().parents[1]/'retry_saved_review_groups.py'))['merge_result']
pending=runpy.run_path(str(Path(__file__).resolve().parents[1]/'retry_saved_review_groups.py'))['pending_ids']
verify_cache=runpy.run_path(str(Path(__file__).resolve().parents[1]/'retry_saved_review_groups.py'))['verify_response_root']


def test_pending_ids_do_not_reintroduce_already_resolved_targets():
    assert pending({'errors': ["Unresolved local transcription: ['B']"]}, ['A','B']) == {'B'}
    assert pending({'errors': []}, ['A']) == set()
    assert pending({}, ['A']) == {'A'}
    assert pending({'remaining_unresolved_ids': ['B'], 'errors': []}, ['A']) == {'B'}


def test_cache_requires_matching_source_and_group_identity(tmp_path):
    from shape_integration.common import save, sha
    source=tmp_path/'source.png'; source.write_bytes(b'source pixels')
    rows=tmp_path/'input_rows.json'; rows.write_text('[]')
    cache=tmp_path/'cache'; cache.mkdir()
    save(cache/'provenance.json',{'sources':{str(p.resolve()):sha(p) for p in (source,rows)}})
    verify_cache(cache,source,rows)
    rows.write_text('[{}]')
    with pytest.raises(ValueError,match='provenance mismatch'):
        verify_cache(cache,source,rows)


def row(ident,box):
    return {'id':ident,'text':'Label','bbox_2d':box,'kind':'text','latex':'','rotation_deg':0,
            'runs':[{'text':'Label','script':'normal'}],'font':'Arial','color':'#000000',
            'bold':False,'italic':False}


FRAME={'scale':1,'offset':[0,0],'box':[0,0,200,100],'source_size':[200,100]}


def test_selected_review_preserves_unselected_occurrences_and_inputs():
    a=row('A',[10,10,60,30]);b=row('B',[100,10,150,30]);rows=[a,b];before=deepcopy(rows)
    part={**a,'text':'Correct','runs':[{'text':'Correct','script':'normal'}],'status':'located'}
    result,resolved,unresolved=merge(rows,[a],{'items':[part]},FRAME)
    assert next(r for r in result if r['id']=='B')==b
    assert next(r for r in result if r['id']=='A')['text']=='Correct'
    assert resolved=={'A'} and not unresolved
    assert rows==before


def test_selected_review_rejects_neighbor_identity_collision():
    a=row('A',[10,10,60,30]);b=row('B',[100,10,150,30])
    part={**a,'bbox_2d':b['bbox_2d'],'status':'located'}
    with pytest.raises(ValueError,match='collisions'):
        merge([a,b],[a],{'items':[part]},FRAME)


def test_ambiguous_review_does_not_delete_original():
    a=row('A',[10,10,60,30])
    result,resolved,unresolved=merge([a],[a],{'items':[{'id':'A','status':'ambiguous'}]},FRAME)
    assert result==[a] and not resolved and unresolved==['A']
