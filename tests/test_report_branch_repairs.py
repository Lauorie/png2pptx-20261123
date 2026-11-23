from copy import deepcopy

from full_recovery.image_assets import consolidate_nested_assets
from full_recovery.text_diagnostic_repair import repair_typography
from full_recovery.text_recovery_policy import reconcile
from full_recovery.fresh_pipeline import archive_unpublished_requests


def test_nested_registry_conserves_larger_content_and_one_native_claim(tmp_path):
    from PIL import Image
    small_path=tmp_path/'small.png';large_path=tmp_path/'large.png'
    Image.new('RGBA',(10,10),'blue').save(small_path)
    Image.new('RGBA',(30,40),'red').save(large_path)
    small = {'id':'inner','replace_names':['owner/mark'],'registry_ids':['inner-reg'],
             'placement_source_half_open':[10,10,20,20],'replace_arrows':['inside'],
             'path':str(small_path),'edit':{'source_pixels_preserved':True}}
    large = {'id':'outer','replace_names':['owner/mark'],'registry_ids':['outer-reg'],
             'placement_source_half_open':[0,0,30,40],'replace_arrows':['inside'],'path':str(large_path)}
    before = deepcopy([small,large])
    assets, receipts = consolidate_nested_assets([small,large],tmp_path)
    assert len(assets)==1 and assets[0]['id']=='outer'
    assert assets[0]['registry_ids']==['inner-reg','outer-reg']
    assert assets[0]['replace_arrows']==['inside'] and receipts
    assert [small,large]==before
    with Image.open(assets[0]['path']) as image:
        assert image.getpixel((15,15))==(0,0,255,255)
        assert image.getpixel((25,35))==(255,0,0,255)


def test_partial_overlap_and_distinct_occurrences_are_not_collapsed():
    assets = [{'id':'left','replace_names':['a'],'placement_source_half_open':[0,0,20,20]},
              {'id':'right','replace_names':['a'],'placement_source_half_open':[10,0,30,20]},
              {'id':'different','replace_names':['b'],'placement_source_half_open':[0,0,40,40]}]
    assert consolidate_nested_assets(assets)==(assets,[])


def test_unrealized_image_or_different_arrow_cleanup_cannot_take_over():
    small={'id':'small','replace_names':['a'],'replace_arrows':['shaft'],
           'placement_source_half_open':[10,10,20,20]}
    large={'id':'large','replace_names':['a'],'replace_arrows':[],
           'placement_source_half_open':[0,0,30,30]}
    assert consolidate_nested_assets([small,large])==([small,large],[])
    large.update(replace_arrows=['shaft'],pending_edit=True)
    assert consolidate_nested_assets([small,large])==([small,large],[])


def test_typography_warning_rereads_source_once_and_does_not_copy_peer_size():
    old = {'id':'anchor','text':'label','bbox_2d':[10,10,40,20],'font_size_px':30}
    calls=[]
    def job(index, rows, wide):
        calls.append(deepcopy(rows))
        return 'recovery', {'anchor':[{**rows[0],'font_size_px':12}]}, [], ['anchor'], []
    selected,audit=repair_typography([{'id':'anchor','peer_median_px':8}], [old],
        {'anchor':[old]}, [], job, lambda parts,parent:(parts,[]), reconcile)
    assert selected['anchor'][0]['font_size_px']==12 and calls==[[old]]
    assert audit[0]['state']=='source_reread_applied'
    _,audit=repair_typography([{'id':'anchor'}], [old], {'anchor':[old]}, ['anchor'],
                             job, lambda p,k:(p,[]), reconcile)
    assert len(calls)==1 and audit[0]['state']=='needs_review'


def test_unsuccessful_source_diagnostic_preserves_existing_occurrence():
    old={'id':'anchor','bbox_2d':[10,10,40,20]}
    selected={'anchor':[old]}
    result,audit=repair_typography([{'id':'anchor'}],[old],selected,[],
        lambda *a:('group',{},[],[],[]),lambda *a:None,reconcile)
    assert result==selected and audit[0]['state']=='needs_review'


def test_resume_archives_only_unpublished_interruption_and_is_idempotent(tmp_path):
    import json
    incomplete=tmp_path/'staged/global/calls/request_00'
    incomplete.mkdir(parents=True);(incomplete/'request.json').write_text('{}')
    complete=tmp_path/'local/calls/request_00'
    complete.mkdir(parents=True);(complete/'request.json').write_text('{}')
    (complete/'identity.json').write_text('{}');(complete/'result.json').write_text('{"kept":true}')
    before=(complete/'result.json').read_bytes()
    moved=archive_unpublished_requests(tmp_path)
    assert len(moved)==1 and not incomplete.exists()
    assert (complete/'result.json').read_bytes()==before
    assert archive_unpublished_requests(tmp_path)==[]


def test_local_identity_notes_do_not_override_shared_connector_scope():
    from shape_integration.scope import ARROW_OWNERSHIP
    from shape_integration.fitting.semantics import GEOMETRY_RULES
    from shape_integration.fitting.observation import IDENTITY_NOTE_SCOPE
    assert 'without a separately registered arrow ID' in ARROW_OWNERSHIP
    assert '连接路线仍按统一箭头职责' in IDENTITY_NOTE_SCOPE
    assert GEOMETRY_RULES['scope'].count(ARROW_OWNERSHIP)==1
