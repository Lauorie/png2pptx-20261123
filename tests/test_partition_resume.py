from pathlib import Path
import fcntl
import json
import threading
import time

import pytest

from full_recovery.partition_resume import validate_partition, case_view, provision_view, all_shape_products
from full_recovery.split_coordinator import acquire_released_slots


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_partition_cannot_overlap_omit_or_move_an_active_case():
    validate_partition(['a','b','c'], [['a'],['b','c']], ['a'])
    for groups, active in (([['a','b'],['b','c']],[]),([['a'],['b']],[]),([['a'],['b','c']],['b'])):
        with pytest.raises(ValueError):validate_partition(['a','b','c'],groups,active)


def test_views_share_only_their_assigned_cases_and_keep_manifests_private(tmp_path):
    source=tmp_path/'source'
    items=[{'id':'a'},{'id':'b'}]
    write(source/'source_manifest.json',{'images':items})
    write(source/'shape/source_manifest.json',{'images':items})
    for name in ('settings.json','pipeline_config.json'):write(source/'shape'/name,{})
    for cid in ('a','b'):
        (source/'texts/cases'/cid).mkdir(parents=True)
    a,b=tmp_path/'a',tmp_path/'b'
    case_view(source,a,['a']);case_view(source,b,['b'])
    assert not (source/'shape/geometry/cases/a').exists()
    provision_view(a);provision_view(b)
    assert not (a/'shape/geometry/cases/b').exists()
    assert not (b/'shape/geometry/cases/a').exists()
    write(a/'shape/geometry/cases/a/result.json',{'ok':True})
    assert (source/'shape/geometry/cases/a/result.json').is_file()
    assert not (source/'shape/geometry/cases/b/result.json').exists()
    write(a/'shape/localization/manifest.json',{'scope':'a'})
    assert not (b/'shape/localization/manifest.json').exists()
    assert not (source/'shape/localization/manifest.json').exists()


def test_drain_waits_for_live_case_and_holds_released_capacity(tmp_path):
    path=tmp_path/'00.lock';path.write_text('live case')
    original=path.open('r');fcntl.flock(original,fcntl.LOCK_EX)
    result=[];progress=[];cancel=threading.Event()
    thread=threading.Thread(target=lambda:result.extend(acquire_released_slots(tmp_path,1,cancel,progress.append)))
    thread.start();time.sleep(.05)
    assert thread.is_alive() and path.read_text()=='live case'
    fcntl.flock(original,fcntl.LOCK_UN);original.close();thread.join(3)
    assert len(result)==1 and progress==[1]
    with path.open('r') as contender:
        with pytest.raises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result[0].close()
    with path.open('r') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_merge_waits_for_every_shape_and_verifies_hashes(tmp_path):
    import hashlib
    pilot=tmp_path/'pilot';source=tmp_path/'source'
    for base,cid in ((pilot,'a'),(source,'b')):write(base/'source_manifest.json',{'images':[{'id':cid}]})
    plan={'pilot_run':str(pilot),'source_run':str(source),'merge_expected':2}
    ppt=tmp_path/'shape.pptx';ppt.write_bytes(b'committed')
    receipt={'pptx':str(ppt),'pptx_sha256':hashlib.sha256(ppt.read_bytes()).hexdigest()}
    write(pilot/'shape/geometry/cases/a/complete.json',receipt)
    assert all_shape_products(plan) is None
    write(source/'shape/geometry/cases/b/complete.json',receipt)
    assert len(all_shape_products(plan))==2
    ppt.write_bytes(b'changed')
    with pytest.raises(ValueError):all_shape_products(plan)
