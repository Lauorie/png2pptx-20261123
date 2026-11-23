"""Offline checks for paid-stage boundaries and multi-process credentials."""
import copy
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'lib')]
import post_geometry_missing as missing
from transport.common import write_json as write, read_json as read
from transport.credentials import load_credentials
from test_post_geometry_missing import row, FakeModel
from test_bbox_pipeline import BASE, locks


def select_credential(_):
    _, key, auth = load_credentials('http://example.test/v1/')
    return key, auth


@pytest.mark.parametrize('relative',['sam_client.py','checkpoint/upstream/sam_supplement/code/sam_client.py'])
def test_sam_endpoint_file_resolves_without_altering_model(relative,tmp_path):
    endpoint=tmp_path/'endpoint.json';endpoint.write_text(json.dumps({'url':'http://example.test:18103'}))
    spec=importlib.util.spec_from_file_location('sam_endpoint_test',ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=module.SamClient('file://'+str(endpoint),2400)
    assert client.url=='http://example.test:18103' and client.timeout==2400


def test_rotation_shared_across_processes(tmp_path, monkeypatch):
    ring=tmp_path/'ring.json';ring.write_text(json.dumps(['fake-a','fake-b','fake-c']));ring.chmod(0o600)
    monkeypatch.setenv('ARROW_API_KEYRING_FILE',str(ring))
    monkeypatch.setenv('ARROW_API_KEYRING_COUNTER',str(tmp_path/'counter'))
    with ProcessPoolExecutor(max_workers=6) as pool:
        answers=list(pool.map(select_credential,range(90)))
    ordered=sorted(answers,key=lambda x:x[1]['request_sequence'])
    assert [a['request_sequence'] for _,a in ordered]==list(range(90))
    assert [a['key_slot'] for _,a in ordered]==[1,2,3]*30
    assert [key for key,_ in ordered]==['fake-a','fake-b','fake-c']*30
    assert all(not a['key_saved'] and 'fake-' not in json.dumps(a) for _,a in ordered)


def test_sam_and_deferred_still_reviewed_when_global_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(missing,'global_discover',lambda *a:pytest.fail('global discovery called'))
    old=[dict(group='old',state='completed',rows=[row()])]
    sam=tmp_path/'sam/cases/case'
    write(sam/'retained_candidates.json',[dict(id='S1',bbox=[70,40,90,60],member_ids=[])])
    write(sam/'status.json',dict(state='completed',issues=[]))
    cfg={**BASE,'post_geometry_sam_run':str(tmp_path/'sam'),
         'post_geometry_global_discovery_enabled':False,'missing_review_context_enabled':True}
    deferred=[dict(target_id='R1',bbox_native=[25,35,50,55])]
    model=FakeModel();events=[]
    final,issues=missing.run(Image.new('RGB',(100,80),'white'),old,tmp_path/'output',cfg,model,locks(),'case',events.append,deferred)
    checks=[call[1]['candidate']['id'] for call in model.calls if 'candidate' in call[1]]
    assert set(checks)=={'SAM_S1','LOCALIZATION_UNCERTAIN_R1'}
    assert 'global_missing_discovery' not in events
    assert read(tmp_path/'output/audit.json')['global_candidates']==0
    assert len(final)==2 and not issues


def test_runner_finishes_after_reconciliation_without_style(tmp_path, monkeypatch):
    import bbox_inputs,bbox_geometry,pipeline,reconcile_stage,style_pipeline,native_runtime,native_foreground
    from lib import native
    import transport.model
    path=ROOT/'checkpoint/runners/full_pipeline/reconstruct.py'
    spec=importlib.util.spec_from_file_location('isolated_reconstruct',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'loc';out=tmp_path/'cases/case';out.mkdir(parents=True)
    inp=source/'inputs/case/source.png';inp.parent.mkdir(parents=True)
    Image.new('RGB',(100,80),'white').save(inp)
    for name,data in [('status.json',{'state':'completed'}),('result.json',{'state':'completed','issues':[]}),('reconstruction_handoff.json',{'tasks':[]})]:
        write(source/'cases/case'/name,data)
    cfg={**BASE,'source_localization':str(source),'source_controller_run':str(tmp_path),
         'case_workers':1,'api_concurrency':1,'cpu_workers':1,'sam_workers':1,
         'shared_cpu_slot_dir':str(tmp_path/'cpu'),'shared_sam_slot_dir':str(tmp_path/'sam'),
         'stop_after_reconciliation':True,'post_geometry_missing_enabled':False}
    write(tmp_path/'config.json',cfg);write(tmp_path/'cases.json',['case']);write(tmp_path/'manifest.json',{'files':{}})
    monkeypatch.setattr(native_runtime,'validated_renderer',lambda:('fake-renderer','offline'))
    monkeypatch.setattr(transport.model,'Model',lambda *a,**k:type('Fake',(),{'slots':None,'blocked':None})())
    monkeypatch.setattr(bbox_inputs,'normalize',lambda *a:{'tasks':[],'issues':[],'deferred_candidates':[]})
    stages=[]
    def reconcile(clean,initial,*args):
        stages.append('reconcile');return initial,[]
    monkeypatch.setattr(reconcile_stage,'run',reconcile)
    monkeypatch.setattr(style_pipeline,'finish_case',lambda *a,**k:pytest.fail('style called'))
    def publish(clean,rows,dest):
        (dest/'result').mkdir(parents=True,exist_ok=True)
        write(dest/'scene.json',{'size':[100,80],'arrows':[]})
        (dest/'arrows.pptx').write_bytes(b'offline-ppt')
        clean.save(dest/'result/styled.png');clean.save(dest/'result/overlap.png')
        return []
    def render(ppt,dest,size):
        dest.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',size,'white').save(dest/'styled.png')
        (dest/'styled.svg').write_text('<svg/>')
    monkeypatch.setattr(pipeline,'publish',publish)
    monkeypatch.setattr(native,'render',render)
    monkeypatch.setattr(native_foreground,'render_layer',lambda path,size,*a:Image.new('RGBA',size,(0,0,0,0)))
    module.run(tmp_path)
    result=read(out/'completed.json')
    assert stages==['reconcile'] and result['state']=='completed'
    assert result['stop_after']=='reconciliation' and not result['style_recovery_run']
    assert read(out/'progress.json')['stage']=='finished_before_style'
    assert (out/'arrows.pptx').read_bytes()==b'offline-ppt'
    assert not (out/'styles').exists()
