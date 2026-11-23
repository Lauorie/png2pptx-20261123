"""Check global-only routing and concurrent request identity, without API calls."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import contracts as C
import global_discovery_prompt as GP
import local_review
import run
import transport.model as transport_model
from frames import global_frame, local_views, visible_global_bounds
from transport.common import write_json


def config():
    return {'api_concurrency': 2, 'base_url': 'http://unused.invalid',
            'model': 'glm-5.3-flash', 'temperature': .1, 'max_tokens': 113000,
            'thinking': {'type': 'enabled'}, 'format_retries': 0, 'transport_retries': 0}


def test_concurrent_override_does_not_mutate_default(tmp_path, monkeypatch):
    seen = []; barrier = threading.Barrier(2)
    image = tmp_path/'input.png'; Image.new('RGB',(10,10),'white').save(image)
    def request(cfg, system, prompt, images, folder):
        barrier.wait(timeout=5)
        seen.append((prompt['name'], system))
        return '{"frame":"GLOBAL","regions":[]}'
    monkeypatch.setattr(transport_model, 'request', request)
    m = transport_model.Model(config(),run_dir=tmp_path,system=C.SYSTEM)
    def call(name, override=None):
        return m.ask(tmp_path/name, {'name':name}, [('CLEAN', image)],
                     lambda a:C.discovery_valid(a,[10,10]), system=override)
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(call,'global',GP.SYSTEM); b=pool.submit(call,'local')
        a.result(); b.result()
    assert dict(seen)=={'global':GP.SYSTEM,'local':C.SYSTEM}
    assert m.system==C.SYSTEM
    assert json.loads((tmp_path/'global/identity.json').read_text())['system']==GP.SYSTEM
    assert json.loads((tmp_path/'local/identity.json').read_text())['system']==C.SYSTEM


def test_changed_system_cannot_reuse_old_request(tmp_path, monkeypatch):
    seen=[]
    image=tmp_path/'input.png'; Image.new('RGB',(10,10),'white').save(image)
    def request(cfg, system, prompt, images, folder):
        seen.append(system); return '{"frame":"GLOBAL","regions":[]}'
    monkeypatch.setattr(transport_model,'request',request)
    m=transport_model.Model(config(),run_dir=tmp_path,system=C.SYSTEM)
    args=(tmp_path/'request',{},[('CLEAN',image)],lambda a:C.discovery_valid(a,[10,10]))
    m.ask(*args); m.ask(*args,system=GP.SYSTEM)
    assert seen==[C.SYSTEM,GP.SYSTEM]
    assert len(list((tmp_path/'request/revisions').glob('*/result.json')))==1


def test_pipeline_changes_global_request_only(tmp_path, monkeypatch):
    cid='test_case'; inp=tmp_path/'inputs'/cid;inp.mkdir(parents=True)
    native=Image.new('RGB',(100,60),'white');native.save(inp/'source.png')
    g=global_frame(native.size);g.image(native).save(inp/'global.png')
    specs=[]
    for vid,f in local_views(native.size,.1):
        f.image(native).save(inp/(vid+'.png'))
        specs.append({'id':vid,'path':vid+'.png','bounds_global':visible_global_bounds(f,g)})
    write_json(inp/'frames.json',{'global_size':list(g.input_size),'native_size':list(native.size),'local_views':specs})
    monkeypatch.setattr(run,'publish',lambda *a:None)
    monkeypatch.setattr(run,'latest',lambda *a:None)
    monkeypatch.setattr(local_review,'run',lambda root,cid,rows,*a:(rows,[]))
    calls=[]
    class Fake:
        def ask(self,folder,prompt,images,validator,**kwargs):
            calls.append((folder,prompt,images,kwargs))
            return validator({'frame':'GLOBAL','regions':[]})
    run.run_case(tmp_path,cid,Fake(),{'local_workers':4,'observation_merge_iou':.95})
    assert len(calls)==5
    for folder,prompt,images,kwargs in calls:
        if folder.name=='global_discovery':
            assert kwargs=={'system':GP.SYSTEM}
            assert prompt==GP.prompt(list(g.input_size))
            assert len(images)==1
        else:
            spec=next(s for s in specs if s['id']==folder.name)
            assert kwargs=={}
            assert prompt==C.local_prompt(list(g.input_size),spec['bounds_global'])
            assert len(images)==2
