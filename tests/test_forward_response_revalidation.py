"""A validator repair can reuse the exact complete reply without new inference."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiment/shape_sam_guided'))
spec=importlib.util.spec_from_file_location('revalidation_runtime',ROOT/'experiment/shape_sam_guided/runtime.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_complete_rejected_response_revalidated_without_model_call(tmp_path):
    calls=[]
    def post(*args,**kwargs):
        calls.append(1)
        return {'choices':[{'finish_reason':'stop','message':{'content':'{"new_objects":["frame"],"disputed":true}'}}]}
    models=SimpleNamespace(config={'glm_max_tokens':100,'seed':0,'glm_url':'unused'},
                           slots=threading.BoundedSemaphore(1),post=post)
    runtime=module.ForwardRuntime(models,{'shape_schema_retries':0},None)
    def strict(raw):raise ValueError('A conflicting coverage claim')
    with pytest.raises(RuntimeError,match='conflicting'):
        runtime.call(tmp_path,'coverage','system',{},[],strict,100)
    folder=tmp_path/'coverage_00';error=(folder/'error_0.json').read_bytes()
    def isolate(raw):return {**raw,'status':'uncertain'}
    result=runtime.call(tmp_path,'coverage','system',{},[],isolate,100)
    assert result['new_objects']==['frame'] and result['status']=='uncertain'
    assert calls==[1] and (folder/'error_0.json').read_bytes()==error
    receipt=json.loads((folder/'response_revalidation.json').read_text())
    assert receipt['model_calls']==0
    assert runtime.call(tmp_path,'coverage','system',{},[],isolate,100)==result
    with pytest.raises(ValueError,match='identity changed'):
        runtime.call(tmp_path,'coverage','system',{'different':True},[],isolate,100)


@pytest.mark.parametrize('finish,content',[('length','{}'),('stop','invalid JSON')])
def test_incomplete_or_malformed_response_is_not_revalidated(tmp_path,finish,content):
    module.save(tmp_path/'error_0.json',{'type':'ValueError','error':'bad'})
    module.save(tmp_path/'response_0.json',{'choices':[{'finish_reason':finish,'message':{'content':content}}]})
    assert module.revalidate_response(tmp_path,tmp_path,'coverage',lambda raw:raw) is None
    assert not (tmp_path/'result.json').exists()
