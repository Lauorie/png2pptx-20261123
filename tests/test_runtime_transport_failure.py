import json
from pathlib import Path

import pytest

from shape_integration.common import Runtime


def test_incomplete_transport_is_terminal_on_resume_without_schema_replay(tmp_path):
    calls=[]
    class Client:
        def request_parameters(self):return {'max_tokens':100}
        def redact(self,error):return str(error)
        def call(self,name,*args):
            calls.append(name)
            folder=tmp_path/name;folder.mkdir()
            (folder/'error_0.json').write_text(json.dumps({'type':'RuntimeError','error':'HTTP 502',
                                                        'retryable':False,'completed_response':False}))
            raise RuntimeError('HTTP 502')
    runtime=Runtime({'api_concurrency':1,'shape_schema_retries':2},'local','')
    runtime.client=lambda root,budget:Client()
    for _ in range(2):
        with pytest.raises(RuntimeError,match='HTTP 502'):
            runtime.call(tmp_path,'review','system',{},[],lambda x:x,100)
    assert calls==['review_00'] and not (tmp_path/'review_01').exists()
