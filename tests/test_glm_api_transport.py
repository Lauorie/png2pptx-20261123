import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment/shape_sam_guided'))
from models import Models
from runtime import ForwardRuntime


class ApiTransportTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(os.environ,{})
        environment.start();self.addCleanup(environment.stop)
        os.environ.pop('GLM_API_RELAY_URL',None)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.key=self.root/'key';self.key.write_text('test-private-credential');self.key.chmod(0o600)
        self.config={'glm_transport':'api','glm_url':'https://example.invalid/v1',
            'glm_api_model':'glm-5.3-flash','glm_api_key_file':str(self.key),
            'glm_model':'api:glm-5.3-flash','glm_concurrency':2,'sam_ports':[],
            'glm_max_tokens':100000,'seed':42}
        self.response={'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}
    def tearDown(self):self.tmp.cleanup()

    def test_api_auth_and_parameters_stay_out_of_traces(self):
        client=Models(self.config)
        body={'model':'local-alias','messages':[],'chat_template_kwargs':{'reasoning_effort':'high','clear_thinking':True}}
        with patch('models.request_json',return_value=self.response) as request:
            client.post(self.config['glm_url']+'/chat/completions',body)
        args,kw=request.call_args
        self.assertIs(args[0],client.api_opener)
        self.assertEqual(kw['headers'],{'Authorization':'Bearer test-private-credential'})
        self.assertEqual(args[2]['model'],'glm-5.3-flash')
        self.assertEqual(args[2]['reasoning_effort'],'high')
        self.assertEqual(args[2]['thinking'],{'type':'enabled'})
        self.assertNotIn('chat_template_kwargs',args[2])
        self.assertIn('chat_template_kwargs',body)

    def test_existing_stream_relay_bypasses_proxy_without_forwarding_credentials(self):
        client=Models(self.config)
        body={'messages':[],'chat_template_kwargs':{'reasoning_effort':'high'}}
        with patch.dict(os.environ,{'GLM_API_RELAY_URL':'http://127.0.0.1:18188/v1'}):
            with patch('models.request_json',return_value=self.response) as request:
                client.post(self.config['glm_url']+'/chat/completions',body)
        args,kw=request.call_args
        self.assertIs(args[0],client.opener)
        self.assertEqual(args[1],'http://127.0.0.1:18188/v1/chat/completions')
        self.assertEqual(kw['headers'],{'X-PPT-Request-Timeout-Seconds':'1200.0'})
        self.assertEqual(kw['attempts'],1)
        self.assertEqual(args[2]['reasoning_effort'],'high')
        self.assertEqual(args[2]['model'],'glm-5.3-flash')

    def test_internal_sam_bypasses_proxy_and_auth(self):
        client=Models(self.config);body={'image':'original.png'}
        with patch('models.request_json',return_value={}) as request:client.post('http://internal:18103/',body)
        args,kw=request.call_args
        self.assertIs(args[0],client.opener);self.assertIs(args[2],body);self.assertNotIn('headers',kw)

    def test_reject_readable_credential(self):
        self.key.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'0600'):Models(self.config).post(self.config['glm_url']+'/chat/completions',{})

    def test_provider_is_in_cache_identity_but_credential_is_not(self):
        runtime=ForwardRuntime(Models(self.config),{'shape_schema_retries':0},None)
        with patch('models.request_json',return_value=self.response):
            out=runtime.call(self.root/'calls','check','Return JSON',{},[],lambda x:x,8192)
        self.assertTrue(out['ok'])
        manifest=(self.root/'calls/check_00/request.json').read_text()
        self.assertIn('api_provider',manifest);self.assertIn('glm-5.3-flash',manifest)
        self.assertNotIn('test-private-credential',manifest);self.assertNotIn('Authorization',manifest)
        with patch.dict(os.environ,{'GLM_API_RELAY_URL':'http://127.0.0.1:18188/v1'}):
            with patch('models.request_json') as request:
                self.assertEqual(runtime.call(self.root/'calls','check','Return JSON',{},[],lambda x:x,8192),out)
            request.assert_not_called()

    def test_local_body_and_opener_preserved(self):
        c={k:v for k,v in self.config.items() if not k.startswith('glm_api_') and k!='glm_transport'}
        client=Models(c);body={'chat_template_kwargs':{'reasoning_effort':'high'}}
        with patch('models.request_json',return_value={}) as request:client.post('http://local/v1/chat/completions',body)
        args,kw=request.call_args
        self.assertIs(args[0],client.opener);self.assertIs(args[2],body);self.assertNotIn('headers',kw)


if __name__=='__main__':unittest.main()
