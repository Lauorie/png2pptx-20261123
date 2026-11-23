from pathlib import Path
from shape_integration.common import Runtime,save,read


def test_a_partial_batch_is_not_reused_as_a_completed_new_run(tmp_path):
    calls=[]
    class Client:
        def __init__(self,root,budget):self.root=Path(root)
        def request_parameters(self):return {'model':'fixture'}
        def call(self,name,system,prompt,images,validator):
            calls.append(name)
            raw={'groups':[],'failed_groups':[{'id':'bad'}] if len(calls)==1 else []}
            result=validator(raw);save(self.root/name/'result.json',raw);return result
        def redact(self,error):return str(error)
    class LocalRuntime(Runtime):
        def client(self,root,budget):return Client(root,budget)
    config={'api_concurrency':1,'shape_schema_retries':0}
    old=tmp_path/'old';new=tmp_path/'new'
    first=LocalRuntime(config,'fixture://same','')
    assert first.call(old/'cases/c','review','inspect',{},[],lambda d:d,100)['failed_groups']
    current=LocalRuntime({**config,'reuse_output_root':str(new),
        'reuse_request_roots':[str(old)]},'fixture://same','')
    assert not current.call(new/'cases/c','review','inspect',{},[],lambda d:d,100)['failed_groups']
    assert len(calls)==2
    assert not (new/'cases/c/review_00/reuse.json').exists()
    assert read(old/'cases/c/review_00/result.json')['failed_groups']


def test_reuse_requires_identical_effective_request_and_current_validation(tmp_path):
    calls=[]
    class Client:
        def __init__(self,root,budget):self.root=Path(root);self.budget=budget
        def request_parameters(self):return {'model':'fixture','max_tokens':self.budget}
        def call(self,name,system,prompt,images,validator):
            calls.append(prompt);raw={'value':len(calls)}
            result=validator(raw);save(self.root/name/'result.json',raw);return result
        def redact(self,error):return str(error)
    class LocalRuntime(Runtime):
        def client(self,root,budget):return Client(root,budget)
    config={'api_concurrency':1,'shape_schema_retries':0}
    source=tmp_path/'source.png';source.write_bytes(b'original image bytes')
    old=tmp_path/'old';new=tmp_path/'new'
    first=LocalRuntime(config,'fixture://same','')
    assert first.call(old/'cases/c/asset','review','inspect',{'target':'icon'},[('source',source)],lambda d:d,100)=={'value':1}
    current=LocalRuntime({**config,'reuse_output_root':str(new),'reuse_request_roots':[str(old)]},'fixture://same','')
    assert current.call(new/'cases/c/asset','review','inspect',{'target':'icon'},[('source',source)],lambda d:d,100)=={'value':1}
    assert len(calls)==1 and (new/'cases/c/asset/review_00/reuse.json').exists()
    for change in ('prompt','pixels','budget','validator'):
        target=tmp_path/change
        current.config['reuse_output_root']=str(target)
        prompt={'target':'different'} if change=='prompt' else {'target':'icon'}
        source.write_bytes(b'changed image bytes' if change=='pixels' else b'original image bytes')
        def validator(d):
            if change=='validator' and d['value']==1:raise ValueError('old answer rejected by current contract')
            return d
        current.call(target/'cases/c/asset','review','inspect',prompt,[('source',source)],validator,101 if change=='budget' else 100)
    assert len(calls)==5
    assert read(old/'cases/c/asset/review_00/result.json')=={'value':1}
