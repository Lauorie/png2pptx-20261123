from shape_integration.styling.model import ask, prompt
from shape_integration.common import RequestTimeout


def test_exact_row_correction_requests_only_missing_ids_with_full_context(tmp_path):
    targets = [{'id': 'P0'}, {'id': 'P1'}]
    class Model:
        def for_stage(self, **overrides):
            return self
        def call(self, folder, tag, system, actual, images, validator, budget):
            if tag == 'critic_00':
                return validator({'checks': [{'id': 'P0', 'verdict': 'accept'}]})
            assert [row['id'] for row in actual['targets']] == ['P1']
            assert [row['id'] for row in actual['read_only_targets']] == ['P0']
            return validator({'checks': [{'id': 'P1', 'verdict': 'accept'}]})
    result = ask(Model(), tmp_path, 'critic', prompt('critic', targets, []), [], ['P0', 'P1'],
                 {'P0': {}, 'P1': {}}, {'protocol_retries': 1, 'max_tokens': 4096})
    assert set(result['valid']) == {'P0', 'P1'} and len(result['events']) == 2


def test_timeout_keeps_valid_rows_and_never_sends_second_request(tmp_path):
    class Model:
        calls = 0
        def for_stage(self, **overrides):
            return self
        def call(self, *args):
            self.calls += 1
            raise RequestTimeout('simulated deadline')
    model = Model()
    result = ask(model, tmp_path, 'critic', prompt('critic', [{'id': 'P0'}], []), [], ['P0'],
                 {'P0': {}}, {'protocol_retries': 1, 'max_tokens': 4096})
    assert model.calls == 1 and not result['valid'] and result['events'][0]['timeout']


def test_isolated_primitive_critic_uses_low_budget_but_overlaps_keep_full_review(tmp_path):
    import copy
    targets=[{'id':'P0','kind':'rectangle','source_bbox':[10,10,60,40],
              'current_style':{'fill':'#FFFFFF','stroke':'#000000','dash':'solid'}}]
    base=prompt('critic',targets,[]);base['overlap_context']=[]
    class Model:
        def for_stage(self,**overrides):self.overrides=overrides;return self
        def call(self,folder,tag,system,actual,images,validator,budget):
            self.budget=budget
            return validator({'checks':[{'id':'P0','verdict':'accept'}]})
    for mode in ('isolated','overlap','transparent','gradient','extra_context'):
        p=copy.deepcopy(base);table={'P0':{}}
        if mode=='overlap':p['overlap_context']=[{'back':'A','front':'B'}]
        if mode=='transparent':p['targets'][0]['current_style']['fill_opacity']=.5
        if mode=='gradient':p['targets'][0]['current_style']['fill']={'type':'linear_gradient'}
        if mode=='extra_context':table['P1']={}
        model=Model();ask(model,tmp_path/mode,'critic',p,[],['P0'],table,
                         {'protocol_retries':0,'max_tokens':100000})
        assert model.budget==(8192 if mode=='isolated' else 100000)
        assert model.overrides.get('reasoning_effort')==('low' if mode=='isolated' else None)
