from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from full_recovery.line_group_choice import compare
from shape_integration.common import RequestTimeout, read


def test_failed_batch_preserves_union_and_does_not_discard_other_decisions(tmp_path):
    table, groups, contexts, native = {}, [], {}, {}
    for index in range(7):
        ident = 'g'+str(index)
        nodes = []
        for layer in ['shape', 'arrows']:
            name = ident+':'+layer
            nodes.append(name)
            table[name] = {'id': name, 'layer': layer, 'points': np.array([[1., 1.], [9., 9.]])}
            native[(ident, layer)] = None
        groups.append(([{'id': ident}], nodes))
        contexts[ident] = {}
    class Model:
        config = {'mapping_workers': 1}
        calls = 0
        def for_stage(self, **overrides):
            assert overrides['shape_schema_retries'] == 0
            return self
        def call(self, folder, tag, system, prompt, images, validator, budget):
            self.calls += 1
            if self.calls == 1:
                raise RequestTimeout('simulated total deadline')
            return validator({'groups': [{'id': row['id'], 'status': 'selected', 'keep_branch': 'B2',
                                         'reason': 'Full source path supported'} for row in prompt['groups']]})
    model = Model()
    with patch('full_recovery.line_group_choice.aligned_view', return_value=None):
        choices, calls = compare(groups, table, Image.new('RGB', (10, 10)), native,
            lambda batch, nodes: ([], [(0, 0, 10, 10)]), contexts, tmp_path, model)
    assert calls == model.calls == 2
    assert len(choices) == 7
    assert all(len(choices['g'+str(i)]['keep_ids']) == 2 for i in range(6))
    assert choices['g6']['source_relation'] == 'selected'
    assert len(choices['g6']['keep_ids']) == 1
    assert not read(tmp_path/'summary.json')['ownership_complete']


def test_valid_batch_group_survives_an_invalid_peer_without_another_call():
    from full_recovery.line_group_choice import validate_batch
    prepared={ident:{'aliases':{'C1':'a','C2':'b'},'branches':{'C1':'B1','C2':'B2'},
        'implementations':{'B1':['C1'],'B2':['C2']},'required':[]} for ident in ['good','bad']}
    raw={'groups':[{'id':'good','status':'selected','keep_branch':'B2','reason':'Source supported complete path'},
                   {'id':'bad','status':'selected','keep_branch':'B99','reason':'Invalid branch'}]}
    result=validate_batch(raw,prepared)
    assert result['groups'][0]['keep_ids']==['C2']
    assert result['groups'][1]['keep_ids']==['C1','C2'] and result['groups'][1]['status']=='unresolved'
    assert result['failed_groups'][0]['id']=='bad'
    # Runtime revalidates its saved canonical result on resume.
    assert validate_batch(result,prepared)['groups']==result['groups']

