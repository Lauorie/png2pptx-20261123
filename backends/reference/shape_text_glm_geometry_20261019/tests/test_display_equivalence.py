from copy import deepcopy

import pytest

from shape_integration.texts.contracts import recognition
from shape_integration.texts.display_equivalence import equivalent


def scripted(base, sub, sup):
    return [{'text': base, 'script': 'normal'},
            {'text': sup, 'script': 'super'}, {'text': sub, 'script': 'sub'}]


def test_script_order_does_not_change_formula_meaning():
    assert equivalent('u_i^h', scripted('u', 'i', 'h'))
    assert equivalent('z₁²', scripted('z', '1', '2'))
    assert equivalent('x_{n-2}', scripted('x', 'n−2', ''))
    assert not equivalent('u_i^h', scripted('u', 'h', 'i'))
    assert not equivalent('u_i^h', scripted('v', 'i', 'h'))
    assert not equivalent('u_i_j', scripted('u', 'ij', ''))
    assert not equivalent('x_i+y_j', scripted('x+y', 'ij', ''))


def test_contract_accepts_only_equivalent_formula_without_mutating_input():
    raw = {'texts': [{'text': 'u_i^h', 'kind': 'formula', 'latex': 'u_i^h',
                      'bbox_2d': [1, 1, 30, 30], 'runs': scripted('u', 'i', 'h')}]}
    before = deepcopy(raw)
    result = recognition(raw, (100, 100))
    assert result['texts'][0]['text'] == 'uhi'
    assert result['texts'][0]['latex'] == 'u_i^h'
    assert raw == before
    raw['texts'][0]['runs'] = scripted('u', 'h', 'i')
    with pytest.raises(ValueError, match='concatenation'):
        recognition(raw, (100, 100))
    raw['texts'][0].update(kind='text', runs=scripted('u', 'i', 'h'))
    with pytest.raises(ValueError, match='concatenation'):
        recognition(raw, (100, 100))
