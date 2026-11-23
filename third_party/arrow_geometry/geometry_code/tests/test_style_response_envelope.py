import json

import pytest

from transport.model import parse


@pytest.mark.parametrize('key', ['styles', 'checks'])
def test_complete_style_answer_with_markdown_and_commentary(key, tmp_path):
    answer = {key: [{'id': 'A1'}]}
    log = tmp_path / 'repair.json'
    text = '```json\n' + json.dumps(answer) + '\n```\n\nVerification: physical endpoints checked.'
    assert parse(text, log) == answer
    assert json.loads(log.read_text())['values_and_coordinates_unchanged']


def test_request_example_is_not_a_style_answer():
    request = {'task': 'estimate', 'response_format': {'styles': []}}
    answer = {'styles': [{'id': 'A1'}]}
    assert parse(json.dumps(request) + '\n' + json.dumps(answer)) == answer


def test_ambiguous_or_incomplete_style_answers_still_fail():
    with pytest.raises(ValueError):
        parse('{"styles": []}\n{"styles": []}')
    with pytest.raises(ValueError):
        parse('```json\n{"styles": [{"id": "A1"')
