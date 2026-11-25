"""Text-stage replies that drop the "text" key of a run must stay usable.

Observed with Sage-S2.1-Flash on formula-heavy figures: a run that starts with
punctuation is written as {", y", "script": ...} instead of
{"text": ", y", "script": ...}, which makes the whole inventory unparsable.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from full_recovery.text_json_repair import loads, restore_run_text_keys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT/'backends/reference/shape_text_glm_geometry_20261019'

BROKEN = ('{"texts": [{"kind": "text", "runs": ['
          '{"text": "y", "script": "normal", "color": "#000000"}, '
          '{"text": "1", "script": "sub", "color": "#000000"}, '
          '{", y", "script": "normal", "color": "#000000"}, '
          '{"text": "2", "script": "sub", "color": "#000000"}, '
          '{",....x", "script": "normal", "color": "#000000"}]}]}')


def test_dropped_run_text_keys_are_restored_with_their_exact_values():
    repaired, count = restore_run_text_keys(BROKEN)
    runs = json.loads(repaired)['texts'][0]['runs']
    assert count == 2
    assert [r['text'] for r in runs] == ['y', '1', ', y', '2', ',....x']
    assert all(set(r) == {'text', 'script', 'color'} for r in runs)


def test_loads_accepts_the_broken_reply():
    assert loads(BROKEN)['texts'][0]['runs'][2] == {
        'text': ', y', 'script': 'normal', 'color': '#000000'}


def test_valid_json_is_returned_unchanged_even_when_a_string_looks_broken():
    lookalike = {'texts': [{'runs': [{'text': '{", y", "script": "normal"}', 'script': 'normal'}]}]}
    text = json.dumps(lookalike)
    assert restore_run_text_keys(text) == (text, 0)
    assert loads(text) == lookalike


@pytest.mark.parametrize('text', [
    '{"texts": [',                                   # truncated
    '{"a" "b"}',                                     # missing colon, not a run
    '{"texts": [{", y", "color": "#000000"}]}',      # keyless string, but not followed by script
    '{"texts": [{"runs": [{", y", "script": "normal",}]}]}',  # a second, unrelated defect
])
def test_other_malformed_json_is_still_rejected(text):
    with pytest.raises(json.JSONDecodeError):
        loads(text)


INSTALL_CHECK = r"""
import json, sys
root, backend, broken = sys.argv[1:4]
sys.path[:0] = [backend + '/src', backend + '/vendor', backend, root]
from full_recovery.text_json_repair import install
install()
from shape_integration import transport
from shape_integration.texts import inventory_recovery, staged_formula
assert transport.parse_json('```json\n' + broken + '\n```')['texts'][0]['runs'][2]['text'] == ', y'
for module in (inventory_recovery, staged_formula):
    assert module.json.loads(broken)['texts'][0]['runs'][4]['text'] == ',....x'
    assert module.json.dumps({'a': 1}) == '{"a": 1}'
try:
    transport.parse_json('{"texts": [')
except json.JSONDecodeError:
    print('INSTALLED')
"""


def test_install_makes_the_text_transport_and_cached_reply_readers_tolerant():
    # Runs in its own interpreter with the text stage's import order: the
    # suite also imports an older shape_integration copy from the repo root.
    proc = subprocess.run([sys.executable, '-c', INSTALL_CHECK, str(ROOT), str(BACKEND), BROKEN],
                          capture_output=True, text=True, timeout=120)
    assert proc.stdout.strip().endswith('INSTALLED'), proc.stderr[-2000:]
