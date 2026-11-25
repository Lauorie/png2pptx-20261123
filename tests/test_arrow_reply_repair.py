"""Arrow geometry replies with two observed format slips must stay usable.

At low reasoning effort Sage-S2.1-Flash wrote 10 of 45 geometry replies in one
of these forms.  The coordinates were present; only the node naming was off.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from full_recovery.arrow_reply_repair import install, name_positional_nodes, quote_node_ids

GEOMETRY_CODE = Path(__file__).resolve().parents[1]/'third_party/arrow_geometry/geometry_code'
BARE_IDS = ('{"nodes":[[p1,507.0,830.0],[p2,507.0,372.0]],"routes":[{"id":"R1","paths":[{"id":"R1a",'
            '"heads":["none","triangle"],"segments":[{"kind":"line","nodes":["p1","p2"]}]}]}],"status":"draw"}')
POSITIONAL = ('{"nodes":[[583.0,300.0],[230.0,300.0],[230.0,785.0],[275.0,785.0]],"routes":[{"id":"R1","paths":['
              '{"id":"R1P1","heads":["none","none"],"segments":[{"kind":"line","nodes":["n0","n1"]},'
              '{"kind":"line","nodes":["n1","n2"]},{"kind":"line","nodes":["n2","n3"]}]}]}],"status":"draw"}')


def test_bare_node_ids_are_quoted_without_touching_coordinates():
    assert json.loads(quote_node_ids(BARE_IDS))['nodes'] == [['p1', 507.0, 830.0], ['p2', 507.0, 372.0]]


def test_valid_json_and_json_literals_are_left_alone():
    valid = '{"nodes":[["p1",1.0,2.0]],"flags":[true,false,null],"note":"[p1,2]"}'
    assert quote_node_ids(valid) == valid
    broken_elsewhere = '{"nodes":[["p1",1.0,2.0]],"routes":[}'
    assert quote_node_ids(broken_elsewhere) == broken_elsewhere


def test_positional_node_references_get_their_ids_in_table_order():
    answer = name_positional_nodes(json.loads(POSITIONAL))
    assert answer['nodes'] == [['n0', 583.0, 300.0], ['n1', 230.0, 300.0], ['n2', 230.0, 785.0], ['n3', 275.0, 785.0]]
    assert answer['routes'] == json.loads(POSITIONAL)['routes']


def test_one_based_positional_references_are_also_named():
    raw = json.loads(POSITIONAL.replace('"n3"', '"n4"').replace('"n2"', '"n3"').replace('"n1"', '"n2"').replace('"n0"', '"n1"'))
    assert [n[0] for n in name_positional_nodes(raw)['nodes']] == ['n1', 'n2', 'n3', 'n4']


@pytest.mark.parametrize('text', [
    POSITIONAL.replace('"n3"', '"n2"'),      # a node is never referenced: numbering base is not established
    POSITIONAL.replace('"n3"', '"n7"'),      # reference outside the table
    POSITIONAL.replace('"n3"', '"end"'),     # not a positional name
    POSITIONAL.replace('["n0","n1"]', '[[583.0,300.0],[230.0,300.0]]'),  # mixed with the backend's own inline form
    '{"nodes":[["a",1.0,2.0],["b",3.0,4.0]],"routes":[{"id":"R1","paths":[{"segments":[{"nodes":["a","b"]}]}]}]}',
    '{"nodes":[],"routes":[],"status":"empty"}',
    '{"regions":[1,2]}',
])
def test_anything_else_is_returned_unchanged(text):
    raw = json.loads(text)
    assert name_positional_nodes(raw) == json.loads(text)


CHECK = r"""
import sys
code, bare, positional, original = sys.argv[1:5]
sys.path[:0] = [code, original, original + '/lib']
from transport import model
assert model.parse(bare)['nodes'][0] == ['p1', 507.0, 830.0]
assert model.parse(positional)['nodes'][3] == ['n3', 275.0, 785.0]
assert model.parse('{"regions": []}') == {'regions': []}
try:
    model.parse('{"nodes": [')
except ValueError:
    print('INSTALLED')
"""


def test_install_makes_the_frozen_arrow_parser_tolerant(tmp_path):
    code = tmp_path/'geometry_code'
    shutil.copytree(GEOMETRY_CODE/'transport', code/'transport')
    install(code)
    proc = subprocess.run([sys.executable, '-c', CHECK, str(code), BARE_IDS, POSITIONAL, str(GEOMETRY_CODE)],
                          capture_output=True, text=True, timeout=120)
    assert proc.stdout.strip().endswith('INSTALLED'), proc.stderr[-2000:]
    with pytest.raises(ValueError):
        install(code)       # a second installation would double-wrap the parser
