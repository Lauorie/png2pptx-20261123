"""Every reasoning effort the job runner offers is accepted by the text branch."""
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'backends/reference/shape_text_glm_geometry_20261019/src'


@pytest.fixture
def options(monkeypatch: pytest.MonkeyPatch) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    monkeypatch.syspath_prepend(str(SOURCE))
    for name in [n for n in sys.modules if n == 'shape_integration' or n.startswith('shape_integration.')]:
        monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module('shape_integration.texts')
    yield module.options
    for name in [n for n in sys.modules if n == 'shape_integration' or n.startswith('shape_integration.')]:
        sys.modules.pop(name, None)


@pytest.mark.parametrize('effort', ['low', 'medium', 'high'])
def test_each_offered_effort_is_accepted(options, effort):
    assert options({'texts': {'geometry_reasoning_effort': effort}})['geometry_reasoning_effort'] == effort


def test_an_unknown_effort_is_still_rejected(options):
    with pytest.raises(ValueError):
        options({'texts': {'geometry_reasoning_effort': 'extreme'}})
